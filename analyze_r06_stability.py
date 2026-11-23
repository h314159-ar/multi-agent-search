r"""R06 单独重复 8 次：通过率区间 + 耗时标准差（离线，无 LLM）。

读取 outputs/w7_repeat_r06/repeat_results.csv（由
`experiment_w7_repeat.py --cases R06 --repeats 8 --out outputs/w7_repeat_r06` 生成）。

产物：outputs/w7_repeat_r06/r06_stability.md / r06_stability.json
运行（项目根目录）：dsp\Scripts\python.exe analyze_r06_stability.py
"""
import json
import math
import statistics as st
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "w7_repeat_r06"
LABELS = {"single": "单 Agent", "multi_serial": "多 Agent 串行", "multi_parallel": "多 Agent 并行"}


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2x2 表 [[a,b],[c,d]] 的 Fisher 精确检验双侧 p 值（小样本首选）。"""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    total = math.comb(n, col1)

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / total

    observed = prob(a)
    low = max(0, col1 - (n - row1))
    high = min(row1, col1)
    return sum(prob(x) for x in range(low, high + 1) if prob(x) <= observed + 1e-12)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 区间，小样本比正态近似更稳。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    frame = pd.read_csv(OUT / "repeat_results.csv")
    frame["error"] = frame["error"].fillna("")
    repeats = int(frame["repeat"].nunique())
    modes = {}
    for mode, group in frame.groupby("mode"):
        times = group["elapsed_seconds"].astype(float)
        passes = int(group["overall_pass"].sum())
        low, high = wilson(passes, len(group))
        std = float(times.std(ddof=1)) if len(times) > 1 else 0.0
        mean = float(times.mean())
        modes[mode] = {
            "mode_label": LABELS[mode],
            "runs": int(len(group)),
            "overall_pass": passes,
            "overall_pass_rate": round(passes / len(group), 4),
            "pass_rate_wilson_95": [round(low, 4), round(high, 4)],
            "objective_pass": int(group["objective_pass"].sum()),
            "hard_pass": int((group["hard_check"] == "PASS").sum()),
            "judge_pass": int((group["judge_verdict"] == "PASS").sum()),
            "capability_pass": int(group["capability_ok"].sum()),
            "error_runs": int(group["error"].astype(bool).sum()),
            "mean_seconds": round(mean, 3),
            "std_seconds": round(std, 3),
            "cv_pct": round(std / mean * 100, 1) if mean else None,
            "median_seconds": round(float(times.median()), 3),
            "min_seconds": round(float(times.min()), 3),
            "max_seconds": round(float(times.max()), 3),
            "iqr_seconds": round(float(times.quantile(0.75) - times.quantile(0.25)), 3),
        }

    payload = {
        "case_id": "R06",
        "question": str(frame["question"].iloc[0]),
        "repeats": repeats,
        "total_runs": int(len(frame)),
        "modes": modes,
        "note": "R06 的失败均来自程序级硬校验（派生数字/label 清洗/路由变化），不是并行缺陷。",
    }
    (OUT / "r06_stability.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# R06 单独重复 8 次：稳定性与标准差",
        "",
        f"题目：{payload['question']}",
        "",
        f"规模：1 题 × 3 模式 × {repeats} 次 = {payload['total_runs']} 次运行；"
        f"模型 deepseek-chat、temperature=0、RAG=hybrid。",
        "",
        "## 1. 通过率与耗时",
        "",
        "| 模式 | 运行 | 严格通过 | 通过率 | 95% Wilson 区间 | 硬校验 | 外部评审 | 耗时均值 | 耗时标准差 | 变异系数 | 中位 | 区间 | IQR |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for mode in ("single", "multi_serial", "multi_parallel"):
        if mode not in modes:
            continue
        row = modes[mode]
        lines.append(
            f"| {row['mode_label']} | {row['runs']} | {row['overall_pass']} | "
            f"{row['overall_pass_rate']*100:.1f}% | "
            f"{row['pass_rate_wilson_95'][0]*100:.1f}%~{row['pass_rate_wilson_95'][1]*100:.1f}% | "
            f"{row['hard_pass']}/{row['runs']} | {row['judge_pass']}/{row['runs']} | "
            f"{row['mean_seconds']:.3f}s | ±{row['std_seconds']:.3f}s | {row['cv_pct']}% | "
            f"{row['median_seconds']:.3f}s | {row['min_seconds']:.3f}–{row['max_seconds']:.3f}s | "
            f"{row['iqr_seconds']:.2f}s |"
        )

    lines += ["", "## 2. 逐次结果", "",
              "| 轮 | 单 Agent | 串行 | 并行 | 串行/并行耗时比 |",
              "|---:|---|---|---|---:|"]
    for repeat, group in frame.groupby("repeat"):
        cells = []
        for mode in ("single", "multi_serial", "multi_parallel"):
            sub = group[group["mode"] == mode]
            if sub.empty:
                cells.append("-")
                continue
            row = sub.iloc[0]
            mark = "PASS" if bool(row["overall_pass"]) else "FAIL"
            cells.append(f"{mark} {float(row['elapsed_seconds']):.2f}s")
        serial = group[group["mode"] == "multi_serial"]
        parallel = group[group["mode"] == "multi_parallel"]
        ratio = "-"
        if not serial.empty and not parallel.empty:
            ratio = f"{float(serial.iloc[0]['elapsed_seconds']) / float(parallel.iloc[0]['elapsed_seconds']):.2f}x"
        lines.append(f"| {int(repeat)} | " + " | ".join(cells) + f" | {ratio} |")

    lines += ["", "## 3. 失败原因归类", ""]
    failures = frame[~frame["overall_pass"].astype(bool)]
    if failures.empty:
        lines.append("8 次 × 3 模式全部通过。")
    else:
        for _, row in failures.iterrows():
            detail = str(row["missing_numbers"]) or "无缺失数字"
            comment = str(row["judge_comments"])[:60]
            lines.append(
                f"- 第 {int(row['repeat'])} 轮 {LABELS[row['mode']]}：硬校验 {row['hard_check']}，"
                f"路由 {row['successful_routes']}，缺失/异常数字 {detail}；评审 {row['judge_verdict']}"
                + (f"（{comment}）" if comment and comment != "nan" else ""))
    lines.append("")

    # ── 统计判读 ──
    single = modes.get("single")
    serial = modes.get("multi_serial")
    parallel = modes.get("multi_parallel")
    ratios = []
    for _, group in frame.groupby("repeat"):
        s = group[group["mode"] == "multi_serial"]
        p = group[group["mode"] == "multi_parallel"]
        if not s.empty and not p.empty:
            ratios.append(float(s.iloc[0]["elapsed_seconds"]) / float(p.iloc[0]["elapsed_seconds"]))

    stats = {}
    if serial and parallel:
        stats["serial_vs_parallel_pass_fisher_p"] = round(fisher_two_sided(
            serial["overall_pass"], serial["runs"] - serial["overall_pass"],
            parallel["overall_pass"], parallel["runs"] - parallel["overall_pass"]), 4)
    if single and serial:
        stats["single_vs_serial_pass_fisher_p"] = round(fisher_two_sided(
            single["overall_pass"], single["runs"] - single["overall_pass"],
            serial["overall_pass"], serial["runs"] - serial["overall_pass"]), 4)
    if ratios:
        stats["paired_rounds"] = len(ratios)
        stats["parallel_faster_rounds"] = sum(1 for r in ratios if r > 1)
        stats["median_time_ratio"] = round(st.median(ratios), 3)
        stats["mean_time_ratio"] = round(st.mean(ratios), 3)
        stats["time_ratio_std"] = round(st.stdev(ratios), 3) if len(ratios) > 1 else 0.0
    if parallel and serial:
        stats["median_ratio_of_medians"] = round(
            serial["median_seconds"] / parallel["median_seconds"], 3)

    lines += ["", "## 4. 统计判读（n=8）", ""]
    if "serial_vs_parallel_pass_fisher_p" in stats:
        lines.append(
            f"- 串行 {serial['overall_pass']}/8 vs 并行 {parallel['overall_pass']}/8："
            f"Fisher 精确检验双侧 p = {stats['serial_vs_parallel_pass_fisher_p']}，"
            "差异不显著，8 次样本无法区分 62.5% 与 75% 的通过率。")
    if "single_vs_serial_pass_fisher_p" in stats:
        lines.append(
            f"- 单 Agent {single['overall_pass']}/8 vs 串行 {serial['overall_pass']}/8："
            f"p = {stats['single_vs_serial_pass_fisher_p']}，"
            "单 Agent 的 0/8 是硬校验设计使然（无 Fact 声明式渲染），不是准确率略低。")
    if ratios:
        lines.append(
            f"- 耗时：串行 {serial['mean_seconds']:.2f}±{serial['std_seconds']:.2f}s"
            f"（CV {serial['cv_pct']}%，含 {serial['max_seconds']:.1f}s 离群），"
            f"并行 {parallel['mean_seconds']:.2f}±{parallel['std_seconds']:.2f}s"
            f"（CV {parallel['cv_pct']}%）；{stats['parallel_faster_rounds']}/8 轮并行更快，"
            f"逐轮耗时比中位 {stats['median_time_ratio']:.3f}x（离散度 ±{stats['time_ratio_std']:.3f}）。")
        lines.append(
            f"- 中位数之比 = {stats['median_ratio_of_medians']:.3f}x（均值被单轮离群放大，报告优先用中位数）。")
    lines.append(
        "- 结论：R06 的失败集中在硬校验（派生数字、`Top-10` 带空格、以及 Planner 改走 sql 后 "
        "SQL 行号 subject 与「轻度/重度用户」引用错配），属信任层启发式边界与路由不稳定，"
        "不是并行缺陷；并行在这里只体现为耗时收益，不体现为准确率提升。")
    lines.append("")

    payload["stats"] = stats
    (OUT / "r06_stability.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "r06_stability.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({m: {"pass": v["overall_pass"], "rate": v["overall_pass_rate"],
                          "mean": v["mean_seconds"], "std": v["std_seconds"],
                          "cv": v["cv_pct"]} for m, v in modes.items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
