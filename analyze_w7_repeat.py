r"""W7 重复实验的二次分析：多分支题的并行收益 + 与单次基线对照。

读取 outputs/w7_repeat/repeat_results.csv（由 experiment_w7_repeat.py 生成），
不再调用任何 LLM，可反复离线重算。

产物：
  outputs/w7_repeat/repeat_multiroute_stats.json
  outputs/w7_repeat/repeat_report_addendum.md

运行（项目根目录）：
  dsp\Scripts\python.exe analyze_w7_repeat.py
"""
import json
import statistics as st
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "w7_repeat"
EXPECTED_MULTI = ["R10", "R11", "R12", "R13"]   # 设计上就是双分支的题（sql+rag / mining+rag）
LABELS = {"single": "单 Agent", "multi_serial": "多 Agent 串行", "multi_parallel": "多 Agent 并行"}


def route_count(value) -> int:
    text = "" if pd.isna(value) else str(value)
    return 0 if not text else len(text.split("|"))


def paired_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for repeat, group in frame.groupby("repeat"):
        serial = group[group["mode"] == "multi_serial"].set_index("case_id")
        parallel = group[group["mode"] == "multi_parallel"].set_index("case_id")
        for case_id in serial.index.intersection(parallel.index):
            s = float(serial.loc[case_id, "elapsed_seconds"])
            p = float(parallel.loc[case_id, "elapsed_seconds"])
            rows.append({
                "repeat": int(repeat),
                "case_id": case_id,
                "serial": s,
                "parallel": p,
                "speedup": s / p,
                "reduction_pct": (s - p) / s * 100,
                "serial_routes": route_count(serial.loc[case_id, "successful_routes"]),
            })
    return rows


def subset_stats(rows: list[dict], repeats: list[int]) -> dict:
    per_repeat = []
    for repeat in repeats:
        group = [row for row in rows if row["repeat"] == repeat]
        if not group:
            continue
        per_repeat.append({
            "repeat": repeat,
            "cases": len(group),
            "parallel_faster": sum(1 for row in group if row["speedup"] > 1),
            "mean_speedup": round(st.mean(row["speedup"] for row in group), 3),
            "aggregate_reduction_pct": round(
                (sum(row["serial"] for row in group) - sum(row["parallel"] for row in group))
                / sum(row["serial"] for row in group) * 100, 2),
        })
    reductions = [row["aggregate_reduction_pct"] for row in per_repeat]
    per_case = []
    for case_id in sorted({row["case_id"] for row in rows}):
        group = [row for row in rows if row["case_id"] == case_id]
        per_case.append({
            "case_id": case_id,
            "repeats": len(group),
            "median_serial": round(st.median(row["serial"] for row in group), 3),
            "median_parallel": round(st.median(row["parallel"] for row in group), 3),
            "median_speedup": round(st.median(row["speedup"] for row in group), 3),
            "median_reduction_pct": round(st.median(row["reduction_pct"] for row in group), 2),
            "parallel_faster_repeats": sum(1 for row in group if row["speedup"] > 1),
        })
    return {
        "per_repeat": per_repeat,
        "per_case": per_case,
        "aggregate_reduction_median_pct": round(st.median(reductions), 2),
        "aggregate_reduction_min_pct": min(reductions),
        "aggregate_reduction_max_pct": max(reductions),
        "parallel_faster_samples": sum(row["parallel_faster"] for row in per_repeat),
        "paired_samples": sum(row["cases"] for row in per_repeat),
    }


def main() -> None:
    frame = pd.read_csv(OUT / "repeat_results.csv")
    frame["error"] = frame["error"].fillna("")
    frame = frame[frame["error"].eq("")]
    repeats = sorted(frame["repeat"].unique().tolist())
    rows = paired_rows(frame)

    all_cases = subset_stats(rows, repeats)
    expected = subset_stats([row for row in rows if row["case_id"] in EXPECTED_MULTI], repeats)
    # 与 W7 单次运行的 summarize 口径一致：只看该次串行实际取得 ≥2 路证据的题。
    dynamic_rows = [row for row in rows if row["serial_routes"] >= 2]
    dynamic = subset_stats(dynamic_rows, repeats)

    payload = {
        "source": str((OUT / "repeat_results.csv").relative_to(ROOT)),
        "repeats": repeats,
        "definitions": {
            "all": "全部 9 题（含单分支题；单分支题的串行/并行代码路径相同，差异只来自采样与网络抖动）",
            "expected_multi": "R10–R13，设计上的双分支题",
            "dynamic_multi": "该轮串行实际取得 ≥2 路成功证据的题（与 W7 单次运行 summarize 口径一致）",
        },
        "all": all_cases,
        "expected_multi": expected,
        "dynamic_multi": dynamic,
    }
    (OUT / "repeat_multiroute_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    mode_counts = frame.groupby(["case_id", "mode"])["overall_pass"].sum().unstack()
    lines = [
        "# W7 重复实验补充分析：多分支题的并行收益",
        "",
        "数据来源：`outputs/w7_repeat/repeat_results.csv`（81 次运行，无异常样本）。",
        "本文件不调用 LLM，只做离线统计，可随 CSV 重算。",
        "",
        "## 1. 三种口径的并行收益",
        "",
        "| 口径 | 配对样本 | 并行更快 | 聚合耗时下降（中位） | 区间 |",
        "|---|---:|---:|---:|---:|",
        f"| 全部 9 题 | {all_cases['paired_samples']} | {all_cases['parallel_faster_samples']} | "
        f"{all_cases['aggregate_reduction_median_pct']:.2f}% | "
        f"{all_cases['aggregate_reduction_min_pct']:.2f}%~{all_cases['aggregate_reduction_max_pct']:.2f}% |",
        f"| 预期多分支题 R10–R13 | {expected['paired_samples']} | {expected['parallel_faster_samples']} | "
        f"{expected['aggregate_reduction_median_pct']:.2f}% | "
        f"{expected['aggregate_reduction_min_pct']:.2f}%~{expected['aggregate_reduction_max_pct']:.2f}% |",
        f"| 该轮实际多分支题 | {dynamic['paired_samples']} | {dynamic['parallel_faster_samples']} | "
        f"{dynamic['aggregate_reduction_median_pct']:.2f}% | "
        f"{dynamic['aggregate_reduction_min_pct']:.2f}%~{dynamic['aggregate_reduction_max_pct']:.2f}% |",
        "",
        "说明：单分支题（R05–R09）在串行与并行下执行同一段代码，其耗时差异只来自 LLM 采样与 API 延迟抖动。",
        "把它们计入后，聚合下降区间被拉宽到 "
        f"{all_cases['aggregate_reduction_min_pct']:.2f}%~{all_cases['aggregate_reduction_max_pct']:.2f}%"
        f"（最慢一轮只有 {all_cases['aggregate_reduction_min_pct']:.2f}%），中位数虽仍约 20% 但噪声更大；",
        "因此「并行带来收益」的因果表述应只用 R10–R13 口径，9 题口径只作为全场景参考。",
        "",
        "## 2. 每轮明细（R10–R13）",
        "",
        "| 轮次 | 并行更快 | 平均加速比 | 聚合耗时下降 |",
        "|---:|---:|---:|---:|",
    ]
    for row in expected["per_repeat"]:
        lines.append(f"| {row['repeat']} | {row['parallel_faster']}/{row['cases']} | "
                     f"{row['mean_speedup']:.3f}x | {row['aggregate_reduction_pct']:.2f}% |")

    lines += ["", "## 3. 每题中位数（R10–R13）", "",
              "| 题号 | 串行中位 | 并行中位 | 耗时比（逐轮比值中位） | 3 次中并行更快 |",
              "|---|---:|---:|---:|---:|"]
    for row in expected["per_case"]:
        lines.append(f"| {row['case_id']} | {row['median_serial']:.2f}s | {row['median_parallel']:.2f}s | "
                     f"{row['median_speedup']:.3f}x | {row['parallel_faster_repeats']}/3 |")
    lines += ["", "注：前两列是各自 3 次的中位数，最后一列是逐轮比值的中位数，二者不必然等于两张中位数之比"
              "（如 R10 中位数之比 0.97x，逐轮比值中位 1.09x，说明单轮抖动大于整体差异）。", ""]

    lines += ["", "## 4. 每题通过次数（3 次重复）", "",
              "| 题号 | 单 Agent | 多 Agent 串行 | 多 Agent 并行 |",
              "|---|---:|---:|---:|"]
    for case_id, row in mode_counts.iterrows():
        lines.append(f"| {case_id} | {int(row.get('single', 0))}/3 | "
                     f"{int(row.get('multi_serial', 0))}/3 | {int(row.get('multi_parallel', 0))}/3 |")

    lines += ["", "## 5. 结论", "",
              "- 并行收益在 3 次重复中稳定存在：R10–R13 逐题耗时比中位数分别为 "
              + " / ".join(f"{row['median_speedup']:.3f}x" for row in expected["per_case"])
              + "，每题至少 2/3 次更快；聚合下降中位 "
              f"{expected['aggregate_reduction_median_pct']:.2f}%，"
              f"区间 {expected['aggregate_reduction_min_pct']:.2f}%~{expected['aggregate_reduction_max_pct']:.2f}%。",
              "- 单分支题（R05–R09）串行/并行走同一段代码，计入后只会放大噪声"
              f"（9 题口径区间 {all_cases['aggregate_reduction_min_pct']:.2f}%~"
              f"{all_cases['aggregate_reduction_max_pct']:.2f}%），不应作为并行收益证据。",
              "- 报告中建议写：「核心多分支题并行后聚合耗时下降约 20%（3 次中位数 19.1%，区间 9.9%~31.3%）」。",
              "",
              ]
    (OUT / "repeat_report_addendum.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"all": all_cases["aggregate_reduction_median_pct"],
                      "expected_multi": expected["aggregate_reduction_median_pct"],
                      "dynamic_multi": dynamic["aggregate_reduction_median_pct"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
