r"""W7 · 核心题重复实验（交接文档 12.2）：R05–R13 × 3 模式 × N 次重复。

目的：把 W7 的"每题每模式跑一次"升级为"每组合跑 N 次"，输出
通过次数的波动、耗时中位数与区间，以及并行/串行配对收益的稳定性。

复用 experiment_w7.evaluate_case，保证与正式实验完全同一套流程、
同一外部评审、同一硬校验，不新增任何评测偏袒。

产物（默认 outputs/w7_repeat/）：
  repeat_results.csv     每次运行的原始结果（含 repeat 列）
  case_mode_stats.csv    每题每模式的通过次数/耗时中位数/区间
  repeat_summary.json    模式级汇总、并行配对统计、运行元数据
  repeat_report.md       可直接抄进报告的中文摘要
  repeat_log.txt         逐次运行日志

运行（项目根目录）：
  dsp\Scripts\python.exe experiment_w7_repeat.py --smoke
  dsp\Scripts\python.exe experiment_w7_repeat.py
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
if not (PROJECT / "experiment_w7.py").exists():
    PROJECT = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search")
sys.path.insert(0, str(PROJECT))

import experiment_w7 as ex          # noqa: E402  (导入即加载 multi_agent / 知识库)
import multi_agent as ma            # noqa: E402

OUT = PROJECT / "outputs" / "w7_repeat"
OUT.mkdir(parents=True, exist_ok=True)

CORE_IDS = ["R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "R13"]
MODES = list(ex.MODES)


def selected_cases(ids: list[str]) -> list[dict]:
    by_id = {case["id"]: case for case in ex.ALL_CASES}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"未知题号：{missing}")
    return [by_id[i] for i in ids]


def _f(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


async def run_once(cases: list[dict], modes: list[str], repeat_index: int,
                   rows: list[dict], logs: list[str]) -> None:
    for case in cases:
        for mode in modes:
            started = time.perf_counter()
            try:
                row, log = await ex.evaluate_case(case, mode)
            except Exception as exc:            # 兜底：任何异常都不中断整轮
                row = {
                    "case_id": case["id"], "question": case["question"], "mode": mode,
                    "expected_routes": "|".join(sorted(case["routes"])),
                    "planned_routes": "", "actual_routes": "", "successful_routes": "",
                    "capability_ok": False, "internal_verdict": "ERROR", "hard_check": "ERROR",
                    "checked_numbers": 0, "missing_numbers": "", "missing_facts": "",
                    "judge_verdict": "ERROR", "judge_comments": "",
                    "objective_pass": False, "overall_pass": False,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "revision_count": 0, "answer": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                log = f"{case['id']} | {mode} | ERROR：{row['error']}\n"
                print("  ERROR：" + row["error"])
            row["repeat"] = repeat_index
            rows.append(row)
            logs.append(f"######## repeat {repeat_index} ########\n" + log)
            pd.DataFrame(rows).to_csv(OUT / "repeat_results.csv",
                                      index=False, encoding="utf-8-sig")
            (OUT / "repeat_log.txt").write_text("\n\n".join(logs), encoding="utf-8")


def case_mode_stats(frame: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for (case_id, mode), group in frame.groupby(["case_id", "mode"], sort=True):
        times = group["elapsed_seconds"].astype(float)
        stats.append({
            "case_id": case_id,
            "mode": mode,
            "mode_label": ex.MODE_LABELS[mode],
            "runs": len(group),
            "overall_pass_count": int(group["overall_pass"].sum()),
            "overall_pass_rate": _f(group["overall_pass"].mean(), 4),
            "objective_pass_count": int(group["objective_pass"].sum()),
            "hard_pass_count": int((group["hard_check"] == "PASS").sum()),
            "judge_pass_count": int((group["judge_verdict"] == "PASS").sum()),
            "capability_pass_count": int(group["capability_ok"].sum()),
            "error_count": int(group["error"].fillna("").astype(bool).sum()),
            "median_seconds": _f(times.median()),
            "mean_seconds": _f(times.mean()),
            "std_seconds": _f(times.std(ddof=1), 3) if len(times) > 1 else None,
            "min_seconds": _f(times.min()),
            "max_seconds": _f(times.max()),
        })
    return pd.DataFrame(stats)


def paired_stats(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    ok = frame[frame["error"].fillna("").eq("")]
    if ok.empty or not {"multi_serial", "multi_parallel"}.issubset(set(ok["mode"])):
        return {}, pd.DataFrame()
    timing = ok.pivot_table(index=["repeat", "case_id"], columns="mode",
                            values="elapsed_seconds")
    timing = timing.dropna(subset=["multi_serial", "multi_parallel"])
    if timing.empty:
        return {}, pd.DataFrame()
    timing = timing.reset_index()
    timing["speedup"] = timing["multi_serial"] / timing["multi_parallel"]
    timing["time_reduction_pct"] = ((timing["multi_serial"] - timing["multi_parallel"])
                                    / timing["multi_serial"] * 100)
    timing.round(4).to_csv(OUT / "paired_timings_by_repeat.csv",
                           index=False, encoding="utf-8-sig")

    per_case = []
    for case_id, group in timing.groupby("case_id", sort=True):
        per_case.append({
            "case_id": case_id,
            "repeats": len(group),
            "median_serial": _f(group["multi_serial"].median()),
            "median_parallel": _f(group["multi_parallel"].median()),
            "median_speedup": _f(group["speedup"].median()),
            "median_reduction_pct": _f(group["time_reduction_pct"].median(), 2),
            "parallel_faster_repeats": int((group["speedup"] > 1).sum()),
        })
    per_case_frame = pd.DataFrame(per_case)
    per_case_frame.to_csv(OUT / "paired_stats_by_case.csv",
                          index=False, encoding="utf-8-sig")

    per_repeat = []
    for rep, group in timing.groupby("repeat", sort=True):
        per_repeat.append({
            "repeat": int(rep),
            "cases": len(group),
            "parallel_faster_cases": int((group["speedup"] > 1).sum()),
            "mean_speedup": _f(group["speedup"].mean()),
            "ratio_of_mean_times": _f(group["multi_serial"].mean() / group["multi_parallel"].mean()),
            "aggregate_reduction_pct": _f(
                (group["multi_serial"].sum() - group["multi_parallel"].sum())
                / group["multi_serial"].sum() * 100, 2),
        })
    per_repeat_frame = pd.DataFrame(per_repeat)

    aggregate_reductions = per_repeat_frame["aggregate_reduction_pct"].astype(float)
    summary = {
        "paired_case_repeats": int(len(timing)),
        "parallel_faster_repeats": int((timing["speedup"] > 1).sum()),
        "per_case_view": {
            "mean_of_median_speedup": _f(per_case_frame["median_speedup"].mean()),
            "median_of_median_speedup": _f(per_case_frame["median_speedup"].median()),
            "cases_parallel_faster_majority": int(
                (per_case_frame["parallel_faster_repeats"] >= 2).sum()),
            "cases_total": int(len(per_case_frame)),
            "median_of_median_reduction_pct": _f(
                per_case_frame["median_reduction_pct"].median(), 2),
        },
        "per_repeat_aggregate": per_repeat,
        "aggregate_reduction_median_pct": _f(aggregate_reductions.median(), 2),
        "aggregate_reduction_min_pct": _f(aggregate_reductions.min(), 2),
        "aggregate_reduction_max_pct": _f(aggregate_reductions.max(), 2),
        "ratio_of_mean_times_median": _f(
            float(per_repeat_frame["ratio_of_mean_times"].median())),
    }
    return summary, per_case_frame


def mode_summary(frame: pd.DataFrame, repeats: int) -> list[dict]:
    rows = []
    for mode, group in frame.groupby("mode", sort=True):
        times = group["elapsed_seconds"].astype(float)
        per_repeat = []
        for rep, rep_group in group.groupby("repeat", sort=True):
            per_repeat.append(int(rep_group["overall_pass"].sum()))
        rows.append({
            "mode": mode,
            "mode_label": ex.MODE_LABELS[mode],
            "runs": int(len(group)),
            "cases": int(group["case_id"].nunique()),
            "repeats": repeats,
            "overall_pass_runs": int(group["overall_pass"].sum()),
            "overall_pass_rate": _f(group["overall_pass"].mean(), 4),
            "overall_pass_per_repeat": per_repeat,
            "overall_pass_per_repeat_median": _f(pd.Series(per_repeat).median(), 2),
            "overall_pass_per_repeat_min": int(min(per_repeat)),
            "overall_pass_per_repeat_max": int(max(per_repeat)),
            "objective_pass_runs": int(group["objective_pass"].sum()),
            "hard_pass_runs": int((group["hard_check"] == "PASS").sum()),
            "judge_pass_runs": int((group["judge_verdict"] == "PASS").sum()),
            "capability_pass_runs": int(group["capability_ok"].sum()),
            "error_runs": int(group["error"].fillna("").astype(bool).sum()),
            "median_seconds": _f(times.median()),
            "mean_seconds": _f(times.mean()),
            "std_seconds": _f(times.std(ddof=1), 3) if len(times) > 1 else None,
            "p90_seconds": _f(times.quantile(0.9)) if len(times) >= 3 else None,
            "min_seconds": _f(times.min()),
            "max_seconds": _f(times.max()),
        })
    return rows


def write_report(payload: dict, frame: pd.DataFrame) -> None:
    meta = payload["metadata"]
    modes = payload["mode_summary"]
    paired = payload["paired_parallel"]
    lines = [
        "# W7 核心题重复实验报告（" + (
            meta["case_ids"][0] if len(meta["case_ids"]) == 1
            else f"{meta['case_ids'][0]}–{meta['case_ids'][-1]}"
        ) + f" × 模式 × {meta['repeats']} 次）",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 模型：{meta['model']}（temperature={meta['temperature']}），RAG 后端：{meta['rag_backend']}",
        f"- 题号：{', '.join(meta['case_ids'])}",
        f"- 重复次数：{meta['repeats']}；总运行次数：{meta['total_runs']}",
        f"- 实际总墙钟耗时（含外部评审）：{meta['wall_seconds_including_judges']}s",
        "",
        "## 1. 模式级结果（每题每模式 N 次）",
        "",
        "| 模式 | 运行次数 | 通过率 | 每次通过题数（中位/最小–最大） | 耗时中位 | 耗时标准差 | 耗时区间 | 硬校验通过 | 外部评审通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in modes:
        lines.append(
            f"| {row['mode_label']} | {row['runs']} | {row['overall_pass_rate']*100:.2f}% | "
            f"{row['overall_pass_per_repeat_median']}/{row['cases']} "
            f"({row['overall_pass_per_repeat_min']}–{row['overall_pass_per_repeat_max']}) | "
            f"{row['median_seconds']:.3f}s | "
            f"{row['std_seconds'] if row['std_seconds'] is not None else '-'} | "
            f"{row['min_seconds']:.3f}–{row['max_seconds']:.3f}s | "
            f"{row['hard_pass_runs']}/{row['runs']} | {row['judge_pass_runs']}/{row['runs']} |"
        )

    lines += ["", "## 2. 每题每模式明细（通过次数 / 耗时中位数）", ""]
    header = "| 题号 | " + " | ".join(ex.MODE_LABELS[m] for m in MODES) + " |"
    divider = "|---|" + "---|" * len(MODES)
    lines += [header, divider]
    for case_id in meta["case_ids"]:
        cells = []
        for mode in MODES:
            sub = frame[(frame["case_id"] == case_id) & (frame["mode"] == mode)]
            if sub.empty:
                cells.append("-")
                continue
            passed = int(sub["overall_pass"].sum())
            total = len(sub)
            cells.append(f"{passed}/{total} · {sub['elapsed_seconds'].median():.2f}s")
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")

    if paired:
        view = paired["per_case_view"]
        lines += [
            "", "## 3. 并行 vs 串行（按重复配对）", "",
            f"- 配对样本：{paired['paired_case_repeats']} 个（题 × 重复），其中并行更快 {paired['parallel_faster_repeats']} 个。",
            f"- 题级耗时比（中位数之比）均值 {view['mean_of_median_speedup']:.3f}x，"
            f"中位 {view['median_of_median_speedup']:.3f}x。",
            f"- 至少 2/3 次并行更快的题：{view['cases_parallel_faster_majority']}/{view['cases_total']}。",
            f"- 聚合耗时下降：中位 {paired['aggregate_reduction_median_pct']:.2f}%"
            f"（{paired['aggregate_reduction_min_pct']:.2f}% ~ {paired['aggregate_reduction_max_pct']:.2f}%）。",
            "",
            "| 重复轮次 | 并行更快的题 | 平均加速比 | 均值耗时比 | 聚合耗时下降 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in paired["per_repeat_aggregate"]:
            lines.append(
                f"| {row['repeat']} | {row['parallel_faster_cases']}/{row['cases']} | "
                f"{row['mean_speedup']:.3f}x | {row['ratio_of_mean_times']:.3f}x | "
                f"{row['aggregate_reduction_pct']:.2f}% |"
            )

    lines += [
        "", "## 4. 结论口径", "",
        "- 通过率与耗时的波动来自 LLM 采样与 API 延迟，本报告给出中位数与区间，不做统计显著性推断。",
        "- 单 Agent 的硬校验失败属设计使然（无 Fact 声明式渲染），指标口径与 W7 主实验一致。",
        "- 串行/并行的差异只在成功取得 ≥2 路证据的多分支题上可归因于物理并行；配对结果见 CSV。",
        "",
        f"明细：case_mode_stats.csv / paired_stats_by_case.csv / paired_timings_by_repeat.csv / "
        f"repeat_results.csv / repeat_log.txt",
        "",
    ]
    (OUT / "repeat_report.md").write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="W7 核心题重复实验")
    parser.add_argument("--cases", nargs="*", default=CORE_IDS)
    parser.add_argument("--modes", nargs="*", choices=MODES, default=MODES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true",
                        help="只跑首题首模式的 1 次，验证环境与写盘")
    parser.add_argument("--out", default=None,
                        help="产物目录（相对项目根目录，默认 outputs/w7_repeat）")
    args = parser.parse_args()

    global OUT
    if args.out:
        OUT = (PROJECT / args.out).resolve()
        OUT.mkdir(parents=True, exist_ok=True)

    cases = selected_cases(args.cases)
    repeats = 1 if args.smoke else max(1, args.repeats)
    if args.smoke:
        cases, cases_label = cases[:1], "冒烟"
        args.modes = args.modes[:1]
    else:
        cases_label = "正式"

    print(f"W7 重复实验（{cases_label}）：{len(cases)} 题 × {len(args.modes)} 模式 × {repeats} 次")
    print(f"模型：deepseek-chat | temperature=0 | RAG={ma.kb.mode}")

    rows: list[dict] = []
    logs: list[str] = []
    start = time.perf_counter()
    for repeat_index in range(1, repeats + 1):
        print(f"\n===== 第 {repeat_index}/{repeats} 轮 =====")
        await run_once(cases, args.modes, repeat_index, rows, logs)
    wall = round(time.perf_counter() - start, 3)

    if args.smoke:
        print(f"\n冒烟完成：{len(rows)} 次运行，{wall}s。产物在 {OUT}")
        return

    frame = pd.DataFrame(rows)
    stats = case_mode_stats(frame)
    stats.to_csv(OUT / "case_mode_stats.csv", index=False, encoding="utf-8-sig")
    paired, per_case_frame = paired_stats(frame)
    modes = mode_summary(frame, repeats)

    payload = {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": "deepseek-chat",
            "temperature": 0,
            "rag_backend": ma.kb.mode,
            "case_ids": [case["id"] for case in cases],
            "modes": args.modes,
            "repeats": repeats,
            "total_runs": len(rows),
            "wall_seconds_including_judges": wall,
            "purpose": "交接文档 12.2：补充核心题重复实验，报告中位数与波动区间",
        },
        "mode_summary": modes,
        "paired_parallel": paired,
        "acceptance": {
            "objective_pass": "成功执行所需工具 + 原始证据/Fact 关系/声明式格式检查通过 + 关键事实命中",
            "overall_pass": "objective_pass + 同一外部 LLM 评审 PASS",
            "elapsed_seconds": "不含外部评审耗时",
        },
        "limitations": [
            "每题每模式仅 N 次重复，用于观察波动而非统计检验",
            "外部 LLM 评审不替代人工抽查",
        ],
    }
    (OUT / "repeat_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, frame)

    print("\n=== 模式汇总（重复 {} 次）===".format(repeats))
    print(pd.DataFrame(modes)[
        ["mode_label", "runs", "overall_pass_rate", "median_seconds", "std_seconds",
         "min_seconds", "max_seconds", "hard_pass_runs", "judge_pass_runs"]
    ].to_string(index=False))
    if paired:
        print("\n=== 并行/串行配对 ===")
        print(json.dumps(paired["per_repeat_aggregate"], ensure_ascii=False, indent=2))
        print(json.dumps(paired["per_case_view"], ensure_ascii=False, indent=2))
    print(f"\n产物目录：{OUT}")


if __name__ == "__main__":
    asyncio.run(main())
