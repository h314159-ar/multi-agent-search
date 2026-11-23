r"""W6 · 10 题端到端回归测试。

每题同时检查：
1. Planner 实际路由覆盖预期路由；
2. Critic 最终 verdict=PASS 且程序硬校验通过；
3. 最终答案包含该题的关键事实要素。

运行：dsp\Scripts\python regression_w6.py
产物：outputs/w6/regression_results.csv / regression_summary.json / regression_log.txt
"""
import asyncio
import json
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

import multi_agent as ma


def must_variants(value: str) -> set[str]:
    """must 事实的合法展示变体：原文 + 声明式舍入（如 0.4077 → 0.41）。"""
    variants = {value}
    try:
        number = Decimal(value.replace("%", "").replace(",", ""))
    except Exception:
        return variants
    for digits in (1, 2):
        quantum = Decimal(1).scaleb(-digits)
        variants.add(str(number.quantize(quantum, rounding=ROUND_HALF_UP)))
    return variants


def missing_musts(case: dict, final: str) -> list[str]:
    return [x for x in case["must"]
            if not any(v in final for v in must_variants(x))]

OUT = Path(__file__).resolve().parent / "outputs" / "w6"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    {"id": "R01", "question": "评分超过 4.0 且至少有 100 人评价的电影有哪些？",
     "routes": {"sql"}, "must": ["47"]},
    {"id": "R02", "question": "评价人数最多的 10 部电影是哪些？请给出评价人数和平均评分。",
     "routes": {"sql"}, "must": []},
    {"id": "R03", "question": "Comedy 类型电影中，平均评分最高且评价人数不少于 50 的 5 部是哪些？",
     "routes": {"sql"}, "must": []},
    {"id": "R04", "question": "用户分群的结果怎么样？喜欢科幻片的人还喜欢什么类型？",
     "routes": {"mining"}, "must": ["0.4077"]},
    {"id": "R05", "question": "决策树预测宽容型评分者的准确率是多少？最重要的特征是什么？",
     "routes": {"mining"}, "must": ["70.3%", "std_rating"]},
    {"id": "R06", "question": "重度用户和轻度用户的关联规则有什么差异？哪些规则跨组稳定？",
     "routes": {"mining"}, "must": ["重度", "轻度"]},
    {"id": "R07", "question": "为什么 K-Means 选 K=2？PCA 为什么不参与聚类训练？",
     "routes": {"rag"}, "must": ["0.4077", "PCA"]},
    {"id": "R08", "question": "关联规则里的 lift 和 support 怎么理解？为什么不能只看 confidence？",
     "routes": {"rag"}, "must": ["lift", "confidence"]},
    {"id": "R09", "question": "LangGraph 并行分支写同一个 State 字段为什么会冲突？reducer 怎么解决？",
     "routes": {"rag"}, "must": ["reducer"]},
    {"id": "R10", "question": "给我一份关于 MovieLens 用户偏好模式的简报，并说明挖掘方法。",
     "routes": {"mining", "rag"}, "must": ["532", "78"]},
]


async def main():
    results = []
    full_logs = []
    suite_start = time.perf_counter()
    print(f"W6 回归测试：{len(CASES)} 题 | RAG 后端：{ma.kb.mode}")
    if ma.kb.fallback_reason:
        print(f"RAG 回退原因：{ma.kb.fallback_reason}")

    for case in CASES:
        case_start = time.perf_counter()
        try:
            state = await ma.arun(case["question"])
            planned_routes = set(state["routes"])
            actual_routes = ma.invoked_routes(state["tool_records"])
            successful_routes = ma.successful_routes(state["tool_records"])
            # 计划路由不等于能力覆盖：必须实际调用并成功执行（空结果除外仍算成功执行）。
            route_ok = case["routes"].issubset(successful_routes)
            missing_facts = missing_musts(case, state["final"])
            passed = (route_ok and state["verdict"] == "PASS"
                      and state["hard_check"] == "PASS" and not missing_facts)
            row = {
                "id": case["id"], "question": case["question"],
                "expected_routes": "|".join(sorted(case["routes"])),
                "planned_routes": "|".join(sorted(planned_routes)),
                "actual_routes": "|".join(sorted(actual_routes)),
                "successful_routes": "|".join(sorted(successful_routes)),
                "route_ok": route_ok, "verdict": state["verdict"],
                "hard_check": state["hard_check"],
                "missing_facts": "|".join(missing_facts),
                "elapsed_seconds": round(state["elapsed_seconds"], 3),
                "passed": passed, "final": state["final"], "error": "",
            }
            full_logs.append(f"{case['id']} 问：{case['question']}\n"
                             + "\n".join(state["trace"])
                             + f"\n最终回答（{state['verdict']}）：\n{state['final']}\n"
                             + (f"内部草稿（仅诊断）：\n{state['draft']}\n"
                                if state["verdict"] == "FAIL" else ""))
        except Exception as exc:
            row = {
                "id": case["id"], "question": case["question"],
                "expected_routes": "|".join(sorted(case["routes"])),
                "planned_routes": "", "actual_routes": "", "successful_routes": "",
                "route_ok": False, "verdict": "ERROR",
                "hard_check": "ERROR", "missing_facts": "",
                "elapsed_seconds": round(time.perf_counter() - case_start, 3),
                "passed": False, "final": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            full_logs.append(f"{case['id']} ERROR：{row['error']}\n")
        results.append(row)

    total_seconds = time.perf_counter() - suite_start
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "regression_results.csv", index=False, encoding="utf-8-sig")
    with (OUT / "regression_log.txt").open("w", encoding="utf-8") as f:
        f.write("\n\n".join(full_logs))

    passed_count = int(frame["passed"].sum())
    summary = {
        "total": len(CASES), "passed": passed_count,
        "pass_rate": round(passed_count / len(CASES), 4),
        "critic_pass": int((frame["verdict"] == "PASS").sum()),
        "hard_check_pass": int((frame["hard_check"] == "PASS").sum()),
        "route_pass": int(frame["route_ok"].sum()),
        "error_count": int((frame["verdict"] == "ERROR").sum()),
        "error_rate": round(float((frame["verdict"] == "ERROR").mean()), 4),
        "total_seconds": round(total_seconds, 3),
        "mean_seconds": round(float(frame["elapsed_seconds"].mean()), 3),
        "rag_backend": ma.kb.mode,
        "rag_fallback_reason": ma.kb.fallback_reason,
        "failed_ids": frame.loc[~frame["passed"], "id"].tolist(),
    }
    with (OUT / "regression_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== W6 回归汇总 ===")
    print(frame[["id", "planned_routes", "actual_routes", "successful_routes",
                 "verdict", "hard_check", "elapsed_seconds", "passed"]].to_string(index=False))
    print(f"\n通过 {passed_count}/{len(CASES)}（{summary['pass_rate']:.0%}） | "
          f"平均耗时 {summary['mean_seconds']:.2f}s | 总耗时 {total_seconds:.1f}s")
    if summary["failed_ids"]:
        print(f"未通过：{', '.join(summary['failed_ids'])}")
    print(f"结果已保存：{OUT / 'regression_summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
