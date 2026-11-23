r"""W7 · 单 Agent / 多 Agent 串行 / 多 Agent 并行对比实验。

公平性约束：
1. 三种模式使用同一批问题、同一 DeepSeek 模型、同一数据与挖掘工具。
2. 单 Agent 额外获得与 RAG Agent 相同的知识库检索工具。
3. 所有模式统一对照原始工具结果检查能力覆盖、关键事实和数字证据，
   并由同一外部评审检查语义质量；外部评审不参与作答，也不等同于多 Agent 内部 Critic。
4. 耗时只计算各模式自身执行时间，不包含外部评审时间。

运行：
  dsp\Scripts\python experiment_w7.py --cases R10        # 单题三模式冒烟
  dsp\Scripts\python experiment_w7.py                    # 13 题完整实验（含 4 道多分支题）
产物：outputs/w7/
"""
import argparse
import asyncio
from datetime import datetime
import json
import re
import time
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import multi_agent as ma
from regression_w6 import CASES, must_variants

# W7 额外增加三道双分支题：原回归集只有 R10 能检验真实专家并行。
EXTRA_CASES = [
    {"id": "R11",
     "question": "评分超过 4.0 且至少有 100 人评价的电影有多少部？同时说明 ratings 表有哪些字段。",
     "routes": {"sql", "rag"}, "must": ["47", "userId"]},
    {"id": "R12",
     "question": "结合 K-Means 分群结果说明轻度和重度用户的差异，并解释为什么轮廓系数可以用来选 K。",
     "routes": {"mining", "rag"}, "must": ["532", "78", "0.4077"]},
    {"id": "R13",
     "question": "请介绍 MovieLens 数据集概况，并用评分数据库核实总评分数、用户数和电影数。",
     "routes": {"sql", "rag"}, "must": ["100836", "610", "9742"]},
]
ALL_CASES = CASES + EXTRA_CASES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "w7"
OUT.mkdir(parents=True, exist_ok=True)
MODES = ("single", "multi_serial", "multi_parallel")
MODE_LABELS = {
    "single": "单 Agent",
    "multi_serial": "多 Agent 串行",
    "multi_parallel": "多 Agent 并行",
}
MINING_TOOLS = {"cluster_profiles", "association_rules", "decision_tree_summary"}


@tool
def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """检索项目知识库，返回 K-Means、Apriori、LangGraph 等方法与项目事实。"""
    hits = ma.kb.retrieve(query, max(1, min(int(top_k), 5)))
    if hits.empty:
        return "知识库没有召回到相关文档"
    return "\n\n".join(
        f"【文档 {row['doc_id']} · {row['topic']}】\n{row['text']}"
        for _, row in hits.iterrows()
    )


SINGLE_TOOLS = [
    ma.run_sql, ma.cluster_profiles, ma.association_rules,
    ma.decision_tree_summary, retrieve_knowledge,
]
SINGLE_SYSTEM = """你是单 Agent 数据分析助手，需要独立理解问题、调用工具并直接给出最终回答。
你可以使用 SQL、K-Means 分群、Apriori 关联规则、决策树和项目知识库工具。
要求：
1. 涉及真实数据或项目结论时必须调用工具，禁止凭记忆编造。
2. 知识库没有的内容明确说不知道；SQL 的展示行数不等于完整结果总数。
3. 只使用工具返回的事实与数字，不自行计算或四舍五入。
4. 用中文直接回答，先结论后细节，200 字以内。

数据库规则：
""" + ma.DATA_DICT.split("\n", 1)[1]


def merge_update(state: dict, update: dict) -> None:
    """模拟 LangGraph 节点写回：合并证据、累加轨迹，其余字段覆盖。"""
    for key, value in update.items():
        if key in ("evidences", "tool_evidences"):
            state[key].update(value)
        elif key in ("trace", "tool_records"):
            state[key].extend(value)
        else:
            state[key] = value


def routes_from_tools(tool_names: set[str]) -> set[str]:
    routes = set()
    if "run_sql" in tool_names:
        routes.add("sql")
    if tool_names & MINING_TOOLS:
        routes.add("mining")
    if "retrieve_knowledge" in tool_names:
        routes.add("rag")
    return routes


async def run_single(question: str) -> dict:
    """单 Agent：一个 LLM 使用全部工具，无 Planner、Synthesizer 或内部 Critic。"""
    start = time.perf_counter()
    messages = [SystemMessage(content=SINGLE_SYSTEM), HumanMessage(content=question)]
    bound = ma.llm.bind_tools(SINGLE_TOOLS)
    tool_map = {item.name: item for item in SINGLE_TOOLS}
    tool_calls = []
    records = []
    answer = ""

    for _ in range(8):
        response = await bound.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            answer = response.content
            break
        for call in response.tool_calls:
            args_text = json.dumps(call["args"], ensure_ascii=False)
            tool_calls.append(f"{call['name']}({args_text})")
            result = await asyncio.to_thread(ma.execute_tool, call["name"], call["args"], tool_map)
            record = result.to_record()
            records.append(record)
            tool_calls[-1] += f" -> {result.status}"
            messages.append(ToolMessage(content=result.text, tool_call_id=call["id"]))
    else:
        response = await ma.llm.ainvoke(messages + [HumanMessage(content=(
            "工具调用轮次已用尽。请只依据已有工具结果，立即给出最终回答；"
            "证据不足的部分明确说明。"))])
        answer = response.content

    return {
        "answer": answer,
        "evidence_text": "\n\n".join(ma.usable_raw_evidences(records)),
        "tool_calls": tool_calls,
        "tool_records": records,
        "planned_routes": set(),
        "actual_routes": ma.invoked_routes(records),
        "successful_routes": ma.successful_routes(records),
        "verdict": "N/A（无内部 Critic）",
        "revision_count": 0,
        "trace": [f"[单 Agent] 工具调用：{'; '.join(tool_calls) or '无'}"],
        "elapsed_seconds": time.perf_counter() - start,
    }


async def run_multi(question: str, parallel: bool) -> dict:
    """手动驱动同一组 W6 节点；唯一差别是专家扇出采用 gather 还是顺序 await。"""
    start = time.perf_counter()
    state = ma.initial_state(question)
    merge_update(state, await ma.planner(state))

    node_map = {"sql": ma.sql_agent, "mining": ma.mining_agent, "rag": ma.rag_agent}
    nodes = [node_map[route] for route in state["routes"]]
    if parallel:
        updates = await asyncio.gather(*(node(state) for node in nodes))
    else:
        updates = []
        for node in nodes:
            updates.append(await node(state))
    for update in updates:
        merge_update(state, update)

    while True:
        merge_update(state, await ma.synthesizer(state))
        merge_update(state, await ma.critic(state))
        if state["verdict"] != "REVISE":
            break

    return {
        "answer": state["final"],
        # 外部实验统一审计原始工具结果，不用专家自行改写后的摘要充当源头。
        "evidence_text": "\n\n".join(
            f"【{key}】\n{value}" for key, value in state["tool_evidences"].items()
        ),
        "tool_calls": [line for line in state["trace"] if "工具调用" in line],
        "tool_records": state["tool_records"],
        "planned_routes": set(state["routes"]),
        "actual_routes": ma.invoked_routes(state["tool_records"]),
        "successful_routes": ma.successful_routes(state["tool_records"]),
        "verdict": state["verdict"],
        "revision_count": state["revision_count"],
        # 原始草稿含 [F###:round:N] 等声明式引用，外部审计以它为对象，
        # 避免对渲染后纯数字二次审计时丢失舍入/百分比声明上下文。
        "draft": state["draft"],
        "trace": state["trace"],
        "elapsed_seconds": time.perf_counter() - start,
    }


async def external_judge(question: str, answer: str, evidence: str) -> tuple[str, str]:
    """三种模式共用的外部质量评审；不修改答案，不将多 Agent Critic 当评委。"""
    if not answer.strip():
        return "FAIL", "回答为空"
    if answer.startswith("【") and "阻断】" in answer[:40]:
        return "FAIL", "系统已阻断该回答，不作为合格答案放行"
    response = await ma.llm.ainvoke([
        SystemMessage(content=(
            "你是独立实验评审，不属于被测 Agent。根据证据判断答案是否直接回答问题、"
            "结论是否与证据一致、是否夸大或遗漏关键限制。不要自行补充外部事实。"
            "只输出 JSON：{\"verdict\":\"PASS\" 或 \"FAIL\","
            " \"comments\":\"简短、具体的理由\"}。"
        )),
        HumanMessage(content=(
            f"问题：{question}\n\n证据：\n{evidence or '（无工具证据）'}"
            f"\n\n待评答案：\n{answer}"
        )),
    ])
    try:
        result = json.loads(re.search(r"\{.*\}", response.content, re.S).group())
        verdict = result.get("verdict", "FAIL").upper()
        if verdict not in ("PASS", "FAIL"):
            verdict = "FAIL"
        return verdict, result.get("comments", "")
    except Exception:
        return "ERROR", "外部评审 JSON 解析失败：" + response.content[:200]


async def evaluate_case(case: dict, mode: str) -> tuple[dict, str]:
    question = case["question"]
    case_start = time.perf_counter()
    print(f"\n[{case['id']} | {MODE_LABELS[mode]}] {question}")
    try:
        if mode == "single":
            result = await run_single(question)
        else:
            result = await run_multi(question, parallel=(mode == "multi_parallel"))

        # 与正式问答共用同一套原始证据、Fact 关系和声明式格式校验。
        # 多 Agent 模式审计原始草稿（含声明式引用）；单 Agent 无草稿则审最终答案。
        audit_target = result.get("draft") or result["answer"]
        audit = ma.audit_answer(audit_target, result["evidence_text"],
                                result["tool_records"], question)
        hard_ok = audit["ok"]
        missing_numbers = audit["missing_numbers"] + audit["errors"]
        number_count = audit["checked_numbers"]
        rendered_answer = audit["rendered_answer"]
        answer_without_commas = rendered_answer.replace(",", "")
        missing_facts = [fact for fact in case["must"]
                         if not any(v in rendered_answer or v in answer_without_commas
                                    for v in must_variants(fact))]
        capability_ok = case["routes"].issubset(result["successful_routes"])
        objective_pass = capability_ok and hard_ok and not missing_facts
        judge_verdict, judge_comments = await external_judge(
            question, rendered_answer, result["evidence_text"]
        )
        overall_pass = objective_pass and judge_verdict == "PASS"
        row = {
            "case_id": case["id"], "question": question, "mode": mode,
            "expected_routes": "|".join(sorted(case["routes"])),
            "planned_routes": "|".join(sorted(result["planned_routes"])),
            "actual_routes": "|".join(sorted(result["actual_routes"])),
            "successful_routes": "|".join(sorted(result["successful_routes"])),
            "capability_ok": capability_ok,
            "internal_verdict": result["verdict"],
            "hard_check": "PASS" if hard_ok else "FAIL",
            "checked_numbers": number_count,
            "missing_numbers": "|".join(missing_numbers),
            "missing_facts": "|".join(missing_facts),
            "judge_verdict": judge_verdict,
            "judge_comments": judge_comments,
            "objective_pass": objective_pass,
            "overall_pass": overall_pass,
            "elapsed_seconds": round(result["elapsed_seconds"], 3),
            "revision_count": result["revision_count"],
            "answer": rendered_answer,
            "error": "",
        }
        log = (
            f"{case['id']} | {MODE_LABELS[mode]}\n问：{question}\n"
            + "\n".join(result["trace"])
            + f"\n\n证据：\n{result['evidence_text']}"
            + f"\n\n最终回答：\n{rendered_answer}"
            + f"\n\n硬校验：{row['hard_check']} | 外部评审：{judge_verdict} | "
              f"{judge_comments}\n耗时：{row['elapsed_seconds']:.3f}s\n"
        )
        print(
            f"  能力={'OK' if capability_ok else 'FAIL'} | "
            f"硬校验={row['hard_check']}({number_count} 个数字) | "
            f"外部评审={judge_verdict} | 总评={'PASS' if overall_pass else 'FAIL'} | "
            f"{row['elapsed_seconds']:.2f}s"
        )
        return row, log
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        row = {
            "case_id": case["id"], "question": question, "mode": mode,
            "expected_routes": "|".join(sorted(case["routes"])),
            "planned_routes": "", "actual_routes": "", "successful_routes": "",
            "capability_ok": False,
            "internal_verdict": "ERROR", "hard_check": "ERROR",
            "checked_numbers": 0, "missing_numbers": "", "missing_facts": "",
            "judge_verdict": "ERROR", "judge_comments": "",
            "objective_pass": False, "overall_pass": False,
            "elapsed_seconds": round(time.perf_counter() - case_start, 3),
            "revision_count": 0, "answer": "", "error": error,
        }
        print("  ERROR：" + error)
        return row, f"{case['id']} | {MODE_LABELS[mode]} | ERROR：{error}\n"


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    summaries = []
    for mode in MODES:
        group = frame[frame["mode"] == mode]
        if group.empty:
            continue
        summaries.append({
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "cases": len(group),
            "objective_pass": int(group["objective_pass"].sum()),
            "objective_pass_rate": round(float(group["objective_pass"].mean()), 4),
            "judge_pass": int((group["judge_verdict"] == "PASS").sum()),
            "overall_pass": int(group["overall_pass"].sum()),
            "overall_pass_rate": round(float(group["overall_pass"].mean()), 4),
            "capability_pass": int(group["capability_ok"].sum()),
            "hard_check_pass": int((group["hard_check"] == "PASS").sum()),
            "error_count": int(group["error"].astype(bool).sum()),
            "error_rate": round(float(group["error"].astype(bool).mean()), 4),
            "mean_seconds": round(float(group["elapsed_seconds"].mean()), 3),
            "median_seconds": round(float(group["elapsed_seconds"].median()), 3),
            "total_seconds": round(float(group["elapsed_seconds"].sum()), 3),
            "mean_revisions": round(float(group["revision_count"].mean()), 3),
        })
    summary_frame = pd.DataFrame(summaries)

    successful_frame = frame[frame["error"].fillna("").eq("")]
    timing = successful_frame.pivot(index="case_id", columns="mode", values="elapsed_seconds")
    paired = {}
    if {"multi_serial", "multi_parallel"}.issubset(timing.columns):
        valid = timing[(timing["multi_serial"] > 0) & (timing["multi_parallel"] > 0)].copy()
        valid["speedup"] = valid["multi_serial"] / valid["multi_parallel"]
        valid["time_reduction_pct"] = (
            (valid["multi_serial"] - valid["multi_parallel"])
            / valid["multi_serial"] * 100
        )
        valid.round(3).to_csv(OUT / "paired_timings.csv", encoding="utf-8-sig")
        paired = {
            "paired_cases": len(valid),
            "parallel_faster_cases": int((valid["speedup"] > 1).sum()),
            "mean_speedup_per_case": round(float(valid["speedup"].mean()), 3),
            "ratio_of_mean_times": round(
                float(valid["multi_serial"].mean() / valid["multi_parallel"].mean()), 3
            ),
            "mean_time_reduction_pct": round(float(valid["time_reduction_pct"].mean()), 2),
            "aggregate_time_reduction_pct": round(float(
                (valid["multi_serial"].sum() - valid["multi_parallel"].sum())
                / valid["multi_serial"].sum() * 100
            ), 2),
        }
        # 只有实际成功取得至少两路证据的题，才把差异主要归因于物理并行。
        route_counts = (successful_frame[successful_frame["mode"] == "multi_serial"]
                        .set_index("case_id")["successful_routes"]
                        .map(lambda value: 0 if not value else len(str(value).split("|"))))
        dual_ids = route_counts[route_counts >= 2].index
        dual = valid.loc[valid.index.intersection(dual_ids)]
        if not dual.empty:
            paired["multi_route_cases"] = list(dual.index)
            paired["multi_route_parallel_faster_cases"] = int((dual["speedup"] > 1).sum())
            paired["multi_route_mean_speedup"] = round(float(dual["speedup"].mean()), 3)
            paired["multi_route_ratio_of_mean_times"] = round(float(
                dual["multi_serial"].mean() / dual["multi_parallel"].mean()
            ), 3)
            paired["multi_route_aggregate_time_reduction_pct"] = round(float(
                (dual["multi_serial"].sum() - dual["multi_parallel"].sum())
                / dual["multi_serial"].sum() * 100
            ), 2)
    return summary_frame, paired


async def main() -> None:
    parser = argparse.ArgumentParser(description="W7 单/多 Agent 公平对比实验")
    parser.add_argument("--cases", nargs="*", default=None, help="只运行指定题号，如 R01 R10")
    parser.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    selected = [case for case in ALL_CASES if not args.cases or case["id"] in args.cases]
    if not selected:
        raise SystemExit("没有匹配的测试题")

    print(f"W7 对比实验：{len(selected)} 题 × {len(args.modes)} 模式")
    print(f"模型：deepseek-chat | temperature=0 | RAG={ma.kb.mode}")
    rows, logs = [], []
    suite_start = time.perf_counter()
    for case in selected:
        for mode in args.modes:
            row, log = await evaluate_case(case, mode)
            rows.append(row)
            logs.append(log)
            # 每题每模式立即落盘，网络中断也不会丢失已完成结果。
            pd.DataFrame(rows).to_csv(OUT / "experiment_results.csv",
                                      index=False, encoding="utf-8-sig")
            with (OUT / "experiment_log.txt").open("w", encoding="utf-8") as file:
                file.write("\n\n".join(logs))

    frame = pd.DataFrame(rows)
    summary_frame, paired = summarize(frame)
    summary_frame.to_csv(OUT / "mode_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "deepseek-chat", "temperature": 0,
        "case_ids": [case["id"] for case in selected], "modes": args.modes,
        "rag_backend": ma.kb.mode,
        "wall_seconds_including_judges": round(time.perf_counter() - suite_start, 3),
        "acceptance": {
            "objective_pass": "成功执行所需工具 + 原始证据/Fact 关系/声明式格式检查通过 + 关键事实命中",
            "overall_pass": "objective_pass + 同一外部 LLM 评审 PASS",
            "capability_ok": "expected_routes ⊆ successful_routes；planned/invoked 不计入覆盖",
            "elapsed_seconds": "不含外部评审耗时；异常样本保留真实耗时",
        },
        "limitations": [
            "每题每模式运行一次，耗时受 API 波动影响，不做统计显著性推断",
            "外部 LLM 评审用于语义复核，不替代人工抽查或确定性事实检查",
        ],
        "paired_parallel": paired,
    }
    with (OUT / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n=== W7 模式汇总 ===")
    print(summary_frame.to_string(index=False))
    if paired:
        print("\n=== 串行/并行配对结果 ===")
        print(json.dumps(paired, ensure_ascii=False, indent=2))
    print(f"\n结果已保存：{OUT}")


if __name__ == "__main__":
    asyncio.run(main())
