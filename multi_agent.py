r"""W6 · 多智能体协作搜索系统：物理并行 + Critic 硬校验

架构保持 W5 六角色不变：
  START → Planner（LLM 路由）
        → fan-out: sql_agent / mining_agent / rag_agent
        → Synthesizer → Critic → END

W6 改造：
1. 专家节点与 LLM 调用改为 async，LangGraph ainvoke 在同一 super-step
   并发执行多个专家，实现物理并行；阻塞的工具调用放入线程。
2. Critic 在 LLM 审查前增加确定性数字硬校验：草稿中的每个数字必须
   能在专家证据中找到（忽略千分位差异）。未通过则 REVISE；最多修订
   1 次，仍未通过则 FAIL，不再默认放行。
3. 运行结果与并行耗时对比写入 outputs/w6/。
4. W7 起额外保留原始工具证据，供外部实验直接审计，不改变内部回答流程。

运行：dsp\Scripts\python multi_agent.py   （key 从 .env 自动读取）
"""
import asyncio
import json
import operator
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

import duckdb
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "mining"))
sys.path.insert(0, str(ROOT / "src" / "rag"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from knowledge_base import KnowledgeBase
from mining_tools import (get_cluster_profiles, get_association_rules,
                          get_decision_tree_summary)

OUT = ROOT / "outputs" / "w6"
OUT.mkdir(parents=True, exist_ok=True)

llm = ChatOpenAI(model="deepseek-chat",
                 base_url="https://api.deepseek.com/v1",
                 api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
kb = KnowledgeBase()


# ───────────────────────────── State ─────────────────────────────
def merge_dict(a: dict, b: dict) -> dict:
    return {**a, **b}


class State(TypedDict):
    question: str
    routes: list[str]
    evidences: Annotated[dict, merge_dict]
    tool_evidences: Annotated[dict, merge_dict]           # 原始工具证据（实验审计用）
    draft: str
    verdict: str                                       # PASS / REVISE / FAIL
    critique: str
    hard_check: str                                    # 程序级数字核验结果
    revision_count: int
    final: str
    elapsed_seconds: float
    trace: Annotated[list[str], operator.add]


# ───────────────────────────── 工具 ─────────────────────────────
@tool
def run_sql(sql: str) -> str:
    """在 MovieLens DuckDB 数据库上执行只读 SQL 查询，返回结果。"""
    con = duckdb.connect(str(ROOT / "data" / "movielens.duckdb"), read_only=True)
    try:
        df = con.execute(sql).fetch_df()
        if len(df) == 0:
            return "查询结果为空"
        total = f"完整查询结果共 {len(df)} 行"
        if len(df) > 15:
            total += "（此处仅展示部分，不能把展示行数当作总数）"
        return total + "\n" + df.head(15).to_string(index=False)
    except Exception as e:
        return f"SQL 执行失败: {e}"
    finally:
        con.close()


@tool
def cluster_profiles() -> str:
    """查询 K-Means 用户分群结果（轻度/重度用户画像）。"""
    return get_cluster_profiles()


@tool
def association_rules(group: str = "all", top_n: int = 5) -> str:
    """查询 Apriori 关联规则。group: all/heavy/light，top_n 条数。"""
    return get_association_rules(top_n=top_n, group=group)


@tool
def decision_tree_summary() -> str:
    """查询决策树分类结果（宽容/严格型评分者预测）。"""
    return get_decision_tree_summary()


DATA_DICT = """你是数据检索 Agent，调用 run_sql 工具查 DuckDB 回答。
表 ratings(userId,movieId,rating,timestamp,rated_at,rated_year)，100836 行。
表 movies(movieId,title + 20 个类型 0/1 列：Action Adventure Animation Children
Comedy Crime Documentary Drama Fantasy Film-Noir Horror IMAX Musical Mystery
Romance Sci-Fi Thriller War Western)，9742 行，title 为英文。
规则：必须调用工具查真数据；需要电影名/类型时 JOIN movies；写标准 DuckDB SQL。
工具若注明「仅展示部分」，展示行数不等于结果总数；需要总数时必须用 COUNT 查询。"""


async def tool_loop(messages: list, tools: list, max_rounds: int = 4) -> tuple[str, list[str], list[str]]:
    """异步 LLM + 工具循环；同时保留未经专家改写的原始工具证据。"""
    bound = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    log = []
    raw_evidences = []
    for _ in range(max_rounds):
        resp = await bound.ainvoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            return resp.content, log, raw_evidences
        for call in resp.tool_calls:
            args_str = json.dumps(call["args"], ensure_ascii=False)
            log.append(f"{call['name']}({args_str[:150]})")
            result = await asyncio.to_thread(tool_map[call["name"]].invoke, call["args"])
            result = str(result)
            raw_evidences.append(f"【{call['name']}】\n{result}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return "（工具轮次用尽）" + messages[-1].content, log, raw_evidences


# ───────────────────────────── 节点 ─────────────────────────────
async def planner(state: State) -> dict:
    prompt = f"""你是多智能体系统的 Planner。判断用户问题需要哪些专家协作，只输出 JSON。
可选专家：
- "sql"：需要查询电影评分数据库（评分排行、数量统计、某电影/类型的评分等）
- "mining"：需要数据挖掘结论（用户分群、关联规则/偏好模式、评分者分类）
- "rag"：需要方法论/概念解释（为什么选某参数、某指标含义、某算法原理、字段定义）
可多选（1~3 个）。输出格式：{{"routes": ["sql"], "reason": "一句话理由"}}

用户问题：{state['question']}"""
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        plan = json.loads(re.search(r"\{.*\}", resp.content, re.S).group())
        routes = [r for r in plan.get("routes", []) if r in ("sql", "mining", "rag")]
        reason = plan.get("reason", "")
    except Exception:
        routes, reason = ["rag"], "（路由解析失败，默认 RAG）"
    if not routes:
        routes = ["rag"]
    return {"routes": routes, "revision_count": 0,
            "trace": [f"[Planner] 路由={routes} | {reason}"]}


def route_selector(state: State) -> list[str]:
    mapping = {"sql": "sql_agent", "mining": "mining_agent", "rag": "rag_agent"}
    return [mapping[r] for r in state["routes"]]


async def sql_agent(state: State) -> dict:
    msgs = [SystemMessage(content=DATA_DICT),
            HumanMessage(content=state["question"])]
    answer, log, raw = await tool_loop(msgs, [run_sql])
    return {"evidences": {"sql": answer},
            "tool_evidences": {"sql": "\n\n".join(raw)},
            "trace": [f"[检索Agent] 工具调用: {'; '.join(log) or '无'}",
                      f"[检索Agent] 证据: {answer[:200]}..."]}


async def mining_agent(state: State) -> dict:
    msgs = [SystemMessage(content=(
        "你是数据挖掘 Agent。根据用户问题选择合适的工具获取挖掘结论，"
        "可多次调用不同工具，最后用中文总结证据（保留关键数字）。")),
        HumanMessage(content=state["question"])]
    answer, log, raw = await tool_loop(msgs, [cluster_profiles, association_rules,
                                              decision_tree_summary])
    return {"evidences": {"mining": answer},
            "tool_evidences": {"mining": "\n\n".join(raw)},
            "trace": [f"[挖掘Agent] 工具调用: {'; '.join(log) or '无'}",
                      f"[挖掘Agent] 证据: {answer[:200]}..."]}


async def rag_agent(state: State) -> dict:
    # 查询重写 + 多查询分解：长问题拆成多个主题分别召回，再按 doc_id 合并。
    rewrite = await llm.ainvoke([HumanMessage(content=(
        "把下面问题拆成 1~3 个用于知识库检索的短查询，每个聚焦一个主题，"
        "每行一个（<=20 字，保留核心算法名/指标名/概念，去掉品牌名和客套话）。\n"
        f"问题：{state['question']}"))])
    queries = [q.strip() for q in rewrite.content.strip().splitlines() if q.strip()][:3]
    merged = {}
    for q in queries:
        hits = await asyncio.to_thread(kb.retrieve, q, 2)
        for _, r in hits.iterrows():
            if r["doc_id"] not in merged or r["score"] > merged[r["doc_id"]]["score"]:
                merged[r["doc_id"]] = r
    import pandas as pd
    hits = pd.DataFrame(merged.values()).sort_values("score", ascending=False).head(3)
    context = "\n\n".join(f"【文档 {r['doc_id']} · {r['topic']}】\n{r['text']}"
                          for _, r in hits.iterrows())
    cited = ", ".join(hits["doc_id"])
    resp = await llm.ainvoke([SystemMessage(content=(
        "你是 RAG Agent。只根据给定文档回答，保留关键数字，"
        "结尾标注引用的 doc_id。文档没有的内容明确说不知道，禁止编造。")),
        HumanMessage(content=f"问题：{state['question']}\n\n{context}")])
    return {"evidences": {"rag": resp.content},
            "tool_evidences": {"rag": context},
            "trace": [f"[RAG Agent] 子查询: {' | '.join(queries)}",
                      f"[RAG Agent] 检索后端: {kb.mode} | 召回文档: {cited}（得分 "
                      f"{'/'.join(str(s) for s in hits['score'])}）",
                      f"[RAG Agent] 证据: {resp.content[:200]}..."]}


async def synthesizer(state: State) -> dict:
    evidence_text = "\n\n".join(f"【{k} 专家证据】\n{v}"
                                for k, v in state["evidences"].items())
    critique = state.get("critique", "")
    revise_note = (f"\n\nCritic 上一轮审查意见（必须逐条修正）：\n{critique}"
                   if critique and state.get("verdict") == "REVISE" else "")
    resp = await llm.ainvoke([SystemMessage(content=(
        "你是 Synthesizer，负责把多个专家的证据综合成一份中文回答。"
        "要求：1) 只使用证据中的事实与数字，禁止新增、四舍五入或自行计算数字；"
        "2) 结构清晰（先结论后细节）；3) 200 字以内，语言简洁。")),
        HumanMessage(content=f"用户问题：{state['question']}\n\n"
                             f"{evidence_text}{revise_note}")])
    return {"draft": resp.content,
            "trace": [f"[Synthesizer] 草稿: {resp.content[:150]}..."]}


# ───────────────────────────── Critic 硬校验 ─────────────────────────────
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:,\d{3})*(?:\.\d+)?[%％]?")


def extract_numbers(text: str) -> list[str]:
    """提取事实数字；去掉有序列表编号，统一千分位和百分号。"""
    values = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\d+[.、)]\s*", "", line)
        for match in NUMBER_RE.finditer(line):
            values.append(match.group().replace(",", "").replace("％", "%"))
    return values


def hard_validate_numbers(draft: str, evidence_text: str) -> tuple[bool, list[str], int]:
    """字符串级程序闸：草稿中的每个数字都必须出现在证据中。"""
    draft_numbers = extract_numbers(draft)
    evidence_numbers = set(extract_numbers(evidence_text))
    missing = sorted({n for n in draft_numbers if n not in evidence_numbers})
    return not missing, missing, len(draft_numbers)


async def critic(state: State) -> dict:
    evidence_text = "\n".join(f"【{k}】{v}" for k, v in state["evidences"].items())
    hard_ok, missing, number_count = hard_validate_numbers(state["draft"], evidence_text)
    n = state.get("revision_count", 0)
    hard_trace = (f"[Critic·硬校验] PASS | {number_count} 个数字均可在证据中逐项找到"
                  if hard_ok else
                  f"[Critic·硬校验] FAIL | 无证据数字: {', '.join(missing)}")

    if not hard_ok:
        comments = ("程序硬校验发现回答含证据中没有的数字："
                    f"{', '.join(missing)}。请删除这些数字或按证据原文修正，不得自行计算。")
        if n < 1:
            return {"verdict": "REVISE", "critique": comments,
                    "hard_check": "FAIL: " + ", ".join(missing),
                    "revision_count": n + 1,
                    "trace": [hard_trace, "[Critic] REVISE | 硬校验未通过，回流修订"]}
        return {"verdict": "FAIL", "critique": comments,
                "hard_check": "FAIL: " + ", ".join(missing),
                "final": "【程序硬校验未通过，答案已阻断】\n" + state["draft"],
                "trace": [hard_trace, "[Critic] FAIL | 已达修订上限，阻断输出"]}

    resp = await llm.ainvoke([SystemMessage(content=(
        "你是 Critic，审查 Synthesizer 的回答质量。程序已确认所有数字都能在证据中找到，"
        "你继续检查：1) 结论是否与证据一致；2) 是否直接回答了问题；"
        "3) 有无夸大（如把小样本结论说成普遍规律）。"
        "输出 JSON：{\"verdict\": \"PASS\" 或 \"REVISE\", \"comments\": \"具体理由"
        "（REVISE 时给出可执行修改建议）\"}。证据充分且回答切题才 PASS。")),
        HumanMessage(content=f"问题：{state['question']}\n\n"
                             f"证据：\n{evidence_text}\n\n待审回答：\n{state['draft']}")])
    try:
        result = json.loads(re.search(r"\{.*\}", resp.content, re.S).group())
        verdict = result.get("verdict", "REVISE").upper()
        if verdict not in ("PASS", "REVISE"):
            verdict = "REVISE"
        comments = result.get("comments", "")
    except Exception:
        verdict, comments = "REVISE", "（审查 JSON 解析失败，按未通过处理）"

    trace = [hard_trace, f"[Critic] {verdict} | {comments}"]
    if verdict == "REVISE" and n < 1:
        return {"verdict": verdict, "critique": comments,
                "hard_check": "PASS", "revision_count": n + 1, "trace": trace}
    if verdict == "REVISE":
        return {"verdict": "FAIL", "critique": comments,
                "hard_check": "PASS", "final": "【审查未通过，答案已阻断】\n" + state["draft"],
                "trace": trace + ["[Critic] FAIL | 已达修订上限，阻断输出"]}
    return {"verdict": "PASS", "critique": comments, "hard_check": "PASS",
            "final": state["draft"], "trace": trace + ["[Critic] 审查通过，输出最终答案"]}


def after_critic(state: State) -> str:
    return "synthesizer" if state["verdict"] == "REVISE" else END


# ───────────────────────────── 构图 ─────────────────────────────
graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("sql_agent", sql_agent)
graph.add_node("mining_agent", mining_agent)
graph.add_node("rag_agent", rag_agent)
graph.add_node("synthesizer", synthesizer)
graph.add_node("critic", critic)
graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", route_selector,
                            ["sql_agent", "mining_agent", "rag_agent"])
for node in ("sql_agent", "mining_agent", "rag_agent"):
    graph.add_edge(node, "synthesizer")
graph.add_edge("synthesizer", "critic")
graph.add_conditional_edges("critic", after_critic, ["synthesizer", END])
app = graph.compile()


def initial_state(question: str) -> dict:
    return {"question": question, "routes": [], "evidences": {}, "tool_evidences": {},
            "draft": "", "verdict": "", "critique": "", "hard_check": "",
            "revision_count": 0, "final": "", "elapsed_seconds": 0.0, "trace": []}


async def arun(question: str) -> State:
    print(f"\n{'=' * 66}\n问：{question}\n{'=' * 66}")
    start = time.perf_counter()
    final_state = await app.ainvoke(initial_state(question))
    final_state["elapsed_seconds"] = time.perf_counter() - start
    for line in final_state["trace"]:
        print("  " + line)
    print(f"\n最终回答（{final_state['verdict']}，{final_state['elapsed_seconds']:.2f}s）："
          f"\n{final_state['final']}")
    return final_state


def run(question: str) -> State:
    """同步入口，保留 PyCharm 直接运行的使用方式。"""
    return asyncio.run(arun(question))


async def benchmark_parallel() -> dict:
    """同一 mining+rag 扇出分别串行/并行执行，测量专家取证阶段墙钟时间。"""
    state = initial_state("给我一份用户偏好模式简报，并说明挖掘方法。")
    start = time.perf_counter()
    await mining_agent(state)
    await rag_agent(state)
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    await asyncio.gather(mining_agent(state), rag_agent(state))
    parallel = time.perf_counter() - start
    result = {
        "branches": ["mining", "rag"],
        "sequential_seconds": round(sequential, 3),
        "parallel_seconds": round(parallel, 3),
        "speedup": round(sequential / parallel, 3) if parallel else None,
        "time_reduction_pct": round((sequential - parallel) / sequential * 100, 1)
        if sequential else None,
    }
    print("\n=== 物理并行对照（mining + rag 专家取证）===")
    print(f"串行 {sequential:.2f}s → 并行 {parallel:.2f}s | "
          f"加速比 {result['speedup']:.2f}x | 耗时下降 {result['time_reduction_pct']:.1f}%")
    with (OUT / "parallel_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


async def main():
    print(app.get_graph().draw_ascii())
    print(f"RAG 检索后端：{kb.mode} | 知识文档：{len(kb.ids)} 条")
    await benchmark_parallel()
    questions = [
        "评分超过 4.0 且至少有 100 人评价的电影有哪些？",
        "用户分群的结果怎么样？喜欢科幻片的人还喜欢什么类型？",
        "为什么 K-Means 选 K=2？PCA 为什么不参与聚类训练？",
        "给我一份关于 MovieLens 用户偏好模式的简报，并说明挖掘方法。",
    ]
    states = [await arun(q) for q in questions]
    log_path = OUT / "run_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        for s in states:
            f.write(f"问：{s['question']}\n")
            f.write("\n".join(s["trace"]) + f"\n最终回答（{s['verdict']}）：\n{s['final']}\n"
                    f"耗时：{s['elapsed_seconds']:.3f}s | 硬校验：{s['hard_check']}\n\n")
    print(f"\n运行日志已保存：{log_path}")


if __name__ == "__main__":
    asyncio.run(main())
