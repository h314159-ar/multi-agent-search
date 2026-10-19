r"""W2 第 3 步：数据检索 Agent v1
= quickstart 的图结构 + 换工具（run_sql）+ 注入数据字典（system prompt）
运行方式：先设 key 再跑（同一个终端）
    $env:DEEPSEEK_API_KEY="sk-..."
    dsp\Scripts\python retriever_agent.py
"""
import os

import duckdb
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── ① State：全体 Agent 共享的"记忆字典"（与 quickstart 相同） ──
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ── ② 工具：movie_stats → run_sql（查真数据） ──
@tool
def run_sql(sql: str) -> str:
    """在 MovieLens 数据库上执行只读 SQL 查询，返回查询结果。"""
    con = duckdb.connect("data/movielens.duckdb", read_only=True)
    try:
        df = con.execute(sql).fetch_df()
        return df.head(20).to_string(index=False) if len(df) else "查询结果为空"
    except Exception as e:
        return f"SQL 执行失败: {e}"
    finally:
        con.close()

tools = [run_sql]
llm_with_tools = llm.bind_tools(tools)

# ── ③ 数据字典（system prompt）：让 LLM 写得出正确的 SQL ──
DATA_DICT = """你是 MovieLens 数据分析助手，通过调用 run_sql 工具查询数据库来回答问题。

数据库包含两张表：

【ratings 表】100,836 行评分记录
- userId: 用户ID（整数）
- movieId: 电影ID（整数，与 movies 表关联）
- rating: 评分，范围 0.5 ~ 5.0
- timestamp: Unix 时间戳
- rated_at: 评分时间（DATETIME）
- rated_year: 评分年份（整数）

【movies 表】9,742 行电影信息
- movieId: 电影ID
- title: 电影名（含年份，英文，如 'Shawshank Redemption, The (1994)'，支持 LIKE 模糊匹配）
- 其余 20 列为电影类型 0/1 标记：Action, Adventure, Animation, Children, Comedy,
  Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery,
  Romance, Sci-Fi, Thriller, War, Western, 以及 '(no genres listed)'

回答规则：
1. 必须调用 run_sql 工具查真实数据，禁止凭记忆编造数字
2. 电影名是英文：用户用中文提及时，先翻译成英文再用 LIKE 匹配
3. 需要电影名或类型信息时，用 movieId JOIN movies 表
4. 写标准 DuckDB SQL（PostgreSQL 方言）
"""

def chatbot(state: State):
    msgs = [SystemMessage(content=DATA_DICT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(msgs)]}

def execute_tools(state: State):
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        output = run_sql.invoke(call["args"])
        # 工具结果必须包成 ToolMessage 并带 tool_call_id 才能配对回 assistant 调用
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}

# ── ④ 条件边路由：决定下一个节点（与 quickstart 相同） ──
def should_use_tool(state: State) -> str:
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", execute_tools)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tool, ["tools", END])
graph.add_edge("tools", "chatbot")          # 工具结果回流，形成环
app = graph.compile()

print(app.get_graph().draw_ascii())         # 图结构（与 quickstart 同构）

# ── ⑤ 三道验收题 ──
questions = [
    "评分超过 4.0 且至少有 100 人评价的电影有哪些？",
    "2015 年以后，每年的评分数是多少？",
    "喜剧类（Comedy）电影里评分最高的 5 部是哪些？",
]
for q in questions:
    print(f"\n{'=' * 60}\n问: {q}")
    for chunk in app.stream({"messages": [("user", q)]}):
        for node_out in chunk.values():
            msgs = node_out.get("messages")
            if not msgs:
                continue
            last = msgs[-1]
            if getattr(last, "tool_calls", None):     # 显示 LLM 生成的 SQL
                print(f"  [Agent 生成 SQL] {last.tool_calls[0]['args'].get('sql', '')[:120]}...")
            elif hasattr(last, "content") and last.content and not isinstance(last, ToolMessage):
                print(f"答: {last.content}")
