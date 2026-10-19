import os

from dotenv import load_dotenv

load_dotenv()   # 从项目根 .env 读取 DEEPSEEK_API_KEY

from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# ── ① State：全体 Agent 共享的"记忆字典" ──
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ── ② Node：一个节点 = 一个 Agent，读 State、写 State ──
@tool
def movie_stats(query: str) -> str:
    """查询电影数据统计（演示用，W2 接真数据）"""
    return "MovieLens: 100,836 条评分 | 610 用户 | 9,724 部电影"

tools = [movie_stats]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def execute_tools(state: State):
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        output = movie_stats.invoke(call["args"])
        # 工具结果必须包成 ToolMessage 并带上 tool_call_id，才能配对回 assistant 的调用
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}

# ── ③ Edge：路由函数 = 协作规则（决定下一个 Agent 是谁）──
def should_use_tool(state: State) -> str:
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", execute_tools)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tool, ["tools", END])
graph.add_edge("tools", "chatbot")          # 工具结果回流，形成环
app = graph.compile()

print(app.get_graph().draw_ascii())         # 打印图结构

for question in ["MovieLens 数据集有多少条评分？", "你好，简单介绍下自己"]:
    print(f"\n{'='*50}\n问: {question}")
    for chunk in app.stream({"messages": [("user", question)]}):
        for node_out in chunk.values():
            if node_out.get("messages"):
                last = node_out["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print(f"答: {last.content}")
