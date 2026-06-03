import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Any
from sklearn.ensemble import RandomForestClassifier

class State(TypedDict):
    model: Any

def node_a(state: State):
    print("Node A running")
    return {"model": RandomForestClassifier()}

def node_b(state: State):
    print("Node B running")
    print("Model is:", state.get("model"))
    return {}

graph = StateGraph(State)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)

async def test():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        app = graph.compile(checkpointer=saver)
        try:
            async for chunk in app.astream({}, config={"configurable": {"thread_id": "1"}}):
                print("Chunk:", chunk.keys())
        except Exception as e:
            print("PIPELINE CRASHED:", type(e), e)

asyncio.run(test())
