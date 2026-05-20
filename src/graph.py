from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.state import AgentState
from src.nodes.intent_node import intent_node
from src.nodes.db_context_node import db_context_node
from src.nodes.planner_node import planner_node
from src.nodes.execution_node import execution_node
from src.nodes.feature_plan_node import feature_plan_node
from src.nodes.feature_engineering_node import feature_engineering_node
from src.nodes.automl_node import automl_node
from src.nodes.evaluation_node import evaluation_node
from src.nodes.report_node import report_node

def build_graph() -> StateGraph:
    """
    Data Scientist Agent — LangGraph pipeline.

    Active pipeline:
        START → intent_node → db_context_node → planner_node
        → execution_node → feature_plan_node → feature_engineering_node
        → automl_node → evaluation_node → report_node → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent_node", intent_node)
    graph.add_node("db_context_node", db_context_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("execution_node", execution_node)
    graph.add_node("feature_plan_node", feature_plan_node)
    graph.add_node("feature_engineering_node", feature_engineering_node)
    graph.add_node("automl_node", automl_node)
    graph.add_node("evaluation_node", evaluation_node)
    graph.add_node("report_node", report_node)

    graph.add_edge(START, "intent_node")
    graph.add_edge("intent_node", "db_context_node")
    graph.add_edge("db_context_node", "planner_node")
    graph.add_edge("planner_node", "execution_node")
    graph.add_edge("execution_node", "feature_plan_node")
    graph.add_edge("feature_plan_node", "feature_engineering_node")
    graph.add_edge("feature_engineering_node", "automl_node")
    graph.add_edge("automl_node", "evaluation_node")
    graph.add_edge("evaluation_node", "report_node")
    graph.add_edge("report_node", END)

    return graph.compile()


# Conditional routing logic
def route_after_execution(state: AgentState) -> str:
    """
    Routes flow based on intent after execution node.
    EDA/Report → Direct to evaluation
    ML intents → feature_engineering
    """
    intent = state.get("intent", "eda")
    if intent in ("classification", "regression", "clustering"):
        return "feature_engineering"
    return "stats_only"
