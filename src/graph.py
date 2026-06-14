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
from src.nodes.critic_node import critic_node
from src.nodes.report_node import report_node
from src.nodes.sql_node import sql_node
from src.nodes.target_discovery_node import target_discovery_node
from src.nodes.leakage_detector_node import leakage_detector_node
from src.nodes.entity_resolution_node import entity_resolution_node
from src.nodes.analysis_node import analysis_node


# Routing functions
def route_after_db_context(state: AgentState) -> str:
    intent = state.get("intent", "eda")
    if intent == "sql_query":
        return "sql_node"
    return "target_discovery_node"


def route_after_execution(state: AgentState) -> str:
    """
    split for execution node after
      classification / regression / clustering → entity_resolution_node
      deep_analysis                            → analysis_node (model akışına girmez)
      eda / anomaly_detection / report         → automl_node
    """
    intent = state.get("intent", "eda")
    if intent == "deep_analysis":
        return "analysis_node"
    if intent in ("classification", "regression", "clustering"):
        return "entity_resolution_node"
    if intent == "anomaly_detection":
        return "automl_node"
    return "report_node"


def route_after_critic(state: AgentState) -> str:
    """
    split for critic node after
      retry varsa -> next_retry_node (feature_plan_node veya automl_node)
      yoksa -> report_node
    """
    return state.get("next_retry_node", "report_node")


from typing import Optional
from langgraph.checkpoint.base import BaseCheckpointSaver

# Graph
def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    Data Scientist Agent — LangGraph pipeline.

    Routing:
        START
          └─ intent_node
               └─ db_context_node          
                    ├─ [sql_query] → sql_node → END
                    └─ [other] → planner_node
                                      → target_discovery_node
                                           → execution_node
                                                 ├─ [classification/regression/clustering]
                                                 │    → entity_resolution_node → leakage_detector_node → feature_plan_node
                                                 │         → feature_engineering_node → automl_node → evaluation_node → critic_node
                                                 │              ├─ [retry] → leakage_detector_node / feature_plan_node / automl_node
                                                 │              └─ [pass]  → report_node → END
                                                 └─ [eda/anomaly_detection/report]
                                                      → automl_node → evaluation_node → critic_node → report_node → END
    """
    graph = StateGraph(AgentState)

    # Register Nodes 
    graph.add_node("intent_node",              intent_node)
    graph.add_node("db_context_node",          db_context_node)
    graph.add_node("sql_node",                 sql_node)
    graph.add_node("planner_node",             planner_node)
    graph.add_node("target_discovery_node",    target_discovery_node)
    graph.add_node("execution_node",           execution_node)
    graph.add_node("feature_plan_node",        feature_plan_node)
    graph.add_node("feature_engineering_node", feature_engineering_node)
    graph.add_node("automl_node",              automl_node)
    graph.add_node("evaluation_node",          evaluation_node)
    graph.add_node("critic_node",              critic_node)
    graph.add_node("report_node",              report_node)
    graph.add_node("leakage_detector_node",    leakage_detector_node)
    graph.add_node("entity_resolution_node",   entity_resolution_node)
    graph.add_node("analysis_node",            analysis_node)

    # START → intent_node → db_context_node
    graph.add_edge(START, "intent_node")
    graph.add_edge("intent_node", "db_context_node")

    # db_context_node → sql_node or planner_node
    graph.add_conditional_edges(
        "db_context_node",
        route_after_db_context,
        {
            "sql_node":              "sql_node",
            "target_discovery_node": "target_discovery_node",   # planner_node yerine
        },
)


    # sql_node → END
    graph.add_edge("sql_node", END)

    # ML pipeline
    graph.add_edge("planner_node", "execution_node")

    graph.add_conditional_edges(
        "execution_node",
        route_after_execution,
        {
            "entity_resolution_node": "entity_resolution_node",
            "analysis_node": "analysis_node",
            "automl_node":       "automl_node",
            "report_node":       "report_node",
        },
    )
    graph.add_edge("analysis_node", END)


    graph.add_edge("entity_resolution_node", "leakage_detector_node")
    graph.add_edge("leakage_detector_node", "feature_plan_node")
    graph.add_edge("feature_plan_node", "feature_engineering_node")
    graph.add_edge("feature_engineering_node", "automl_node")
    graph.add_edge("automl_node",              "evaluation_node")
    graph.add_edge("evaluation_node",          "critic_node")
    
    graph.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {
            "feature_plan_node": "feature_plan_node",
            "automl_node": "automl_node",
            "report_node": "report_node",
        }
    )
    
    graph.add_edge("target_discovery_node", "planner_node")
    graph.add_edge("report_node",              END)

    return graph.compile(checkpointer=checkpointer)