from langgraph.graph import END, START, StateGraph

from .state import TTDState
from .nodes.context_node import ttd_db_context_node
from .nodes.intent_node import ttd_intent_classifier_node
from .nodes.sql_node import ttd_sql_node
from .nodes.validator_node import ttd_result_validator_node
from .nodes.chart_node import ttd_chart_node
from .nodes.interpreter_node import ttd_interpreter_node

def _route_after_intent(state: TTDState) -> str:
    """
    Intent sınıflandırmasına göre routing:
    - clarification_needed → interpreter (soruyu sor)
    - chitchat / schema_question → interpreter (SQL gerekmez)
    - sql_query / analytical → sql_node
    """
    if state.get("clarification_needed"):
        return "ttd_interpreter_node"
    intent = state.get("intent") or "sql_query"
    if intent in ("chitchat", "schema_question"):
        return "ttd_interpreter_node"
    return "ttd_sql_node"


def _route_after_sql(state: TTDState) -> str:
    """SQL hatası varsa retry, yoksa validator'a git."""
    if state.get("sql_error") and (state.get("sql_retry_count") or 0) < 3:
        return "ttd_sql_node"
    return "ttd_result_validator_node"


def _route_after_validator(state: TTDState) -> str:
    """
    Validator sonucuna göre routing:
    - Geçersiz + retry hakkı varsa + empty_result DEĞİLSE → sql_node
    - empty_result: veri gerçekten yok, retry işe yaramaz → direkt chart_node
    - Geçerli veya retry hakkı bittiyse → chart_node
    """
    retry_count = state.get("sql_retry_count") or 0
    issue = state.get("validation_issue")
    if not state.get("validation_passed") and issue != "empty_result" and retry_count < 3:
        print(f"[TTD] validator rejected results (issue={issue}), retrying SQL (attempt {retry_count}/{3})")
        return "ttd_sql_node"
    if issue == "empty_result":
        print(f"[TTD] empty result — veri yok, retry atlanıyor, interpreter'a gidiliyor")
    return "ttd_chart_node"


def build_ttd_graph():
    """
    Talk to Data LangGraph pipeline.

    START → db_context → intent_classifier
                              ├─ [chitchat/schema] → interpreter → END
                              └─ [sql/analytical]  → sql_node
                                                         └─ result_validator
                                                              ├─ [invalid] → sql_node (retry)
                                                              └─ [valid]   → chart → interpreter → END
    """
    g = StateGraph(TTDState)  # type: ignore[type-var]

    g.add_node("ttd_db_context_node",        ttd_db_context_node)
    g.add_node("ttd_intent_classifier_node", ttd_intent_classifier_node)
    g.add_node("ttd_sql_node",               ttd_sql_node)
    g.add_node("ttd_result_validator_node",  ttd_result_validator_node)
    g.add_node("ttd_chart_node",             ttd_chart_node)
    g.add_node("ttd_interpreter_node",       ttd_interpreter_node)

    g.add_edge(START,                          "ttd_db_context_node")
    g.add_edge("ttd_db_context_node",          "ttd_intent_classifier_node")
    g.add_conditional_edges("ttd_intent_classifier_node", _route_after_intent)
    g.add_conditional_edges("ttd_sql_node",               _route_after_sql)
    g.add_conditional_edges("ttd_result_validator_node",  _route_after_validator)
    g.add_edge("ttd_chart_node",               "ttd_interpreter_node")
    g.add_edge("ttd_interpreter_node",         END)

    return g.compile()
