from langgraph.graph import END, START, StateGraph

from .state import VizState
from .nodes.plan_node import viz_plan_node
from .nodes.sql_node import viz_sql_node
from .nodes.render_node import viz_render_node
from .nodes.interpreter_node import viz_interpreter_node

# Context ve validator: Talk-to-Data node'ları aynı state anahtarlarıyla çalışır,
# yeniden yazmak yerine doğrudan yeniden kullanılır (DRY).
from src.graphs.talktodata.nodes.context_node import ttd_db_context_node
from src.graphs.talktodata.nodes.validator_node import ttd_result_validator_node

from src.core.logging_config import get_logger
logger = get_logger(__name__)

_MAX_RETRIES = 3


def _route_after_plan(state: VizState) -> str:
    """Belirsizse interpreter'a (soruyu sor), yoksa SQL'i çalıştır."""
    if state.get("clarification_needed"):
        return "viz_interpreter_node"
    return "viz_sql_node"


def _route_after_sql(state: VizState) -> str:
    """SQL hatası + retry hakkı varsa plan'a dön (yeniden üret), yoksa validator'a."""
    if state.get("sql_error") and (state.get("sql_retry_count") or 0) < _MAX_RETRIES:
        return "viz_plan_node"
    if state.get("sql_error"):
        # retry bitti — yine de render denenmesin, interpreter hata mesajı versin
        return "viz_interpreter_node"
    return "viz_validator_node"


def _route_after_validator(state: VizState) -> str:
    """
    - Geçersiz + retry hakkı + empty_result DEĞİL → plan'a dön (yeni spec/SQL)
    - empty_result: veri yok, retry işe yaramaz → render (boş/uyarı yorumu)
    - Geçerli / retry bitti → render
    """
    retry_count = state.get("sql_retry_count") or 0
    issue = state.get("validation_issue")
    if not state.get("validation_passed") and issue != "empty_result" and retry_count < _MAX_RETRIES:
        logger.info(f"[VIZ] validator rejected (issue={issue}), replanning (attempt {retry_count}/{_MAX_RETRIES})")
        return "viz_plan_node"
    if issue == "empty_result":
        logger.warning("[VIZ] empty result — veri yok, interpreter'a gidiliyor")
        return "viz_interpreter_node"
    return "viz_render_node"


def build_dataviz_graph():
    """
    Data Visualization LangGraph pipeline (Talk-to-Data'nın görselleştirme muadili).

    START → context → plan
                        ├─ [clarification] → interpreter → END
                        └─ sql → validator
                                   ├─ [invalid & retry] → plan
                                   ├─ [empty]           → interpreter → END
                                   └─ [valid]           → render → interpreter → END
    """
    g = StateGraph(VizState)  # type: ignore[type-var]

    g.add_node("viz_context_node",     ttd_db_context_node)       # reuse
    g.add_node("viz_plan_node",        viz_plan_node)
    g.add_node("viz_sql_node",         viz_sql_node)
    g.add_node("viz_validator_node",   ttd_result_validator_node) # reuse
    g.add_node("viz_render_node",      viz_render_node)
    g.add_node("viz_interpreter_node", viz_interpreter_node)

    g.add_edge(START, "viz_context_node")
    g.add_edge("viz_context_node", "viz_plan_node")
    g.add_conditional_edges("viz_plan_node",      _route_after_plan,      ["viz_sql_node", "viz_interpreter_node"])
    g.add_conditional_edges("viz_sql_node",       _route_after_sql,       ["viz_plan_node", "viz_validator_node", "viz_interpreter_node"])
    g.add_conditional_edges("viz_validator_node", _route_after_validator, ["viz_plan_node", "viz_render_node", "viz_interpreter_node"])
    g.add_edge("viz_render_node", "viz_interpreter_node")
    g.add_edge("viz_interpreter_node", END)

    return g.compile()
