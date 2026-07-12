"""
data_quality_node — Katman 2 (Anlamsal) LangGraph adapter.

İnce glue: state'ten okur, `src.services.data_quality.run_data_quality` çağırır,
sonucu (data_quality_status + data_quality_report) partial return eder. Tüm mantık
serviste (node/service split, bkz. CLAUDE.md).

Konum: execution_node → data_quality_node → route_after_data_quality
  status=="failed" → report_node (kısa-devre; aksiyona dönük mesaj)
  aksi halde       → mevcut route_after_execution DEĞİŞMEDEN devreder.
"""

from src.core.nodes import node
from src.state import AgentState
from src.services.data_quality import run_data_quality


@node("data_quality_node")
def data_quality_node(state: AgentState) -> AgentState:
    result = run_data_quality(dict(state))
    return {
        "data_quality_status": result["data_quality_status"],
        "data_quality_report": result["data_quality_report"],
        "current_step": "data_quality_node",
    }
