from src.core.nodes import node
"""
Entity Resolution Node
=======================
execution_node'dan sonra, leakage_detector_node'dan önce çalışır.

Görev:
  1. Entity-level veri kalıplarını tespit et
  2. Gerekirse aggregate et (yeni parquet yaz, data_path güncelle)
  3. Sonuçları state'e yaz

Akış:
  execution_node → entity_resolution_node → leakage_detector_node → ...
"""

from src.state import AgentState
from src.services.entity_resolver import resolve_entities

from src.core.logging_config import get_logger
logger = get_logger(__name__)


@node("entity_resolution_node")
def entity_resolution_node(state: AgentState) -> AgentState:
    """
    Entity-level veri kalıplarını tespit eder ve gerekirse aggregate eder.

    Girdiler:
        data_path         — parquet dosya yolu
        target_definition — {column_name, ...}
        db_intelligence   — {columns, tables, relationships, ...}

    Çıktılar:
        entity_resolution — EntityResolution raporu (dict)
        data_path         — güncellenmiş parquet path (aggregate ise yeni)
    """

    result = resolve_entities(state)

    new_state = {
        "entity_resolution": result.get("entity_resolution", {}),
        "data_path": result.get("data_path", state.get("data_path", "")),
        "current_step": "entity_resolution_node",
    }

    # DEBUG
    resolution = result.get("entity_resolution", {})
    logger.info(f"[DEBUG ENTITY_RES NODE] action = {resolution.get('action', 'N/A')}")
    logger.info(f"[DEBUG ENTITY_RES NODE] message = {resolution.get('message', 'N/A')}")
    logger.info(f"[DEBUG ENTITY_RES NODE] new data_path = {new_state.get('data_path', 'N/A')}")

    return new_state
