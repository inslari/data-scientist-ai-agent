from __future__ import annotations
from src.core.nodes import node

import time as _time
from langchain_core.runnables import RunnableConfig

from src.state import AgentState
from src.domains import get_active_pack

from src.core.logging_config import get_logger
logger = get_logger(__name__)


def _save_discovered_queries(analysis, data_group_id: int) -> None:
    """Persist all discovered_queries from expert findings to data_group_queries table."""
    all_queries = [
        (finding.expert, dq)
        for finding in analysis.findings
        for dq in finding.discovered_queries
        if dq.sql.strip()
    ]
    if not all_queries:
        return
    try:
        from api.core.database import SessionLocal
        from api.models.data_group_query import DataGroupQueryModel
        db = SessionLocal()
        try:
            for expert_key, dq in all_queries:
                db.add(DataGroupQueryModel(
                    data_group_id=data_group_id,
                    title=dq.title,
                    description=dq.description or None,
                    sql=dq.sql,
                    expert=expert_key,
                    pack_name=analysis.pack_name,
                    created_at=int(_time.time()),
                ))
            db.commit()
            logger.info(f"[merchant_node] {len(all_queries)} keşfedilen sorgu data_group {data_group_id} için kaydedildi.")
        finally:
            db.close()
    except Exception as e:  # pragma: no cover - defensive
        logger.info(f"[merchant_node] Sorgular kaydedilemedi: {e}")


@node("merchant_intelligence_node")
async def merchant_intelligence_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Run the active pack's tool-using expert agents over the real dataset.

    Produces ``merchant_analysis`` (data-grounded findings) consumed by feature
    engineering / SQL / report touch points. If no pack is active, or anything
    fails, the node passes through and the hooks fall back to static guidance —
    the pipeline never breaks.
    """

    pack = get_active_pack(state.get("domain_pack"))
    if pack.name == "none":
        return {"current_step": "merchant_intelligence_node"}

    data_group_id = config.get("configurable", {}).get("data_group_id")

    try:
        # Resolve the data group's owner + source type (for the SQL tool + dialect).
        user_id = None
        source_type = ""
        if data_group_id:
            from api.core.database import SessionLocal
            from api.models.data_group import DataGroupModel
            db = SessionLocal()
            try:
                dg = db.query(DataGroupModel).filter(DataGroupModel.id == data_group_id).first()
                if dg:
                    user_id = dg.user_id
                    source_type = dg.source_type or ""
            finally:
                db.close()

        logger.info(f"[merchant_node] pack='{pack.name}' data_group={data_group_id} source={source_type} → expert ajanları çalışıyor...")
        analysis = await pack.analyze(
            user_query=state.get("user_query", ""),
            db_intelligence=state.get("db_intelligence"),
            domain_doc=state.get("domain_doc", "") or "",
            data_group_id=data_group_id,
            user_id=user_id,
            source_type=source_type,
        )
        if analysis is not None:
            logger.info(f"[merchant_node] analiz tamam → expert(ler)={analysis.council}, bulgu={len(analysis.findings)}")
            if data_group_id:
                _save_discovered_queries(analysis, data_group_id)
            return {
                "merchant_analysis": analysis.to_dict(),
                "current_step": "merchant_intelligence_node",
            }
        logger.info("[merchant_node] analiz None döndü; statik ipuçlarına düşülecek.")
    except Exception as e:  # pragma: no cover - defensive
        logger.info(f"[merchant_node] analiz başarısız ({e}); statik ipuçlarına düşülüyor.")

    return {"current_step": "merchant_intelligence_node"}
