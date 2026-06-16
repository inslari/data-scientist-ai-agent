from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from src.state import AgentState
from src.domains import get_active_pack


async def merchant_intelligence_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Run the active pack's tool-using expert agents over the real dataset.

    Produces ``merchant_analysis`` (data-grounded findings) consumed by feature
    engineering / SQL / report touch points. If no pack is active, or anything
    fails, the node passes through and the hooks fall back to static guidance —
    the pipeline never breaks.
    """
    print("\n========================================\n[NODE START] merchant_intelligence_node başlıyor...\n========================================")

    pack = get_active_pack(state.get("domain_pack"))
    if pack.name == "none":
        return {**state, "current_step": "merchant_intelligence_node"}

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

        print(f"[merchant_node] pack='{pack.name}' data_group={data_group_id} source={source_type} → expert ajanları çalışıyor...")
        analysis = await pack.analyze(
            user_query=state.get("user_query", ""),
            db_intelligence=state.get("db_intelligence"),
            domain_doc=state.get("domain_doc", "") or "",
            data_group_id=data_group_id,
            user_id=user_id,
            source_type=source_type,
        )
        if analysis is not None:
            print(f"[merchant_node] analiz tamam → expert(ler)={analysis.council}, bulgu={len(analysis.findings)}")
            return {
                **state,
                "merchant_analysis": analysis.to_dict(),
                "current_step": "merchant_intelligence_node",
            }
        print("[merchant_node] analiz None döndü; statik ipuçlarına düşülecek.")
    except Exception as e:  # pragma: no cover - defensive
        print(f"[merchant_node] analiz başarısız ({e}); statik ipuçlarına düşülüyor.")

    return {**state, "current_step": "merchant_intelligence_node"}
