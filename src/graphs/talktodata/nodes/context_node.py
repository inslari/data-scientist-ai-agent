from langchain_core.runnables import RunnableConfig

from src.intelligence_engine import DatabaseIntelligenceEngine
from api.core.database import SessionLocal
from api.schemas.data_group import (
    get_data_group_schema_logic as get_data_group_schema,
    run_data_group_query_logic as run_data_group_query,
    QueryRequest as DGQueryRequest,
)

from ..state import TTDState

async def ttd_db_context_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    Schema + db_intelligence üretir.
    State'te zaten db_intelligence varsa (cache hit) atlar.
    """
    print("\n[TTD] ttd_db_context_node başlıyor...")

    if state.get("db_intelligence") and state.get("raw_schema"):
        print("[TTD] db_context: cache hit, skipping LLM analysis")
        return {**state, "current_step": "ttd_db_context_node"}  # type: ignore[return-value]

    data_group_id = config.get("configurable", {}).get("data_group_id") or state.get("data_group_id")
    user_id = config.get("configurable", {}).get("user_id") or state.get("user_id")
    if not data_group_id or not user_id:
        return {**state, "error": "data_group_id or user_id missing", "current_step": "ttd_db_context_node"}  # type: ignore[return-value]

    db = SessionLocal()
    try:
        schema_result = await get_data_group_schema(data_group_id, db, user_id)
        raw_schema = schema_result.get("schema", "")

        # sample data
        table_names = []
        for line in raw_schema.splitlines():
            ls = line.strip()
            if ls.startswith("- Table '"):
                try:
                    table_names.append(ls.split("'")[1])
                except IndexError:
                    pass
            elif ls.startswith("Table: "):
                table_names.append(ls.split("Table: ")[1].strip())

        sample_parts = []
        for tname in table_names[:5]:
            try:
                req = DGQueryRequest(query=f'SELECT * FROM "{tname}" LIMIT 3')
                r = await run_data_group_query(data_group_id, req, db, user_id)
                if r.get("status") == "success":
                    txt = str(r.get("result", ""))[:500]
                    sample_parts.append(f"Table: {tname}\n{txt}")
            except Exception:
                pass
        sample_data = "\n\n".join(sample_parts) or "No sample data."
    finally:
        db.close()

    engine = DatabaseIntelligenceEngine()
    try:
        db_model = engine.analyze_schema(schema_text=raw_schema, sample_data=sample_data)
    except Exception:
        db_model = engine.analyze_schema(schema_text=raw_schema, sample_data="Sample data omitted.")

    print("[TTD] db_context: LLM analysis complete")
    return {
        **state,
        "raw_schema":      raw_schema,
        "db_intelligence": db_model.model_dump(),
        "current_step":    "ttd_db_context_node",
    }  # type: ignore[return-value]
