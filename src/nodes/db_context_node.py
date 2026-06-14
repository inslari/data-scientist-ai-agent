from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from src.core.config import settings
from src.intelligence_engine import DatabaseIntelligenceEngine  # type: ignore[import-not-found]
from src.state import AgentState  # type: ignore[import-not-found]

from api.core.database import SessionLocal
from api.schemas.data_group import get_data_group_schema_logic, run_data_group_query_logic, QueryRequest
from api.models.data_group import DataGroupModel

async def db_context_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Async node — LangGraph ainvoke ile uyumlu."""
    print("\n========================================\n[NODE START] db_context_node başlıyor...\n========================================")
    
    print(">>> db_context_node basladi!")
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        raise ValueError("data_group_id is missing in config")

    # Fetch schema + sample data
    print(">>> DataGroup'dan sema ve ornek veri cekiliyor...")
    raw_schema, sample_data = await _fetch_schema_and_samples(data_group_id)
    print(">>> Sema basariyla cekildi. Uzunluk:", len(raw_schema))

    engine   = DatabaseIntelligenceEngine()
    print(">>> DatabaseIntelligenceEngine baslatildi. Sema analizi icin LLM'e istek atiliyor (Bu islem uzun surebilir)...")
    try:
        db_model = engine.analyze_schema(
            schema_text=raw_schema,
            sample_data=sample_data,
        )
    except Exception as e:
        print(f">>> LLM analizi basarisiz oldu: {str(e)[:200]}")
        print(">>> Fallback moduna geciliyor (ornek veri olmadan tekrar deneniyor)...")
        db_model = engine.analyze_schema(
            schema_text=raw_schema,
            sample_data="Örnek veri çok uzun olduğu için LLM halüsinasyonuna yol açtı, çıkarıldı.",
        )
    print(">>> LLM analizi basariyla tamamlandi!")


    intent          = state.get("intent", "eda")
    relevant_tables = _filter_tables_by_intent(db_model, intent)

    return {
        **state,
        "raw_schema":      raw_schema,
        "db_intelligence": db_model.model_dump(),
        "relevant_tables": relevant_tables,
        "current_step":    "db_context_node",
    }


async def _fetch_schema_and_samples(data_group_id: int) -> tuple[str, str]:
    db = SessionLocal()
    try:
        dg = db.query(DataGroupModel).filter(DataGroupModel.id == data_group_id).first()
        if not dg:
            raise ValueError(f"DataGroup {data_group_id} not found")
        user_id = dg.user_id

        schema_result = await get_data_group_schema_logic(data_group_id, db, user_id)
        raw_schema = schema_result.get("schema", "")
        
        table_info = []
        for line in raw_schema.splitlines():
            line_str = line.strip()
            if line_str.startswith("- Table '"):
                try:
                    table_name = line_str.split("'")[1]
                    table_info.append(table_name)
                except IndexError:
                    pass
            elif line_str.startswith("Table: "):
                table_name = line_str.split("Table: ")[1].strip()
                table_info.append(table_name)
                
        sample_parts = []
        for table in table_info[:5]:
            try:
                query = f'SELECT * FROM "{table}" LIMIT 3'
                req = QueryRequest(query=query)
                r = await run_data_group_query_logic(data_group_id, req, db, user_id)
                if r.get("status") == "success":
                    r_text = str(r.get("result", ""))
                    if len(r_text) > 500:
                        r_text = r_text[:500] + "\n... (truncated)"
                    sample_parts.append(f"Table: {table}\n{r_text}")
            except Exception:
                pass

        sample_data = "\n\n".join(sample_parts) if sample_parts else "No sample data."
        return raw_schema, sample_data
    finally:
        db.close()


def _filter_tables_by_intent(db_model, intent: str) -> list[str]:
    scored = sorted(
        db_model.tables,
        key=lambda t: t.analytics_potential_score,
        reverse=True,
    )
    if intent in ("eda", "report"):
        return [t.table_name for t in scored]
    return [t.table_name for t in scored[:3]]