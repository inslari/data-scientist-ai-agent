from __future__ import annotations
from src.core.nodes import node

import hashlib

from langchain_core.runnables import RunnableConfig

from src.core.config import settings
from src.intelligence_engine import DatabaseIntelligenceEngine, DatabaseIntelligenceModel  # type: ignore[import-not-found]
from src.state import AgentState  # type: ignore[import-not-found]

from api.core.database import SessionLocal
from api.schemas.data_group import get_data_group_schema_logic, run_data_group_query_logic, QueryRequest
from api.models.data_group import DataGroupModel
from api.models.db_intelligence_cache import DbIntelligenceCacheModel

from src.core.logging_config import get_logger
logger = get_logger(__name__)

@node("db_context_node")
async def db_context_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Async node — LangGraph ainvoke ile uyumlu."""
    
    logger.info(">>> db_context_node basladi!")
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        raise ValueError("data_group_id is missing in config")

    # Fetch schema + sample data
    logger.info(">>> DataGroup'dan sema ve ornek veri cekiliyor...")
    raw_schema, sample_data = await _fetch_schema_and_samples(data_group_id)
    logger.info(">>> Sema basariyla cekildi. Uzunluk: %s", len(raw_schema))

    # Project bazlı cache: aynı şema için pahalı analyze_schema LLM çağrısını atla.
    # thread_id == project_id (api/core/worker.py).
    project_id = _coerce_project_id(config.get("configurable", {}).get("thread_id"))
    cache_enabled = settings.ENABLE_INTELLIGENCE_CACHE and project_id is not None
    cache_key = _intelligence_cache_key(raw_schema) if cache_enabled else None

    db_model = None
    if cache_enabled:
        cached = _load_cached_intelligence(project_id, cache_key)
        if cached is not None:
            try:
                db_model = DatabaseIntelligenceModel.model_validate(cached)
                logger.warning(f">>> db_intelligence cache HIT (project {project_id}) — LLM analizi atlandi.")
            except Exception as e:  # bozuk/uyumsuz payload → yeniden hesapla
                logger.info(f">>> Cache payload gecersiz, yeniden hesaplaniyor: {str(e)[:150]}")
                db_model = None

    if db_model is None:
        engine = DatabaseIntelligenceEngine()
        logger.info(">>> DatabaseIntelligenceEngine baslatildi. Sema analizi icin LLM'e istek atiliyor (Bu islem uzun surebilir)...")
        try:
            db_model = engine.analyze_schema(
                schema_text=raw_schema,
                sample_data=sample_data,
            )
        except Exception as e:
            logger.info(f">>> LLM analizi basarisiz oldu: {str(e)[:200]}")
            logger.info(">>> Fallback moduna geciliyor (ornek veri olmadan tekrar deneniyor)...")
            db_model = engine.analyze_schema(
                schema_text=raw_schema,
                sample_data="Örnek veri çok uzun olduğu için LLM halüsinasyonuna yol açtı, çıkarıldı.",
            )
        logger.info(">>> LLM analizi basariyla tamamlandi!")
        if cache_enabled:
            _save_cached_intelligence(project_id, cache_key, db_model.model_dump())
            logger.info(f">>> db_intelligence cache yazildi (project {project_id}).")


    intent          = state.get("intent", "eda")
    relevant_tables = _filter_tables_by_intent(db_model, intent)

    return {
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


# ── Project-bazlı intelligence cache (application DB) ─────────
def _coerce_project_id(raw) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _intelligence_cache_key(raw_schema: str) -> str:
    """hash(intelligence model + raw_schema). Şema/model değişince eşleşmez → recompute."""
    payload = f"{settings.INTELLIGENCE_ENGINE_MODEL}\x00{raw_schema or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_intelligence(project_id: int, cache_key: str) -> dict | None:
    db = SessionLocal()
    try:
        row = db.query(DbIntelligenceCacheModel).filter(
            DbIntelligenceCacheModel.project_id == project_id
        ).first()
        if row and row.cache_key == cache_key:
            return row.payload
        return None
    except Exception as e:  # cache okuması asla pipeline'ı düşürmemeli
        logger.info(f">>> db_intelligence cache okunamadi: {str(e)[:150]}")
        return None
    finally:
        db.close()


def _save_cached_intelligence(project_id: int, cache_key: str, payload: dict) -> None:
    db = SessionLocal()
    try:
        row = db.query(DbIntelligenceCacheModel).filter(
            DbIntelligenceCacheModel.project_id == project_id
        ).first()
        if row:
            row.cache_key = cache_key
            row.payload = payload
        else:
            db.add(DbIntelligenceCacheModel(
                project_id=project_id, cache_key=cache_key, payload=payload,
            ))
        db.commit()
    except Exception as e:  # cache yazımı asla pipeline'ı düşürmemeli
        db.rollback()
        logger.info(f">>> db_intelligence cache yazilamadi: {str(e)[:150]}")
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