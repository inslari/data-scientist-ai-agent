"""
Background Worker — ML (proje pipeline) iş katmanı
==================================================
Bu modül YALNIZCA makine öğrenmesi işlerini çalıştırır:
  - run_project              → LangGraph DS pipeline (src/graph.py)
  - generate_ttd_suggestions → Talk-to-Data örnek soru üretimi

Text-analysis işleri (run_text_analysis) BURADA DEĞİLDİR — onlar tamamen ayrı bir
worker'da (`api/core/text_analysis_worker.py`) yaşar. İki worker yalnızca nötr
kuyruk altyapısını (`api/core/job_runner.py`) paylaşır; aynı `background_jobs`
tablosunu kullanır ama birbirlerinin işlerine asla dokunmaz (her biri kendi
lane'ini yokların).

Neden sıralı (sequential)?
  - LangGraph yürütmesi zaten ağırlıkla I/O-bound'dur (LLM çağrıları).
  - Paralel proje çalıştırmak token/bellek tüketimini katlar; sıralı olan
    öngörülebilir ve güvenlidir. Gerekirse ileride semafor eklenebilir.
"""

import asyncio
import time
import traceback
import json

from api.core.database import SessionLocal, SQLALCHEMY_DATABASE_URL
from api.models.job import BackgroundJobModel
from api.models.project import ProjectModel, ProjectStepModel
from api.models.data_group import DataGroupModel
from api.core.job_runner import run_job_loop, recover_jobs, mark_job as _mark_job
from src.graph import build_graph
from src.core.config import settings
from src.core.logging_config import set_project

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Bu worker'ın lane'i — yalnızca ML iş tiplerini alır.
PROJECT_TASK_TYPES = ["run_project", "generate_ttd_suggestions"]


async def _dispatch(job: BackgroundJobModel) -> None:
    """ML lane dispatch'i — yalnızca proje/TTD işleri bilinir."""
    if job.task_type == "run_project":
        await _run_project_job(job)
    elif job.task_type == "generate_ttd_suggestions":
        await _generate_ttd_suggestions_job(job)
    else:
        _mark_job(job.id, "failed", f"ML lane: beklenmeyen task_type {job.task_type}")


def _reset_owner(db, stuck: list[BackgroundJobModel]) -> None:
    """Kurtarmada takılı proje kayıtlarını 'queued'a çeker (commit etmez)."""
    proj_ids = [j.project_id for j in stuck if j.task_type == "run_project" and j.project_id]
    if proj_ids:
        db.query(ProjectModel).filter(ProjectModel.id.in_(proj_ids)).update(
            {"status": "queued"}, synchronize_session=False
        )


async def job_worker_loop(task_types: list[str] | None = None) -> None:
    """ML lane poll döngüsü (nötr altyapı + ML dispatch)."""
    await run_job_loop(task_types or PROJECT_TASK_TYPES, _dispatch)


def recover_interrupted_jobs(task_types: list[str] | None = None) -> int:
    """Başlangıç kurtarması — yalnızca ML lane'i (Project → queued)."""
    return recover_jobs(task_types or PROJECT_TASK_TYPES, _reset_owner)


async def _generate_ttd_suggestions_job(job: BackgroundJobModel) -> None:
    """
    Bir data group için Talk-to-Data örnek sorularını üretir ve
    DataGroupModel.ttd_suggestions kolonuna yazar. Best-effort: hata olsa bile
    job 'completed' işaretlenir (öneriler kritik değil, boş liste kabul edilebilir).
    """
    data_group_id = job.data_group_id
    if not data_group_id:
        _mark_job(job.id, "failed", "generate_ttd_suggestions: data_group_id missing")
        return

    # user_id'yi data group'tan çöz (schema logic yetki kontrolü için kullanır)
    db = SessionLocal()
    try:
        dg = db.query(DataGroupModel).filter(DataGroupModel.id == data_group_id).first()
        if not dg:
            _mark_job(job.id, "failed", f"Data group {data_group_id} not found")
            return
        user_id = dg.user_id
    finally:
        db.close()

    suggestions: list[str] = []
    try:
        from src.services.ttd_suggestion_generator import generate_ttd_suggestions
        db = SessionLocal()
        try:
            suggestions = await generate_ttd_suggestions(data_group_id, db, user_id)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[Worker] TTD öneri üretimi hata (group {data_group_id}): {str(e)[:200]}")

    # Sonucu yaz (boş liste de yazılır → 'üretildi ama soru çıkmadı' ile 'henüz üretilmedi/NULL' ayrışır)
    db = SessionLocal()
    try:
        dg = db.query(DataGroupModel).filter(DataGroupModel.id == data_group_id).first()
        if dg:
            dg.ttd_suggestions = suggestions
            db.commit()
            logger.info(f"[Worker] TTD önerileri kaydedildi (group {data_group_id}): {len(suggestions)} adet")
    except Exception as e:
        db.rollback()
        logger.info(f"[Worker] TTD öneri kaydı başarısız (group {data_group_id}): {str(e)[:200]}")
    finally:
        db.close()

    _mark_job(job.id, "completed")


async def _run_project_job(job: BackgroundJobModel) -> None:
    """
    Executes the LangGraph pipeline for a project, streaming results into
    ProjectStepModel rows and updating ProjectModel.status when finished.
    """

    project_id = job.project_id
    query = job.query
    data_group_id = job.data_group_id
    is_resume = bool(job.is_resume)
    force_restart = bool(job.force_restart)

    # Bu işin tüm logları (node decorator'ları dahil) bu project_id ile etiketlenir.
    set_project(project_id)

    # Resolve domain pack: project override → data group → global DOMAIN_PACK
    resolved_domain_pack = settings.DOMAIN_PACK
    domain_doc = ""

    db = SessionLocal()
    project_intent = None
    project_target_col = None
    project_modality = None
    project_time_col = None
    project_entity_col = None
    try:
        project: ProjectModel | None = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not project:
            logger.warning(f"[Worker] Project {project_id} not found; skipping job {job.id}.")
            _mark_job(job.id, "failed", "Project not found")
            return

        project_intent = getattr(project, "intent", None)
        # DetachedInstanceError'dan kaçınmak için session AÇIKKEN lokal değişkene oku
        project_target_col = getattr(project, "target_col", None)
        project_modality = getattr(project, "modality", None)
        project_time_col = getattr(project, "time_col", None)
        project_entity_col = getattr(project, "entity_col", None)

        dg = db.query(DataGroupModel).filter(DataGroupModel.id == data_group_id).first() if data_group_id else None
        if project.domain_pack:
            resolved_domain_pack = project.domain_pack
        elif dg and dg.domain_pack:
            resolved_domain_pack = dg.domain_pack
        domain_doc = (dg.domain_doc if dg else None) or ""
        logger.info(f"[Worker] Resolved domain_pack='{resolved_domain_pack}', domain_doc={'yes' if domain_doc else 'no'} for project {project_id}")

        project.status = "running"
        project.error = None
        db.commit()
    finally:
        db.close()

    error_trace: str | None = None
    is_postgres = SQLALCHEMY_DATABASE_URL.startswith("postgresql")

    async def run_with_saver(saver, is_postgres_db: bool):
        nonlocal is_resume, error_trace
        try:
            app = build_graph(checkpointer=saver)
            config = {
                "configurable": {
                    "data_group_id": data_group_id,
                    "thread_id": str(project_id),
                },
                # Ana hat (~12 node) + critic self-correction döngüsü
                # (max_retries × feature_plan→…→critic ≈ 5 node/tur) LangGraph'ın
                # varsayılan 25 super-step limitini aşabilir. Retry bütçesiyle
                # uyumlu, güvenli bir tavan veriyoruz.
                "recursion_limit": 50,
            }

            # domain_pack resolved above: project override → data group → global default
            domain_pack = resolved_domain_pack
            # domain_doc resolved above from the data group (dataset description)
            initial_state: dict | None = {"user_query": query, "domain_pack": domain_pack, "domain_doc": domain_doc}
            if project_intent:
                initial_state["intent"] = project_intent
            if project_target_col:
                # Manuel hedef seçimi: target_discovery_node bypass edilecek.
                # feature_plan_node'un confidence guard'ı bozulmasın diye
                # target_definition'ı confidence=1.0 ile dolu veriyoruz.
                initial_state["target_col"] = project_target_col
                initial_state["target_definition"] = {
                    "column_name": project_target_col,
                    "table_name": "",
                    "is_binary": False,
                    "description": "",
                    "confidence": 1.0,
                    "confirmed": True,
                }

            # Veri modality ekseni (domain_pack'ten bağımsız). Boş/"tabular" =
            # mevcut davranış; "timeseries" feature_builder'da kronolojik split tetikler.
            if project_modality:
                initial_state["modality"] = project_modality
            if project_time_col:
                initial_state["time_col"] = project_time_col
            # entity_col (FAZ B2-b): boş → GLOBAL (tek seri); dolu → feature_builder
            # lag/rolling/frekans/split'i entity grubu içinde hesaplar.
            if project_entity_col:
                initial_state["entity_col"] = project_entity_col
            print(f"[Worker] modality='{project_modality or 'tabular'}', time_col='{project_time_col or '-'}', entity_col='{project_entity_col or '-'}' for project {project_id}")

            if is_resume:
                try:
                    state_snapshot = await app.aget_state(config)
                    if state_snapshot and getattr(state_snapshot, "values", None):
                        initial_state = None  # Resume from checkpoint
                    else:
                        is_resume = False
                except Exception as exc:
                    logger.error(f"[Worker] Could not fetch state snapshot: {exc}")
                    is_resume = False

            if force_restart:
                if is_postgres_db:
                    async with saver.conn.connection() as conn:
                        await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (str(project_id),))
                        await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (str(project_id),))
                else:
                    await saver.conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ?", (str(project_id),)
                    )
                    await saver.conn.execute(
                        "DELETE FROM writes WHERE thread_id = ?", (str(project_id),)
                    )
                    await saver.conn.commit()

            # Clear existing steps for a fresh run
            if not is_resume:
                db_s = SessionLocal()
                try:
                    db_s.query(ProjectStepModel).filter(ProjectStepModel.project_id == project_id).delete()
                    db_s.commit()
                finally:
                    db_s.close()

            final_state: dict = {}

            async for chunk in app.astream(initial_state, config=config):
                node_name = list(chunk.keys())[0]
                logger.info(f"\n{'='*40}\n[Worker][NODE END] {node_name}\n{'='*40}")

                chunk_data: dict = chunk[node_name]
                final_state.update(chunk_data)

                safe_data: dict = {}
                for k, v in chunk_data.items():
                    if k == "model_artifact" or "DataFrame" in str(type(v)):
                        continue
                    if isinstance(v, list) and v and "DataFrame" in str(type(v[0])):
                        continue
                    if k in ["sql_results", "sql_query_results", "X_train", "X_test", "y_train", "y_test"]:
                        if isinstance(v, list):
                            safe_data[k] = v[:5]
                            continue
                        else:
                            safe_data[k] = "[Büyük veri kısıtlandı]"
                            continue
                    safe_data[k] = v

                # --- NaN/Inf sanitization ---
                # psycopg2 uses json.dumps internally and NaN is invalid in standard JSON.
                # We do a full round-trip: serialize with allow_nan=True to catch everything,
                # then replace NaN/Inf tokens, then deserialize back to a clean dict.
                import math
                def _sanitize(obj):
                    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                        return None
                    if isinstance(obj, dict):
                        return {kk: _sanitize(vv) for kk, vv in obj.items()}
                    if isinstance(obj, list):
                        return [_sanitize(i) for i in obj]
                    return obj

                try:
                    # Full-depth sanitize then verify no NaN slipped through
                    safe_data = _sanitize(safe_data)
                    json.dumps(safe_data, allow_nan=False)  # raises if still invalid
                except Exception:
                    # Last resort: convert entire payload to string repr
                    safe_data = {"_raw": str(safe_data)}

                db_s = SessionLocal()
                try:
                    step = ProjectStepModel(
                        project_id=project_id,
                        node_name=node_name,
                        step_data=safe_data,
                        created_at=int(time.time()),
                    )
                    db_s.add(step)
                    db_s.commit()
                finally:
                    db_s.close()

            # Mark project completed
            db_s = SessionLocal()
            try:
                project_obj = db_s.query(ProjectModel).filter(ProjectModel.id == project_id).first()
                if project_obj:
                    project_obj.status = "completed"
                    project_obj.report = final_state.get("report", "")
                    db_s.commit()
            finally:
                db_s.close()

        except asyncio.CancelledError:
            raise
        except Exception:
            error_trace = traceback.format_exc()
            logger.error(f"[Worker] Graph execution failed for project {project_id}:\n{error_trace}")

            db_s = SessionLocal()
            try:
                project_obj = db_s.query(ProjectModel).filter(ProjectModel.id == project_id).first()
                if project_obj:
                    project_obj.status = "failed"
                    project_obj.error = error_trace
                    db_s.commit()
            finally:
                db_s.close()

    # --- Setup saver and run ---
    try:
        if is_postgres:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
            from psycopg.rows import dict_row
            from psycopg import AsyncConnection
            from typing import Any
            
            pg_conn_str = SQLALCHEMY_DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
            
            pool_kwargs = {
                "autocommit": True,
                "row_factory": dict_row,
            }
            
            async with AsyncConnectionPool(
                pg_conn_str, 
                kwargs=pool_kwargs
            ) as pool:
                saver = AsyncPostgresSaver(pool)  # type: ignore
                await saver.setup()
                await run_with_saver(saver, is_postgres_db=True)
        else:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as saver:
                await saver.setup()
                await run_with_saver(saver, is_postgres_db=False)
    except asyncio.CancelledError:
        raise
    except Exception:
        if not error_trace:
            error_trace = traceback.format_exc()
            logger.error(f"[Worker] Saver setup/execution failed for project {project_id}:\n{error_trace}")
            
            db_s = SessionLocal()
            try:
                project_obj = db_s.query(ProjectModel).filter(ProjectModel.id == project_id).first()
                if project_obj:
                    project_obj.status = "failed"
                    project_obj.error = error_trace
                    db_s.commit()
            finally:
                db_s.close()

    _mark_job(job.id, "failed" if error_trace else "completed", error_trace)
