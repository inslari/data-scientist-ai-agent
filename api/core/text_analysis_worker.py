"""
Text-Analysis Worker — yazılı geri bildirim analizi iş katmanı
==============================================================
ML worker'dan (`api/core/worker.py`) TAMAMEN AYRI. Text-analiz işlerinin
(`task_type="run_text_analysis"`) çalıştırılması, dispatch'i ve çökmeden
kurtarması burada yaşar. ML kodu (proje pipeline'ı, TTD önerileri) bu modülde
HİÇ yoktur ve bu modül de ML worker'ından hiçbir şey import etmez.

Ortak nokta yalnızca nötr kuyruk altyapısıdır (`api/core/job_runner.py`): iki
worker aynı `background_jobs` tablosunu paylaşır ama birbirlerinin işlerine asla
dokunmaz (her biri kendi lane'ini yokların).

Giriş noktası: `text_analysis_worker_main.py` (ayrı süreç).
"""
import asyncio
import traceback

from api.core.database import SessionLocal
from api.models.job import BackgroundJobModel
from api.core.job_runner import run_job_loop, recover_jobs, mark_job as _mark_job
from src.core.logging_config import set_project, get_logger

logger = get_logger(__name__)

# Bu worker'ın lane'i — yalnızca bu iş tiplerini alır.
TEXT_ANALYSIS_TASK_TYPES = ["run_text_analysis"]


async def _run_text_analysis_job(job: BackgroundJobModel) -> None:
    """
    Yazılı geri bildirim analizi grafiğini çalıştırır (paralel pipeline).

    run_project'ten farklı olarak LangGraph checkpoint'i KULLANMAZ: iş kısa,
    resume gerekmez. Sonuç `TextAnalysisModel.result` (JSON) + `.report`
    (markdown) alanlarına yazılır; durum completed/failed olur.
    """
    from api.models.text_analysis import TextAnalysisModel
    from src.graphs.textanalysis.graph import build_text_analysis_graph

    ta_id = job.text_analysis_id
    if not ta_id:
        _mark_job(job.id, "failed", "run_text_analysis: text_analysis_id missing")
        return

    set_project(ta_id)

    # Girdiyi oku + durumu 'running' yap.
    db = SessionLocal()
    title = None
    raw_texts: list = []
    raw_timestamps: list = []
    raw_ratings: list = []
    raw_labels: list = []
    language = "auto"
    try:
        ta = db.query(TextAnalysisModel).filter(TextAnalysisModel.id == ta_id).first()
        if not ta:
            _mark_job(job.id, "failed", f"TextAnalysis {ta_id} not found")
            return
        title = ta.title
        raw_texts = list(ta.source_texts or [])
        raw_timestamps = list(ta.source_timestamps or [])
        raw_ratings = list(ta.source_ratings or [])
        raw_labels = list(ta.source_labels or [])
        language = ta.language or "auto"
        ta.status = "running"
        ta.error = None
        db.commit()
    finally:
        db.close()

    error_trace: str | None = None
    final_state: dict = {}
    try:
        app = build_text_analysis_graph()
        initial_state = {
            "title": title,
            "raw_texts": raw_texts,
            "raw_timestamps": raw_timestamps,
            "raw_ratings": raw_ratings,
            "raw_labels": raw_labels,
            "language": language,
        }
        final_state = await app.ainvoke(initial_state)
    except asyncio.CancelledError:
        raise
    except Exception:
        error_trace = traceback.format_exc()
        logger.error(f"[TextWorker] Text analysis failed for id={ta_id}:\n{error_trace}")

    # Graf içi (taşınan) hata da başarısızlık sayılır.
    graph_error = final_state.get("error") if isinstance(final_state, dict) else None

    db = SessionLocal()
    try:
        ta = db.query(TextAnalysisModel).filter(TextAnalysisModel.id == ta_id).first()
        if ta:
            if error_trace or graph_error:
                ta.status = "failed"
                ta.error = error_trace or graph_error
                ta.report = final_state.get("report") if isinstance(final_state, dict) else None
            else:
                ta.status = "completed"
                ta.report = final_state.get("report", "")
                ta.result = {
                    "item_results": final_state.get("item_results", []),
                    "sentiment_distribution": final_state.get("sentiment_distribution", {}),
                    "category_distribution": final_state.get("category_distribution", {}),
                    "urgency_distribution": final_state.get("urgency_distribution", {}),
                    "emotion_distribution": final_state.get("emotion_distribution", {}),
                    "aspect_distribution": final_state.get("aspect_distribution", {}),
                    "rating_validation": final_state.get("rating_validation", {}),
                    "label_validation": final_state.get("label_validation", {}),
                    "timestamp_detection": final_state.get("timestamp_detection", {}),
                    "time_trend": final_state.get("time_trend", []),
                    "theme_trends": final_state.get("theme_trends", []),
                    "themes": final_state.get("themes", []),
                    "priority_items": final_state.get("priority_items", []),
                    "detected_language": final_state.get("detected_language"),
                    "duplicate_groups": final_state.get("duplicate_groups", {}),
                }
            db.commit()
    finally:
        db.close()

    _mark_job(job.id, "failed" if (error_trace or graph_error) else "completed", error_trace or graph_error)


async def _dispatch(job: BackgroundJobModel) -> None:
    """Text-analiz lane dispatch'i. Yalnızca run_text_analysis bilinir."""
    if job.task_type == "run_text_analysis":
        await _run_text_analysis_job(job)
    else:
        # Lane filtresi gereği buraya normalde gelinmez (savunma amaçlı).
        _mark_job(job.id, "failed", f"text-analysis lane: beklenmeyen task_type {job.task_type}")


def _reset_owner(db, stuck: list[BackgroundJobModel]) -> None:
    """Kurtarmada takılı text-analiz kayıtlarını 'queued'a çeker (commit etmez)."""
    from api.models.text_analysis import TextAnalysisModel
    ta_ids = [j.text_analysis_id for j in stuck
              if j.task_type == "run_text_analysis" and j.text_analysis_id]
    if ta_ids:
        db.query(TextAnalysisModel).filter(TextAnalysisModel.id.in_(ta_ids)).update(
            {"status": "queued"}, synchronize_session=False
        )


async def job_worker_loop(task_types: list[str] | None = None) -> None:
    """Text-analiz lane poll döngüsü (nötr altyapı + text dispatch)."""
    await run_job_loop(task_types or TEXT_ANALYSIS_TASK_TYPES, _dispatch)


def recover_interrupted_jobs(task_types: list[str] | None = None) -> int:
    """Başlangıç kurtarması — yalnızca text-analiz lane'i (TextAnalysis → queued)."""
    return recover_jobs(task_types or TEXT_ANALYSIS_TASK_TYPES, _reset_owner)
