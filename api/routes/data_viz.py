"""
Data Visualization route'ları.

Talk-to-Data chat route'unun (data_group.py::chat_with_data_group) görselleştirme
muadili: doğal dil / seçilen öneri kartını alır, bağımsız Data Visualization
LangGraph pipeline'ından (build_dataviz_graph) geçirir:

    viz_context_node → viz_plan_node → viz_sql_node → viz_validator_node
                     → viz_render_node → viz_interpreter_node

Şema cache davranışı Talk-to-Data ile aynıdır (SQLite tabanlı, _read/_write_db_cache).
Üretilen grafik VisualizationModel'e kaydedilir (kalıcı galeri).
"""
import asyncio
import time as _time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db
from api.core.deps import get_current_user
from api.models.user import UserModel
from api.models.data_group import DataGroupModel, VisualizationModel
from api.schemas.viz import VizRequest, VizResponse, VizItem, VizSuggestion
from api.core.jobs import enqueue_viz_suggestions

from src.graphs.dataviz_graph import build_dataviz_graph
from src.core.config import settings as _ds_settings

# Talk-to-Data ile aynı SQLite şema-cache yardımcıları (yeniden kullanılır)
from api.routes.data_group import _read_db_cache, _write_db_cache

from src.core.logging_config import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/data-groups", tags=["Data Visualization"])


def _get_group_or_404(group_id: int, db: Session, user_id: int) -> DataGroupModel:
    db_group = db.query(DataGroupModel).filter(
        DataGroupModel.id == group_id, DataGroupModel.user_id == user_id
    ).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
    return db_group


@router.get("/{group_id}/viz-suggestions", response_model=List[VizSuggestion])
def get_viz_suggestions(group_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """
    Data group'a özel görselleştirme öneri kartlarını döner. Henüz üretilmediyse
    (NULL) arka plan job'ı tetikler ve boş liste döner.
    """
    db_group = _get_group_or_404(group_id, db, current_user.id)
    if db_group.viz_suggestions is None:
        enqueue_viz_suggestions(db, group_id)
        return []
    return db_group.viz_suggestions or []


@router.get("/{group_id}/visualizations", response_model=List[VizItem])
def list_visualizations(group_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """Data group için kaydedilmiş görselleştirmeleri (galeri) kronolojik döner."""
    _get_group_or_404(group_id, db, current_user.id)
    return db.query(VisualizationModel).filter(
        VisualizationModel.data_group_id == group_id
    ).order_by(VisualizationModel.created_at.asc()).all()


@router.delete("/{group_id}/visualizations/{viz_id}")
def delete_visualization(group_id: int, viz_id: int, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    """Bir galeri kaydını siler."""
    _get_group_or_404(group_id, db, current_user.id)
    viz = db.query(VisualizationModel).filter(
        VisualizationModel.id == viz_id, VisualizationModel.data_group_id == group_id
    ).first()
    if not viz:
        raise HTTPException(status_code=404, detail="Visualization not found")
    db.delete(viz)
    db.commit()
    return {"status": "success", "deleted": viz_id}


@router.post("/{group_id}/visualize", response_model=VizResponse)
async def create_visualization(
    group_id: int,
    req: VizRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Doğal dil isteği veya seçilen öneri kartından bir görselleştirme üretir.
    Talk-to-Data chat route'uyla aynı yürütme deseni (ayrı thread + kendi event loop).
    """
    db_group = _get_group_or_404(group_id, db, current_user.id)

    if not req.message and not req.suggestion:
        raise HTTPException(status_code=400, detail="Either 'message' or 'suggestion' is required")

    suggestion_dict = req.suggestion.model_dump() if req.suggestion else None
    # user_query: serbest istek ya da kart başlığı (kayıt/etiket için)
    user_query = req.message or (req.suggestion.title if req.suggestion else "") or ""

    try:
        graph = build_dataviz_graph()

        cached = _read_db_cache(db_group)

        initial_state: dict = {
            "user_query":          user_query,
            "selected_suggestion": suggestion_dict,
            "data_group_id":       group_id,
            "long_term_memory":    db_group.long_term_memory,
            "domain_pack":         db_group.domain_pack or _ds_settings.DOMAIN_PACK,
            "domain_doc":          db_group.domain_doc or "",
        }
        if cached:
            db_intelligence, raw_schema = cached
            initial_state["db_intelligence"] = db_intelligence
            initial_state["raw_schema"] = raw_schema
            logger.info(f"[viz] SQLite cache hit: group_id={group_id}")
        else:
            logger.info(f"[viz] SQLite cache miss: group_id={group_id} — context node çalışacak")

        config = {"configurable": {"data_group_id": group_id, "user_id": current_user.id}}

        # TTD ile aynı: node'lar async ama içeride bloklayan .invoke() çağırıyor.
        # Tüm graph'ı ayrı thread + kendi event loop'unda çalıştır → API loop bloklanmaz.
        result = await asyncio.to_thread(
            lambda: asyncio.run(graph.ainvoke(initial_state, config=config))
        )

        # Context node çalıştıysa şemayı cache'le
        if not cached and result.get("db_intelligence") and result.get("raw_schema"):
            _write_db_cache(db, db_group, result["db_intelligence"], result["raw_schema"])

        summary = (result.get("viz_summary") or "").strip() or "Visualization generated."
        clarification = bool(result.get("clarification_needed"))
        image_b64 = result.get("viz_image_b64")
        chart = result.get("viz_chart")
        spec = result.get("viz_spec")
        sql = result.get("sql_query_raw")
        columns = result.get("sql_query_columns") or []
        warning = result.get("viz_render_error") or result.get("sql_query_warning") or None

        # Sadece gerçek bir grafik üretildiyse galeriye kaydet
        # (clarification veya render hatası kaydedilmez).
        saved_id = None
        created_at = int(_time.time())
        if image_b64 and not clarification:
            viz = VisualizationModel(
                data_group_id=group_id,
                created_at=created_at,
                user_query=user_query,
                spec=spec,
                image_b64=image_b64,
                chart=chart,
                sql=sql,
                summary=summary,
            )
            db.add(viz)
            db.commit()
            db.refresh(viz)
            saved_id = viz.id

        return VizResponse(
            id=saved_id,
            summary=summary,
            image_b64=image_b64,
            chart=chart,
            sql=sql,
            spec=spec,
            columns=columns,
            warning=warning,
            clarification_needed=clarification,
            created_at=created_at if saved_id else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[create_visualization] Hata: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization pipeline hatası: {str(e)}")
