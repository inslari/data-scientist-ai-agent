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
import re
import time as _time

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db
from api.core.deps import get_current_user
from api.models.user import UserModel
from api.models.data_group import DataGroupModel, VisualizationModel
from api.schemas.viz import (
    VizRequest, VizResponse, VizItem, VizSuggestion,
    VizThemeUpdate, VizPatch, VizRefine, VizReportRequest,
)
from api.core.jobs import enqueue_viz_suggestions

from src.graphs.dataviz_graph import build_dataviz_graph
from src.graphs.dataviz.state import _SPEC_FIELDS
from src.services.viz_themes import theme_catalog
from src.services.viz_data import execute_viz_sql, to_storable, from_storable
from src.services.matplotlib_renderer import render_matplotlib
from src.services.viz_report import build_pdf_report
from src.core.config import settings as _ds_settings

# Talk-to-Data ile aynı SQLite şema-cache yardımcıları (yeniden kullanılır)
from api.routes.data_group import _read_db_cache, _write_db_cache

from src.core.logging_config import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/data-groups", tags=["Data Visualization"])

# Tema kataloğu data group'a bağlı değil; `/data-groups/{group_id}` kalıbıyla çakışmasın
# diye kendi router'ında duruyor (main.py ikisini de include eder).
themes_router = APIRouter(tags=["Data Visualization"])


@themes_router.get("/viz-themes", response_model=List[dict])
def list_viz_themes(current_user: UserModel = Depends(get_current_user)):
    """
    Renk teması kataloğu. Frontend grafikleri istemcide bu paletlerle çizer; sunucu PNG'si
    de aynı kataloğu kullanır → tek kaynak: src/services/viz_themes.py.
    """
    return theme_catalog()


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


@router.patch("/{group_id}/viz-theme")
def set_group_viz_theme(
    group_id: int,
    payload: VizThemeUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Veri grubunun varsayılan temasını değiştirir — bundan sonraki grafikler bunu alır."""
    db_group = _get_group_or_404(group_id, db, current_user.id)
    db_group.viz_theme = payload.theme
    db.commit()
    return {"status": "success", "theme": payload.theme}


async def _run_dataviz_graph(
    db: Session,
    db_group: DataGroupModel,
    user_id: int,
    *,
    user_query: str,
    suggestion: dict | None = None,
    theme=None,
    locale: str | None = None,
    refine_context: dict | None = None,
) -> dict:
    """
    Data Visualization pipeline'ını çalıştırır ve şema cache'ini günceller.
    Hem yeni grafik üretimi hem doğal dille iyileştirme (refine) buradan geçer.
    """
    graph = build_dataviz_graph()
    cached = _read_db_cache(db_group)

    initial_state: dict = {
        "user_query":          user_query,
        "selected_suggestion": suggestion,
        "data_group_id":       db_group.id,
        "long_term_memory":    db_group.long_term_memory,
        "domain_pack":         db_group.domain_pack or _ds_settings.DOMAIN_PACK,
        "domain_doc":          db_group.domain_doc or "",
        "viz_theme":           theme,
        "locale":              (locale or "tr").lower(),
        "refine_context":      refine_context,
    }
    if cached:
        db_intelligence, raw_schema = cached
        initial_state["db_intelligence"] = db_intelligence
        initial_state["raw_schema"] = raw_schema
        logger.info(f"[viz] SQLite cache hit: group_id={db_group.id}")
    else:
        logger.info(f"[viz] SQLite cache miss: group_id={db_group.id} — context node çalışacak")

    config = {"configurable": {"data_group_id": db_group.id, "user_id": user_id}}

    # TTD ile aynı: node'lar async ama içeride bloklayan .invoke() çağırıyor.
    # Tüm graph'ı ayrı thread + kendi event loop'unda çalıştır → API loop bloklanmaz.
    result = await asyncio.to_thread(
        lambda: asyncio.run(graph.ainvoke(initial_state, config=config))
    )

    # Context node çalıştıysa şemayı cache'le
    if not cached and result.get("db_intelligence") and result.get("raw_schema"):
        _write_db_cache(db, db_group, result["db_intelligence"], result["raw_schema"])

    return result


def _get_viz_or_404(group_id: int, viz_id: int, db: Session) -> VisualizationModel:
    viz = db.query(VisualizationModel).filter(
        VisualizationModel.id == viz_id, VisualizationModel.data_group_id == group_id
    ).first()
    if not viz:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return viz


def _rerender_from_stored(viz: VisualizationModel, locale: str | None) -> None:
    """
    Saklanan SQL sonucundan PNG + payload'ı yeniden üretir (LLM ve SQL YOK).
    Veri saklanmamışsa (eski kayıt) sessizce dokunmaz — istemci mevcut PNG'yi gösterir.
    """
    columns, rows = from_storable(viz.data)
    if not columns or not rows:
        return
    result = render_matplotlib(viz.spec or {}, rows, columns,
                               theme=viz.theme, locale=(locale or "tr").lower())
    if result.get("error"):
        raise HTTPException(status_code=400, detail=f"Render failed: {result['error']}")
    viz.image_b64 = result.get("image_b64")
    viz.chart = result.get("chart")


@router.patch("/{group_id}/visualizations/{viz_id}", response_model=VizItem)
def patch_visualization(
    group_id: int,
    viz_id: int,
    payload: VizPatch,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir grafiğin görünümünü değiştirir — LLM ve SQL ÇALIŞMAZ.

    - `theme`: renk teması; istemci zaten yeniden çizer.
    - `spec`: chart_type / sort / top_n / data_labels / başlık gibi alanlar. Saklanan
      SQL sonucundan PNG ve payload yeniden üretilir, böylece indirilen görsel de
      ekrandakiyle aynı kalır.
    """
    _get_group_or_404(group_id, db, current_user.id)
    viz = _get_viz_or_404(group_id, viz_id, db)

    theme_changed = "theme" in payload.model_fields_set
    if theme_changed:
        viz.theme = payload.theme

    if payload.spec:
        # Yalnızca bilinen spec alanları; SQL istemciden DEĞİŞTİRİLEMEZ (enjeksiyon yüzeyi).
        allowed = _SPEC_FIELDS - {"sql"}
        merged = dict(viz.spec or {})
        merged.update({k: v for k, v in payload.spec.items() if k in allowed})
        viz.spec = merged

    if payload.spec or theme_changed:
        _rerender_from_stored(viz, payload.locale)

    db.commit()
    db.refresh(viz)
    return viz


@router.post("/{group_id}/visualizations/{viz_id}/refresh", response_model=VizItem)
async def refresh_visualization(
    group_id: int,
    viz_id: int,
    payload: VizPatch | None = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Grafiğin saklanan SQL'ini yeniden çalıştırıp güncel veriyle çizer — LLM YOK.
    Veri değiştikçe eskiyen raporları tazelemek için.
    """
    _get_group_or_404(group_id, db, current_user.id)
    viz = _get_viz_or_404(group_id, viz_id, db)
    if not viz.sql:
        raise HTTPException(status_code=400, detail="This visualization has no stored SQL to re-run")

    try:
        columns, rows = await execute_viz_sql(group_id, viz.sql, current_user.id)
    except Exception as e:
        logger.error(f"[viz-refresh] SQL başarısız (viz={viz_id}): {e}")
        raise HTTPException(status_code=400, detail=f"Query failed: {e}")

    if not rows:
        raise HTTPException(status_code=400, detail="Query returned no rows")

    locale = (payload.locale if payload else None) or "tr"
    result = render_matplotlib(viz.spec or {}, rows, columns, theme=viz.theme, locale=locale.lower())
    if result.get("error"):
        raise HTTPException(status_code=400, detail=f"Render failed: {result['error']}")

    viz.image_b64 = result.get("image_b64")
    viz.chart = result.get("chart")
    viz.data = to_storable(columns, rows)
    # created_at'e DOKUNULMAZ: galeri ona göre sıralanıyor, tazelenen grafik yerinde kalmalı.
    db.commit()
    db.refresh(viz)
    return viz


@router.post("/{group_id}/visualizations/{viz_id}/refine", response_model=VizItem)
async def refine_visualization(
    group_id: int,
    viz_id: int,
    payload: VizRefine,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Mevcut bir grafiği doğal dille değiştirir ("bunu yatay yap", "sadece ilk 5",
    "geçen yılla karşılaştır"). Kayıt YERİNDE güncellenir — galeriye ikinci bir kart
    eklenmez, böylece kullanıcı aynı grafiği iterasyonla düzeltebilir.

    Not: yalnızca görünüm değişecekse (tip/sıralama/ilk-N) PATCH daha ucuzdur; bu uç
    nokta SQL'in de değişmesi gerekebilecek istekler içindir.
    """
    db_group = _get_group_or_404(group_id, db, current_user.id)
    viz = _get_viz_or_404(group_id, viz_id, db)

    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="'message' is required")

    try:
        result = await _run_dataviz_graph(
            db, db_group, current_user.id,
            user_query=message,
            theme=viz.theme,
            locale=payload.locale,
            refine_context={
                "spec": viz.spec or {},
                "sql":  viz.sql or "",
                "original_request": viz.user_query or "",
            },
        )
    except Exception as e:
        logger.error(f"[viz-refine] pipeline hatası (viz={viz_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Refine failed: {e}")

    if result.get("clarification_needed"):
        raise HTTPException(
            status_code=400,
            detail=result.get("clarification_question") or "Could not understand the change.",
        )
    image_b64 = result.get("viz_image_b64")
    if not image_b64:
        detail = result.get("viz_render_error") or result.get("error") or "Could not build the chart."
        raise HTTPException(status_code=400, detail=detail)

    viz.spec      = result.get("viz_spec")
    viz.image_b64 = image_b64
    viz.chart     = result.get("viz_chart")
    viz.data      = result.get("viz_data")
    viz.sql       = result.get("sql_query_raw")
    viz.summary   = (result.get("viz_summary") or "").strip() or viz.summary
    # created_at'e dokunulmaz: kart galerideki yerinde kalır.
    db.commit()
    db.refresh(viz)
    return viz


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


@router.post("/{group_id}/report")
def export_report(
    group_id: int,
    payload: VizReportRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Seçilen grafiklerden tek parça PDF rapor üretir ve indirilebilir olarak döner.
    Grafikler `viz_ids` içindeki SIRAYLA yerleştirilir (kullanıcının düzeni korunur).
    """
    db_group = _get_group_or_404(group_id, db, current_user.id)
    if not payload.viz_ids:
        raise HTTPException(status_code=400, detail="'viz_ids' is empty")

    found = {
        v.id: v for v in db.query(VisualizationModel).filter(
            VisualizationModel.data_group_id == group_id,
            VisualizationModel.id.in_(payload.viz_ids),
        ).all()
    }
    missing = [i for i in payload.viz_ids if i not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"Visualizations not found: {missing}")

    locale = (payload.locale or "tr").lower()
    items = [{
        "title":     (found[i].spec or {}).get("title") or found[i].user_query or "",
        "image_b64": found[i].image_b64,
        "summary":   found[i].summary,
        "sql":       found[i].sql,
        "spec":      found[i].spec,
        "data":      found[i].data,
        "locale":    locale,
    } for i in payload.viz_ids]

    try:
        pdf_bytes = build_pdf_report(
            items,
            title=payload.title or db_group.name,
            subtitle=payload.subtitle,
            theme=db_group.viz_theme,
            include_summaries=payload.include_summaries,
            include_sql=payload.include_sql,
            locale=locale,
        )
    except Exception as e:
        logger.error(f"[viz-report] PDF üretimi başarısız (group={group_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    safe_name = re.sub(r"[^\w\-. ]", "_", (payload.title or db_group.name or "rapor")).strip()[:60]
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name or "rapor"}.pdf"'},
    )


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
        # Tema önceliği: isteğe özel > grubun varsayılanı > katalog varsayılanı
        theme = req.theme if req.theme is not None else db_group.viz_theme
        result = await _run_dataviz_graph(
            db, db_group, current_user.id,
            user_query=user_query,
            suggestion=suggestion_dict,
            theme=theme,
            locale=req.locale,
        )

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
                theme=theme,
                data=result.get("viz_data"),
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
            theme=theme,
            data=result.get("viz_data"),
            warning=warning,
            clarification_needed=clarification,
            created_at=created_at if saved_id else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[create_visualization] Hata: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization pipeline hatası: {str(e)}")
