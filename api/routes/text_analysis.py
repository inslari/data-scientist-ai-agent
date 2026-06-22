import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db
from api.models.text_analysis import TextAnalysisModel
from api.models.job import BackgroundJobModel
from api.schemas.text_analysis import (
    TextAnalysisCreate,
    TextAnalysisUpdate,
    TextAnalysisResponse,
    TextAnalysisListItem,
    TextItemsPage,
)
from api.core.deps import get_current_user
from api.models.user import UserModel

router = APIRouter(prefix="/text-analyses", tags=["Text Analysis"])


def _split_raw_text(raw_text: str) -> list[str]:
    """
    Yapıştırılan blok metni ayrı geri bildirimlere böler. Önce boş satır
    (paragraf) ayracı denenir; yoksa satır satır bölünür.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []
    if "\n\n" in raw_text:
        parts = raw_text.split("\n\n")
    else:
        parts = raw_text.splitlines()
    return [p.strip() for p in parts if p.strip()]


def _resolve_texts(texts, raw_text) -> list[str]:
    if texts:
        return texts
    if raw_text:
        return _split_raw_text(raw_text)
    return []


@router.post("", response_model=TextAnalysisResponse)
def create_text_analysis(
    payload: TextAnalysisCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Yeni bir metin analizi oluşturur (metinleri saklar, henüz çalıştırmaz)."""
    texts = _resolve_texts(payload.texts, payload.raw_text)
    if not texts:
        raise HTTPException(status_code=400, detail="En az bir metin girilmelidir")

    ta = TextAnalysisModel(
        title=payload.title,
        description=payload.description,
        source_texts=texts,
        language=payload.language or "auto",
        user_id=current_user.id,
        status="idle",
        created_at=int(time.time()),
        updated_at=int(time.time()),
    )
    db.add(ta)
    db.commit()
    db.refresh(ta)
    return ta


@router.get("", response_model=List[TextAnalysisListItem])
def list_text_analyses(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    rows = (
        db.query(TextAnalysisModel)
        .filter(TextAnalysisModel.user_id == current_user.id)
        .order_by(TextAnalysisModel.created_at.desc())
        .all()
    )
    return [
        TextAnalysisListItem(
            id=r.id,
            title=r.title,
            description=r.description,
            status=r.status,
            text_count=len(r.source_texts or []),
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


def _get_owned(ta_id: int, db: Session, user: UserModel) -> TextAnalysisModel:
    ta = db.query(TextAnalysisModel).filter(
        TextAnalysisModel.id == ta_id, TextAnalysisModel.user_id == user.id
    ).first()
    if not ta:
        raise HTTPException(status_code=404, detail="Text analysis not found")
    return ta


@router.get("/{ta_id}", response_model=TextAnalysisResponse)
def get_text_analysis(
    ta_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Detay: aggregate'ler + rapor döner. Şişmesin diye ham metinler (source_texts)
    ve büyük item_results burada DÖNMEZ; item_results için /{ta_id}/items (sayfalı).
    """
    ta = _get_owned(ta_id, db, current_user)

    result = ta.result if isinstance(ta.result, dict) else {}
    item_results = result.get("item_results", []) if isinstance(result, dict) else []
    # result'ın item_results HARİÇ kopyası (aggregate'ler: dağılımlar, temalar,
    # priority_items, aspect_distribution, time_trend, timestamp_detection kalır).
    trimmed = {k: v for k, v in result.items() if k != "item_results"} if isinstance(result, dict) else result

    return {
        "id": ta.id,
        "title": ta.title,
        "description": ta.description,
        "language": ta.language,
        "user_id": ta.user_id,
        "source_texts": None,
        "status": ta.status,
        "result": trimmed or None,
        "report": ta.report,
        "error": ta.error,
        "item_count": len(item_results),
        "created_at": ta.created_at,
        "updated_at": ta.updated_at,
    }


@router.get("/{ta_id}/items", response_model=TextItemsPage)
def get_text_analysis_items(
    ta_id: int,
    page: int = 1,
    page_size: int = 25,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """item_results'ı sayfalı döndürür (tek seferde tüm 100+ kaydı çekmemek için)."""
    ta = _get_owned(ta_id, db, current_user)
    result = ta.result if isinstance(ta.result, dict) else {}
    items = result.get("item_results", []) if isinstance(result, dict) else []

    page = max(1, page)
    page_size = max(1, min(page_size, 500))
    start = (page - 1) * page_size
    return TextItemsPage(
        items=items[start:start + page_size],
        total=len(items),
        page=page,
        page_size=page_size,
    )


@router.put("/{ta_id}", response_model=TextAnalysisResponse)
def update_text_analysis(
    ta_id: int,
    payload: TextAnalysisUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    ta = _get_owned(ta_id, db, current_user)
    if payload.title is not None:
        ta.title = payload.title
    if payload.description is not None:
        ta.description = payload.description
    if payload.language is not None:
        ta.language = payload.language
    if payload.texts is not None or payload.raw_text is not None:
        texts = _resolve_texts(payload.texts, payload.raw_text)
        if not texts:
            raise HTTPException(status_code=400, detail="En az bir metin girilmelidir")
        ta.source_texts = texts
    ta.updated_at = int(time.time())
    db.commit()
    db.refresh(ta)
    return ta


@router.post("/{ta_id}/run", response_model=TextAnalysisResponse)
def run_text_analysis(
    ta_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Analizi arka plan kuyruğuna ekler (Outbox pattern); hemen döner."""
    ta = _get_owned(ta_id, db, current_user)
    if not ta.source_texts:
        raise HTTPException(status_code=400, detail="Analiz edilecek metin yok")

    ta.status = "queued"
    ta.error = None

    job = BackgroundJobModel(
        task_type="run_text_analysis",
        text_analysis_id=ta.id,
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(ta)
    return ta


@router.delete("/{ta_id}")
def delete_text_analysis(
    ta_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    ta = _get_owned(ta_id, db, current_user)
    db.delete(ta)
    db.commit()
    return {"status": "success", "message": f"Text analysis {ta_id} deleted"}
