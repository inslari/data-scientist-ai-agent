import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Tuple

from api.core.database import get_db
from api.models.text_analysis import TextAnalysisModel
from api.models.csv import CSVFileModel
from api.models.job import BackgroundJobModel
from api.schemas.text_analysis import (
    TextAnalysisCreate,
    TextAnalysisUpdate,
    TextAnalysisResponse,
    TextAnalysisListItem,
    TextItemsPage,
    TextAppendRequest,
)
from api.core.deps import get_current_user
from api.core.storage import storage
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


def _extract_from_csv(
    db: Session, user: UserModel, csv_file_id: int,
    text_column: str, date_column: Optional[str], rating_column: Optional[str],
    label_column: Optional[str] = None,
) -> Tuple[list[str], Optional[list], Optional[list], Optional[list]]:
    """
    Bir CSV dosyasının seçili kolonlarından metin (+opsiyonel tarih/puan/etiket)
    çeker. Tüm kolonlar metinle HİZALI döner (aynı satır). Boş metin satırları
    elenir; hizayı korumak için tarih/puan/etiket de aynı satırda elenir.

    label_column: ground-truth duygu etiketi (Durum/sentiment...). Farklı isim ve
    sözcükler (Olumlu/Olumsuz, positive/negative) downstream'de normalize edilir.
    """
    if not text_column:
        raise HTTPException(status_code=400, detail="text_column zorunludur")

    csv = db.query(CSVFileModel).filter(
        CSVFileModel.id == csv_file_id, CSVFileModel.user_id == user.id
    ).first()
    if not csv:
        raise HTTPException(status_code=404, detail="CSV bulunamadı")

    from api.core.csv_utils import read_csv_path
    usecols = [c for c in [text_column, date_column, rating_column, label_column] if c]
    try:
        with storage.download_to_tempfile(csv.file_path) as path:
            df = read_csv_path(path, usecols=lambda c: c in usecols)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Kolon okunamadı: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV okunamadı: {e}")

    if text_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"'{text_column}' kolonu CSV'de yok")

    texts: list[str] = []
    timestamps: list = []
    ratings: list = []
    labels: list = []
    for _, row in df.iterrows():
        t = row.get(text_column)
        if not isinstance(t, str) or not t.strip():
            continue
        texts.append(t.strip())
        if date_column:
            d = row.get(date_column)
            timestamps.append(str(d) if (d is not None and str(d).strip() and str(d) != "nan") else None)
        if rating_column:
            r = row.get(rating_column)
            try:
                ratings.append(float(r) if r is not None and str(r) != "nan" else None)
            except (TypeError, ValueError):
                ratings.append(None)
        if label_column:
            lb = row.get(label_column)
            labels.append(str(lb).strip() if (lb is not None and str(lb).strip() and str(lb) != "nan") else None)

    return (
        texts,
        (timestamps if date_column else None),
        (ratings if rating_column else None),
        (labels if label_column else None),
    )


def _resolve_inputs(
    db: Session, user: UserModel, payload
) -> Tuple[list[str], Optional[list], Optional[list], Optional[list]]:
    """texts > raw_text > CSV kolonu öncelik sırasıyla (metin, tarih, puan, etiket) çözer."""
    if getattr(payload, "texts", None):
        return payload.texts, None, None, None
    if getattr(payload, "raw_text", None):
        return _split_raw_text(payload.raw_text), None, None, None
    if getattr(payload, "csv_file_id", None):
        return _extract_from_csv(
            db, user, payload.csv_file_id, payload.text_column,
            payload.date_column, payload.rating_column, getattr(payload, "label_column", None),
        )
    return [], None, None, None


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
    texts, timestamps, ratings, labels = _resolve_inputs(db, current_user, payload)
    if not texts:
        raise HTTPException(status_code=400, detail="En az bir metin girilmelidir")

    ta = TextAnalysisModel(
        title=payload.title,
        description=payload.description,
        source_texts=texts,
        source_timestamps=timestamps,
        source_ratings=ratings,
        source_labels=labels,
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


@router.post("/{ta_id}/append", response_model=TextAnalysisResponse)
def append_text_analysis(
    ta_id: int,
    payload: TextAppendRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Mevcut analize yeni geri bildirim ekler (incremental). Hizalı tarih/puan
    dizileri korunur. payload.run=True ise yeniden kuyruğa alınır — re-run
    sınıflandırma cache'inden yararlanır (eski metinler yeniden LLM'e gitmez).
    """
    ta = _get_owned(ta_id, db, current_user)
    new_texts, new_ts, new_ratings, new_labels = _resolve_inputs(db, current_user, payload)
    if not new_texts:
        raise HTTPException(status_code=400, detail="Eklenecek metin yok")

    old_texts = list(ta.source_texts or [])
    old_ts = list(ta.source_timestamps or []) if ta.source_timestamps else [None] * len(old_texts)
    old_ratings = list(ta.source_ratings or []) if ta.source_ratings else [None] * len(old_texts)
    old_labels = list(ta.source_labels or []) if ta.source_labels else [None] * len(old_texts)
    # Hizayı koru: eski diziler eksikse None ile doldur.
    old_ts += [None] * (len(old_texts) - len(old_ts))
    old_ratings += [None] * (len(old_texts) - len(old_ratings))
    old_labels += [None] * (len(old_texts) - len(old_labels))

    merged_texts = old_texts + new_texts
    merged_ts = old_ts + (new_ts if new_ts else [None] * len(new_texts))
    merged_ratings = old_ratings + (new_ratings if new_ratings else [None] * len(new_texts))
    merged_labels = old_labels + (new_labels if new_labels else [None] * len(new_texts))

    ta.source_texts = merged_texts
    ta.source_timestamps = merged_ts if any(x is not None for x in merged_ts) else None
    ta.source_ratings = merged_ratings if any(x is not None for x in merged_ratings) else None
    ta.source_labels = merged_labels if any(x is not None for x in merged_labels) else None
    ta.updated_at = int(time.time())

    if payload.run:
        ta.status = "queued"
        ta.error = None
        db.add(BackgroundJobModel(task_type="run_text_analysis", text_analysis_id=ta.id, status="pending"))

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
