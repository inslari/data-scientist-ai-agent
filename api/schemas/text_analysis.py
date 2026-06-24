from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional, List, Any, Dict


class TextAnalysisBase(BaseModel):
    title: str
    description: Optional[str] = None
    language: Optional[str] = "auto"


class TextAnalysisCreate(TextAnalysisBase):
    # Üç giriş yolu (öncelik sırası): yapılandırılmış liste (texts) → blok metin
    # (raw_text, satır/boş satıra göre bölünür) → CSV kolonu (csv_file_id +
    # text_column; opsiyonel date_column / rating_column ile hizalı tarih/puan).
    texts: Optional[List[str]] = None
    raw_text: Optional[str] = None

    # CSV kolon girişi (ürünün asıl veri akışı).
    csv_file_id: Optional[int] = None
    text_column: Optional[str] = None
    date_column: Optional[str] = None
    rating_column: Optional[str] = None
    # Ground-truth duygu etiketi kolonu (Durum/sentiment...). Performans karşılaştırması açar.
    label_column: Optional[str] = None

    @field_validator("texts")
    @classmethod
    def _strip_texts(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        return [t.strip() for t in v if isinstance(t, str) and t.strip()]


class TextAppendRequest(BaseModel):
    """Mevcut bir analize yeni geri bildirim ekler (incremental). Re-run cache'ten
    yararlanır: değişmeyen metinler yeniden LLM'e gitmez."""
    texts: Optional[List[str]] = None
    raw_text: Optional[str] = None
    csv_file_id: Optional[int] = None
    text_column: Optional[str] = None
    date_column: Optional[str] = None
    rating_column: Optional[str] = None
    label_column: Optional[str] = None
    run: bool = True  # ekledikten sonra otomatik çalıştır


class TextAnalysisUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    texts: Optional[List[str]] = None
    raw_text: Optional[str] = None


class TextAnalysisResponse(TextAnalysisBase):
    id: int
    user_id: Optional[int] = None
    # Detayda ham metinler (source_texts) ve büyük item_results DÖNMEZ — şişmesin.
    # source_texts gerekirse ayrı endpoint'ten, item_results sayfalı endpoint'ten.
    source_texts: Optional[List[str]] = None
    status: str
    result: Optional[Dict[str, Any]] = None   # item_results hariç (aggregate'ler kalır)
    report: Optional[str] = None
    error: Optional[str] = None
    item_count: Optional[int] = None          # toplam analiz edilen metin sayısı (pagination için)
    created_at: int
    updated_at: int

    model_config = ConfigDict(from_attributes=True)


class TextItemsPage(BaseModel):
    """item_results için sayfalı yanıt."""
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


class TextAnalysisListItem(BaseModel):
    """Liste görünümü için hafif özet (büyük result/source_texts hariç)."""
    id: int
    title: str
    description: Optional[str] = None
    status: str
    text_count: int = 0
    created_at: int
    updated_at: int

    model_config = ConfigDict(from_attributes=True)
