from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional, List, Any, Dict


class TextAnalysisBase(BaseModel):
    title: str
    description: Optional[str] = None
    language: Optional[str] = "auto"


class TextAnalysisCreate(TextAnalysisBase):
    # Kullanıcı ya yapılandırılmış bir liste (texts) ya da tek bir blok metin
    # (raw_text — satırlara/boş satırlara göre bölünür) gönderebilir.
    texts: Optional[List[str]] = None
    raw_text: Optional[str] = None

    @field_validator("texts")
    @classmethod
    def _strip_texts(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        return [t.strip() for t in v if isinstance(t, str) and t.strip()]


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
