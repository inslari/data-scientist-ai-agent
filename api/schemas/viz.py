from pydantic import BaseModel
from typing import Optional, Union


class VizSuggestion(BaseModel):
    """Data Visualization öneri kartı (viz_suggestion_generator çıktısı)."""
    title: str
    description: Optional[str] = None
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    agg: Optional[str] = "none"


class VizRequest(BaseModel):
    """
    Görselleştirme isteği. İkisinden biri verilir:
    - message: serbest doğal dil isteği
    - suggestion: kullanıcının tıkladığı öneri kartı
    """
    message: Optional[str] = None
    suggestion: Optional[VizSuggestion] = None
    # Tema id'si ya da {"id": "custom", "colors": [...]}; verilmezse grubun varsayılanı
    theme: Optional[Union[str, dict]] = None
    # Sunucu PNG'sindeki sayı biçimlendirmesi ("tr" | "en")
    locale: Optional[str] = None


class VizThemeUpdate(BaseModel):
    """Veri grubunun varsayılan renk temasını değiştirir (None = katalog varsayılanı)."""
    theme: Optional[Union[str, dict]] = None


class VizPatch(BaseModel):
    """
    Bir grafiğin görünümünü LLM ve SQL çalıştırmadan değiştirir.

    `theme` yalnızca renkleri etkiler (istemci zaten yeniden çizer). `spec` alanları
    (chart_type, sort, top_n, data_labels, title, ...) saklanan veriden PNG'nin de
    yeniden üretilmesini gerektirir — ikisi de aynı endpoint'ten geçer.
    """
    theme: Optional[Union[str, dict]] = None
    spec: Optional[dict] = None
    locale: Optional[str] = None


class VizRefine(BaseModel):
    """Mevcut bir grafiği doğal dille değiştirme isteği ("bunu yatay yap", "ilk 5")."""
    message: str
    locale: Optional[str] = None


class VizReportRequest(BaseModel):
    """Seçilen grafiklerden PDF rapor üretir."""
    viz_ids: list[int]
    title: Optional[str] = None
    subtitle: Optional[str] = None
    include_summaries: bool = True
    include_sql: bool = False
    locale: Optional[str] = None


class VizResponse(BaseModel):
    """Üretilen bir görselleştirmenin yanıtı (frontend galerisine eklenir)."""
    id: Optional[int] = None
    summary: str
    image_b64: Optional[str] = None       # matplotlib PNG (base64) — indirme/dışa aktarım
    chart: Optional[dict] = None          # tema-bağımsız frontend payload'ı | None
    sql: Optional[str] = None
    spec: Optional[dict] = None
    columns: list[str] = []
    theme: Optional[Union[str, dict]] = None
    data: Optional[dict] = None           # {"columns","rows"} — tablo görünümü
    warning: Optional[str] = None
    clarification_needed: bool = False
    created_at: Optional[int] = None


class VizItem(BaseModel):
    """Kalıcı galeri kaydı (VisualizationModel)."""
    id: int
    created_at: int
    user_query: Optional[str] = None
    spec: Optional[dict] = None
    image_b64: Optional[str] = None
    chart: Optional[dict] = None
    sql: Optional[str] = None
    summary: Optional[str] = None
    theme: Optional[Union[str, dict]] = None
    data: Optional[dict] = None           # {"columns","rows"} — tablo görünümü

    model_config = {"from_attributes": True}
