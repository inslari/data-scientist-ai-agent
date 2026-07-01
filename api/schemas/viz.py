from pydantic import BaseModel
from typing import Optional


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


class VizResponse(BaseModel):
    """Üretilen bir görselleştirmenin yanıtı (frontend galerisine eklenir)."""
    id: Optional[int] = None
    summary: str
    image_b64: Optional[str] = None       # matplotlib PNG (base64)
    chart: Optional[dict] = None          # interaktif recharts payload | None
    sql: Optional[str] = None
    spec: Optional[dict] = None
    columns: list[str] = []
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

    model_config = {"from_attributes": True}
