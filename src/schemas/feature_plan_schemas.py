from pydantic import BaseModel, Field
from typing import Literal

class TargetDefinition(BaseModel):
    model_config = {"extra": "ignore"}

    column_name: str = Field(
        description="Hedef değişkenin adı. Mevcut değilse türetilecek isim (örn: 'churn', 'revenue', 'segment')"
    )
    is_derived: bool = Field(
        description="True ise hedef değişken veriden türetilecek, False ise zaten mevcut"
    )
    derivation_type: Literal["time_based", "threshold_based", "rank_based", "none"] = Field(
        description=(
            "time_based: tarih kolonundan türet (churn gibi). "
            "threshold_based: sayısal eşikten türet. "
            "rank_based: sıralamaya göre türet. "
            "none: zaten mevcut kolon"
        )
    )
    date_column: str = Field(
        default="",
        description="time_based için kullanılacak tarih kolonu adı"
    )
    threshold_days: int = Field(
        default=0,
        description="time_based için gün eşiği. 0 ise dinamik hesaplanır (verinin %80'lik noktası)"
    )
    threshold_value: float = Field(
        default=0.0,
        description="threshold_based için sayısal eşik değeri"
    )
    positive_class_description: str = Field(
        description="Pozitif sınıfın (1) ne anlama geldiği. Örn: 'Son 180 günde sipariş vermemiş müşteri'"
    )

class AggregationConfig(BaseModel):
    model_config = {"extra": "ignore"}

    required: bool = Field(
        description="True ise veri aggregation gerektirir (müşteri bazında gibi)"
    )
    group_by_column: str = Field(
        default="",
        description="Aggregation için groupby kolonu. Örn: 'customer_id'"
    )
    level: Literal["customer", "product", "order", "time", "none"] = Field(
        description="Aggregation seviyesi"
    )

class FeaturePlan(BaseModel):
    model_config = {"extra": "ignore"}

    analysis_summary: str = Field(
        description="Kullanıcının ne yapmak istediğinin tek cümlelik Türkçe özeti"
    )
    target: TargetDefinition
    aggregation: AggregationConfig
    feature_groups: list[Literal["rfm", "temporal_trend", "behavioral", "demographic", "product", "none"]] = Field(
        description=(
            "Üretilecek feature grupları. "
            "rfm: Recency/Frequency/Monetary. "
            "temporal_trend: Zamansal trend (dönem bazlı sipariş/harcama). "
            "behavioral: Davranışsal (kategori çeşitliliği, basket size). "
            "demographic: Demografik (yaş, hesap yaşı). "
            "product: Ürün bazlı (kategori, rating, stok). "
            "none: Aggregation gerekmez, ham veri kullanılır"
        )
    )
    use_data_split_for_label: bool = Field(
        description=(
            "True ise verinin son %20'si label için, ilk %80'i feature için kullanılır. "
            "time_based hedeflerde genellikle True olmalı."
        )
    )
    math_interactions: list[str] = Field(
        default_factory=list,
        description=(
            "LLM tarafından önerilen mantıksal matematiksel etkileşimler. "
            "Örn: ['revenue / max(1, order_count)', 'price * quantity']. "
            "Sadece sayısal kolonlar için kullanılmalıdır."
        )
    )
    reasoning: str = Field(
        description="Bu planın neden seçildiğini açıklayan Türkçe metin"
    )