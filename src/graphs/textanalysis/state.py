"""
Text-analysis LangGraph durumu ve LLM yapısal çıktı şemaları.

Bu graf, tabular ana hat (src/graph.py) ve Talk-to-Data grafiğinden bağımsızdır;
kendi TextAnalysisState'ini taşır. Akış doğrusaldır:
  preprocess → classify → theme → report
"""
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


class TextAnalysisState(TypedDict, total=False):
    # ── Girdi ──
    title: str
    raw_texts: list[str]          # ham yapıştırılan metinler
    # Opsiyonel, raw_texts ile hizalı zaman damgaları (örn. CSV tarih kolonundan).
    # Boş/None → zaman damgası metnin içinde aranır (agent tespiti).
    raw_timestamps: list          # list[str | None]
    # Opsiyonel, raw_texts ile hizalı kullanıcı puanları (örn. CSV rating kolonu).
    raw_ratings: list             # list[float | None]
    # Opsiyonel, raw_texts ile hizalı ground-truth duygu etiketleri (örn. CSV
    # "Durum"/"Görüş" gibi farklı isimli kolon; Olumlu/Olumsuz vb. serbest sözcük).
    raw_labels: list              # list[str | None]
    language: str                 # "auto" | "tr" | "en" ...

    # ── preprocess çıktısı ──
    texts: list[str]              # temizlenmiş metinler (analiz bunun üzerinde)
    timestamps: list              # temizlenmiş metinlerle hizalı (str|None)
    ratings: list                 # temizlenmiş metinlerle hizalı (float|None)
    labels: list                  # temizlenmiş metinlerle hizalı ground-truth (str|None)
    detected_language: str        # tespit/çözülmüş dil (tr|en) — prompt'lara aktarılır
    duplicate_groups: dict        # {temsilci_metin: birleşen_kopya_sayısı} (near-dup)

    # ── timestamp tespiti (agent) ──
    # {has_timestamp: bool, source: "column"|"text"|"none", confidence: float, example: str, note: str}
    timestamp_detection: dict

    # ── classify çıktısı (her metin için) ──
    # {index, text, sentiment, sentiment_score, emotion, category, urgency, summary, aspects}
    item_results: list[dict]
    sentiment_distribution: dict
    category_distribution: dict
    urgency_distribution: dict
    emotion_distribution: dict
    # Aspect-based: {aspect: {positive, negative, neutral, total}}
    aspect_distribution: dict
    # LLM duygusu ↔ CSV puanı tutarlılığı (rating kolonu varsa)
    rating_validation: dict
    # LLM tahmini ↔ ground-truth etiket karşılaştırmalı performansı (label kolonu varsa):
    # {n, accuracy, confusion, per_class, mismatches, labels_seen}
    label_validation: dict

    # ── theme çıktısı ──
    # {label, description, count, share, sentiment_breakdown, example_texts}
    themes: list[dict]
    priority_items: list[dict]    # en acil şikayet/talepler
    # Tema × zaman: her temanın dönem-dönem hacmi/yönü (yükselen/düşen)
    theme_trends: list[dict]

    # ── trend çıktısı (zaman damgası varsa) ──
    # [{period, count, avg_sentiment, positive, negative, neutral, ma_sentiment, delta_sentiment, spike}]
    time_trend: list[dict]

    # ── report çıktısı ──
    report: str                   # markdown yönetici özeti + trendler

    # ── kontrol ──
    current_step: Optional[str]
    error: Optional[str]
    error_category: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# LLM yapısal çıktı şemaları (with_structured_output ile kullanılır)
# ─────────────────────────────────────────────────────────────────────────────

class Aspect(BaseModel):
    """Aspect-based sentiment: bir geri bildirimde geçen konu + o konuya dair duygu."""
    aspect: str = Field(description="The aspect/topic mentioned, in Turkish (e.g. teslimat, fiyat, kalite, müşteri hizmetleri)")
    sentiment: str = Field(default="neutral", description="One of: positive, negative, neutral")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if k in {"aspect", "sentiment"}}
        return values


class ItemClassification(BaseModel):
    index: int = Field(description="0-based index of the text in the batch")
    sentiment: str = Field(description="One of: positive, negative, neutral")
    sentiment_score: float = Field(
        default=0.0,
        description="Sentiment polarity from -1.0 (very negative) to 1.0 (very positive)",
    )
    emotion: str = Field(
        default="neutral",
        description="Dominant emotion, e.g. anger, frustration, satisfaction, joy, disappointment, neutral",
    )
    category: str = Field(
        description="One of: complaint, request, praise, question, other",
    )
    urgency: str = Field(
        default="low",
        description="One of: low, medium, high, critical",
    )
    summary: str = Field(
        default="",
        description="A one-sentence summary of the feedback, written in Turkish",
    )
    aspects: list[Aspect] = Field(
        default_factory=list,
        description="Aspect-based breakdown: the specific topics mentioned and the sentiment toward each",
    )
    timestamp: str = Field(
        default="",
        description="ISO 8601 date (YYYY-MM-DD) if the text explicitly contains a date/time; otherwise empty. Never fabricate.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"index", "sentiment", "sentiment_score", "emotion",
                       "category", "urgency", "summary", "aspects", "timestamp"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class TimestampDetection(BaseModel):
    """Agent çıktısı: veride zaman damgası var mı, hangi formatta?"""
    has_timestamp: bool = Field(description="True if the feedback texts contain a date/time")
    confidence: float = Field(default=0.0, description="0..1 confidence in the detection")
    example: str = Field(default="", description="An example of the detected date/time, verbatim from the data")
    note: str = Field(default="", description="Short Turkish note about the detection (format, where it appears)")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"has_timestamp", "confidence", "example", "note"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class BatchClassification(BaseModel):
    items: list[ItemClassification] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"items"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class ThemeLabel(BaseModel):
    label: str = Field(description="Short theme name (2-5 words), in Turkish")
    description: str = Field(
        default="",
        description="One-sentence description of what this theme captures, in Turkish",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"label", "description"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class ClusterLabel(BaseModel):
    """Batch tema etiketlemede tek bir kümenin sonucu (cluster_id ile hizalı)."""
    cluster_id: int = Field(description="0-based id of the cluster being labeled")
    label: str = Field(description="Short theme name (2-5 words)")
    description: str = Field(default="", description="One-sentence description")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"cluster_id", "label", "description"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class BatchThemeLabel(BaseModel):
    """Tüm kümeleri tek LLM çağrısında etiketler (N çağrı yerine 1)."""
    clusters: list[ClusterLabel] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"clusters"}
            return {k: v for k, v in values.items() if k in allowed}
        return values
