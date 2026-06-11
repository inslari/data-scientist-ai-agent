"""
Entity Resolution Schemas
==========================
Entity-level veri kalıplarını tespit eden ve çözümleyen
entity_resolver servisinin veri yapıları.

Üç deterministik sinyal:
  1. entity_ratio  — toplam satır / benzersiz entity sayısı
  2. target_stable — her entity'de target tek değer mi
  3. has_temporal  — timestamp kolonu var mı
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EntitySignals(BaseModel):
    """Entity-level sinyaller — saf matematik, LLM gerektirmez."""
    model_config = {"extra": "forbid"}

    entity_column: str = Field(
        description="Tespit edilen entity kolonu (ör: 'customer_id')"
    )
    entity_ratio: float = Field(
        description="total_rows / unique_entities — 1.0 ise entity başına tek satır"
    )
    total_rows: int = Field(
        description="Orijinal veri satır sayısı"
    )
    unique_entities: int = Field(
        description="Benzersiz entity sayısı"
    )
    target_stable: bool = Field(
        description="True ise her entity'de target tek değer (entity-level problem)"
    )
    has_temporal: bool = Field(
        description="Timestamp kolonu var mı"
    )
    temporal_columns: list[str] = Field(
        default_factory=list,
        description="Tespit edilen timestamp kolonları"
    )


class EntityResolution(BaseModel):
    """Entity resolution kararı + sonuçları."""
    model_config = {"extra": "forbid"}

    action: Literal["aggregate", "entity_split", "passthrough"] = Field(
        description=(
            "aggregate: entity-level problem, satırları grupla. "
            "entity_split: satır-level problem ama entity-based train/test split gerekli. "
            "passthrough: entity başına tek satır, hiçbir şey yapma."
        )
    )
    signals: EntitySignals = Field(
        description="Hesaplanan deterministik sinyaller"
    )
    aggregated_data_path: str = Field(
        default="",
        description="aggregate ise yeni parquet path, değilse boş"
    )
    aggregation_stats: dict = Field(
        default_factory=dict,
        description=(
            "Aggregation istatistikleri: "
            "{original_rows, aggregated_rows, features_created, entity_column, ...}"
        )
    )
    message: str = Field(
        description="UI'da gösterilecek özet mesaj"
    )
