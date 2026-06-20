import time
from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from api.core.database import Base


class DbIntelligenceCacheModel(Base):
    """
    Project bazlı DatabaseIntelligenceEngine.analyze_schema cache'i.

    analyze_schema pahalı bir LLM çağrısıdır ve aynı şema için deterministik
    (temperature=0) sonuç üretir. Bu tablo her project için en son analiz
    sonucunu (model_dump) saklar; bir sonraki çalıştırmada şema değişmediyse
    LLM çağrısı tamamen atlanır.

    Invalidasyon: ``cache_key`` = hash(intelligence_model + raw_schema). Şema
    (dataset yapısı) değişince hash değişir ve cache otomatik geçersiz olur.
    """
    __tablename__ = "db_intelligence_cache"

    id = Column(Integer, primary_key=True, index=True)

    # Her project'in tek bir cache satırı vardır (per-project cache).
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # hash(model + raw_schema) — şema/dataset değişince eşleşmez → recompute.
    cache_key = Column(String, nullable=False)

    # DatabaseIntelligenceModel.model_dump() (augmentation sonrası nihai hali).
    payload = Column(JSON, nullable=False)

    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))
