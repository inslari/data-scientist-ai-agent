import time
from sqlalchemy import Column, Integer, String, JSON, Index
from api.core.database import Base


class TextAnalysisCacheModel(Base):
    """
    İçerik-adresli cache (metin analizi maliyet/süre optimizasyonu).

    Aynı metin + aynı model için pahalı LLM/embedding çağrısını tekrar etmemek
    içindir. Hem tekrar çalıştırmalarda hem de farklı analizlerdeki örtüşen
    metinlerde çağrıları atlar.

    cache_key = sha256("{kind}|{model}|{text}")  → deterministik, çakışmasız.
      kind  = "classify" | "embed"
      model = sağlayıcı:model (örn. "openai:gpt-4o" / "openai:text-embedding-3-small")
      payload = classify için sınıflandırma dict'i; embed için list[float] vektör
    """
    __tablename__ = "text_analysis_cache"

    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(80), unique=True, index=True)
    kind = Column(String, index=True)
    model = Column(String)
    payload = Column(JSON)
    created_at = Column(Integer, default=lambda: int(time.time()))


# kind+model üzerinden hızlı tarama/temizlik için yardımcı indeks.
Index("ix_text_cache_kind_model", TextAnalysisCacheModel.kind, TextAnalysisCacheModel.model)
