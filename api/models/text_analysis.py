import time
from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from api.core.database import Base


class TextAnalysisModel(Base):
    """
    Yazılı geri bildirim (yorum / şikayet / talep) analizi işi.

    Tabular ML pipeline'ından (ProjectModel) tamamen bağımsız, paralel bir
    özelliktir. Veri "doğrudan metin yapıştırma" ile girilir: kullanıcı bir
    metin listesi (her satır bir yorum/çağrı dökümü) yapıştırır, işi çalıştırır
    ve hibrit (lokal kümeleme + LLM etiketleme) bir analiz + markdown rapor alır.

    Çıktılar `result` (JSON) içinde toplanır:
      - item_results: her metin için sentiment/kategori/aciliyet/özet
      - sentiment_distribution / category_distribution / urgency_distribution
      - themes: lokal kümeleme + LLM etiketli temalar
      - priority_items: en acil şikayetler (önceliklendirme tablosu)
    Yönetici özeti + trendler `report` (markdown) alanında tutulur.
    """
    __tablename__ = "text_analyses"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text, nullable=True)

    # Ham yapıştırılan metinler (her eleman bir geri bildirim).
    source_texts = Column(JSON, nullable=True)   # list[str]
    language = Column(String, nullable=True)     # "auto" | "tr" | "en" ...

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)

    status = Column(String, default="idle")  # idle, queued, running, completed, failed
    result = Column(JSON, nullable=True)      # toplu analiz çıktısı (yukarıdaki yapı)
    report = Column(Text, nullable=True)      # markdown yönetici özeti
    error = Column(Text, nullable=True)

    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))

    user = relationship("UserModel")
