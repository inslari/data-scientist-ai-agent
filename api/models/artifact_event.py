from sqlalchemy import Column, Integer, String, Float, ForeignKey
from api.core.database import Base
import time


class ArtifactEventModel(Base):
    """Champion promotion OLAY günlüğü (Katman 2 "Olaylar" paneli — Deployments'ın altı).

    Champion değişimi SESSİZ olmamalı: hangi sürümden hangisine, ne zaman, KİM tarafından.
    Manuel promotion (`POST .../artifacts/{id}/promote`) her başarılı geçişte BİR satır yazar —
    promote transaction'ının İÇİNDE (aynı commit), böylece "champion değişti ama olay yazılmadı"
    (ya da tersi) durumu YAPISAL olarak imkânsız (atomiklik testli).

    ``from_artifact_id`` = önceki champion (İLK promotion'da NULL — öncesi yok). ``at`` FLOAT epoch
    (predictions.label_backfilled_at/monitoring computed_at deseni; sıralama+keyset tutarlılığı).

    ``actor`` NULLABLE — bugün her promotion manuel (bir kullanıcı) → dolu. İleride otomatik
    yeniden-eğitim promotion'ı gelirse ``actor=NULL, event_type='auto_promote'`` yazılır; ŞEMA
    şimdiden hazır (kolon eklemek migration + geçmiş kayıp demek → baştan nullable)."""

    __tablename__ = "artifact_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    artifact_id = Column(String, index=True)                  # promote EDİLEN (yeni champion)
    event_type = Column(String, nullable=False)               # promote | auto_promote (ileride)
    from_artifact_id = Column(String, nullable=True)          # önceki champion (ilk promotion → NULL)
    at = Column(Float, default=lambda: time.time(), index=True)   # epoch (float; sıralama/keyset)
    actor = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)  # KİM (auto → NULL)
