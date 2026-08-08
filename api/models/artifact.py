from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from api.core.database import Base
import time


class ArtifactModel(Base):
    """Katman 1 inference artifact'inin SORGULANABİLİR index'i (disk = doğruluk kaynağı).

    Ağır Artifact ``data/artifacts/<id>/`` altında (JSON + .pkl blob'lar); bu tablo yalnız
    metadata taşır → "bu projenin serve edilecek artifact'i hangisi" sorusu SQL ile cevaplanır
    (dizin taraması ölçeklenmez + champion seçilemez). Satırı WORKER yazar (Faz A/B/C servisleri
    değil — onlar app-DB'siz graph'ta koşar): koşu sonunda ``final_state``'teki id'den + diskteki
    ``artifact.json``'dan (nihai phase/model_ref/deployable) tek UPSERT.

    ``phase``: features|model|complete (yaşam döngüsü). ``is_champion``: serve edilen sürüm —
    yalnızca MANUEL promotion ile değişir (`POST .../artifacts/{id}/promote`). Worker upsert'i bu
    kolona DOKUNMAZ; yeni satır ``is_champion=False``. Otomatik promotion KALDIRILDI (yeniden eğitim
    serving'i sessizce değiştirmesin — kullanıcı hangi modelin canlı olduğuna kendi karar verir).
    """

    __tablename__ = "artifacts"

    id = Column(String, primary_key=True, index=True)                # artifact_id (uuid hex)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    phase = Column(String, nullable=True)                           # features | model | complete
    deployable = Column(Boolean, default=False)
    model_path = Column(String, nullable=True)                      # data/models/model_<uuid>.pkl (REFERANS)
    model_name = Column(String, nullable=True)
    schema_version = Column(String, nullable=True)                  # artifact_schema_version (ör. 1.2)
    # Serve edilen sürüm. Endpoint: WHERE project_id=? AND is_champion AND deployable AND phase='complete'.
    is_champion = Column(Boolean, default=False, index=True)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))
