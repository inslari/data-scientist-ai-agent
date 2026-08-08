"""Manuel champion promotion + olay günlüğü (Katman 1 serving index'inin yazma yüzü).

Karar: champion promotion OTOMATİK değil MANUEL. Otomatik promotion = sessiz davranış değişikliği
(yeniden eğitim serving'i anında değiştirir); kullanıcı hangi modelin canlı olduğuna kendi karar verir.
Worker artık yalnız artifact metadata'sını UPSERT eder (`_upsert_artifact_row`, is_champion'a dokunmaz);
canlıya alma tek yerden, bu endpoint'ten geçer.

İKİ ENDPOINT:
  * ``POST /projects/{id}/artifacts/{artifact_id}/promote`` — bir artifact'i champion yap.
    FAIL-CLOSED ön-koşul (`load_for_serving` üç kapı → 409) + ATOMİK tek transaction (eski champion'ı
    düşür + yeniyi çıkar + olay yaz, hepsi tek commit → yarım promotion imkânsız) + yanıtta
    ``previous_champion``.
  * ``GET /projects/{id}/artifacts/events`` — promotion olay günlüğü (kim/ne zaman/hangi sürümden hangisine).

⚠️ EŞZAMANLILIK (SQLite ↔ Postgres FARKI, sessiz kalmasın): satır kilidi `with_for_update()` YALNIZ
Postgres'te gerçek kilit → iki eşzamanlı promote seri hâle gelir (biri diğerini bekler, sonuç TEK champion).
SQLite (test ortamı) FOR UPDATE'i no-op'lar → teorik olarak iki istek iç içe girip iki champion bırakabilir.
Üretim Postgres → yarış YOK; ayrıca sistem tek-kullanıcılı ve promotion elle/nadir → pratik risk yok denecek
kadar düşük. Yine de kod-yolu Postgres'te doğru kilidi alır (gate: is_postgres), davranış farkı BURADA belgeli.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.core.database import SQLALCHEMY_DATABASE_URL, get_db
from api.core.deps import get_current_user
from api.models.artifact import ArtifactModel
from api.models.artifact_event import ArtifactEventModel
from api.models.project import ProjectModel
from api.models.user import UserModel
from src.core.config import settings
from src.core.logging_config import get_logger
from src.services.inference.references import ModelFileMissing
from src.services.inference.serialize import ArtifactNotServiceable, load_for_serving

logger = get_logger(__name__)

router = APIRouter(prefix="/projects", tags=["Artifacts"])

_IS_POSTGRES = SQLALCHEMY_DATABASE_URL.startswith("postgresql")


def _require_project(db: Session, project_id: int, user: UserModel) -> ProjectModel:
    project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def is_serviceable(artifact_id: str) -> bool:
    """"Bu artifact GERÇEKTEN serve edilebilir mi?" probe'u — CTA/promotable kapısı için FAIL-CLOSED.

    ``load_for_serving`` üç kapısı (complete ∧ deployable ∧ model-dosyası). HERHANGİ yükleme hatası
    (ArtifactNotServiceable/ModelFileMissing ama aynı zamanda artifact.json yok = FileNotFoundError,
    bozuk şema vb.) → ``False``. Bu bir keşif/probe (serving hot-path DEĞİL) → hata = "hayır" (CTA
    gösterme), 500 değil. Scoring/promote'un asıl fail-closed kapısı ayrı (spesifik hata mesajı verir)."""
    try:
        load_for_serving(artifact_id)
        return True
    except Exception:
        return False


def _promotable_candidate(db: Session, project_id: int):
    """not_promoted hedefi — EN YENİ *gerçekten serve edilebilir* complete+deployable artifact.

    Backend belirler (deterministik, tek yer — frontend seçmez). "Gerçekten": DB satırı complete+
    deployable dese de model dosyası diskte olmayabilir → ``is_serviceable`` geçmeyen aday CTA'ya
    ÇIKMAZ (kullanıcı butona basıp 409 almasın). En yeniden geriye ilk GEÇEN döner."""
    cands = db.query(ArtifactModel).filter(
        ArtifactModel.project_id == project_id,
        ArtifactModel.phase == "complete",
        ArtifactModel.deployable.is_(True),
        ArtifactModel.model_path.isnot(None),
        ArtifactModel.is_champion.is_(False),
    ).order_by(ArtifactModel.created_at.desc()).all()
    for c in cands:
        if is_serviceable(c.id):
            return c
    return None


@router.post("/{project_id}/artifacts/{artifact_id}/promote")
def promote_artifact(
    project_id: int,
    artifact_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Bir artifact'i champion yap — FAIL-CLOSED + ATOMİK.

    ADIMLAR: (1) sahiplik + artifact bu projeye ait mi (404); (2) ön-koşul FAIL-CLOSED —
    ``load_for_serving`` (complete ∧ deployable ∧ model-dosyası; scoring ile AYNI kapı) → 409;
    (3) ATOMİK transaction: (Postgres'te satır kilidi →) eski champion(lar)ı düşür + bu artifact'i
    champion yap + ``artifact_events``'e olay yaz + TEK commit. Ortada herhangi bir hata → rollback →
    HİÇBİRİ olmaz (eski champion korunur, yeni promote OLMAZ, olay yazılmaz). Yanıt: ``previous_champion``.
    """
    _require_project(db, project_id, current_user)

    target = db.query(ArtifactModel).filter(
        ArtifactModel.id == artifact_id,
        ArtifactModel.project_id == project_id,
    ).first()
    if target is None:
        raise HTTPException(status_code=404, detail="Artifact bu projede bulunamadı")

    # ── Ön-koşul FAIL-CLOSED: serve edilemez artifact champion OLAMAZ (scoring ile aynı üç kapı) ──
    try:
        load_for_serving(artifact_id)
    except (ArtifactNotServiceable, ModelFileMissing) as e:
        raise HTTPException(status_code=409,
                            detail=f"Artifact canlıya alınamaz (serve edilemez): {e}")
    except Exception as e:
        # artifact.json yok / bozuk şema vb. — DB satırı var ama disk bütünlüğü kırık → yine 409 (fail-closed).
        raise HTTPException(status_code=409,
                            detail=f"Artifact canlıya alınamaz (yüklenemedi): {e}")

    # ── ATOMİK: eski düşür + yeni çıkar + olay yaz, TEK commit ──
    try:
        # Postgres: aynı projenin artifact satırlarını KİLİTLE (eşzamanlı promote'u seri yap).
        # SQLite: no-op (üstteki modül docstring'i — davranış farkı orada belgeli).
        q = db.query(ArtifactModel).filter(ArtifactModel.project_id == project_id)
        if _IS_POSTGRES:
            q = q.with_for_update()
        artifacts = q.all()

        prev_champ_id = None
        for a in artifacts:
            if a.is_champion and a.id != artifact_id:
                prev_champ_id = a.id            # tek champion invariant → ilki yeter (birden fazlaysa hepsi düşer)
                a.is_champion = False
            elif a.is_champion and a.id == artifact_id:
                a.is_champion = False           # zaten champion → aşağıda tekrar True (idempotent + event yazılır)
        # target aynı obje setinde (q.all projeye çitli, target bu projede) → doğrudan set.
        target.is_champion = True
        target.updated_at = int(time.time())

        ev = ArtifactEventModel(
            project_id=project_id, artifact_id=artifact_id, event_type="promote",
            from_artifact_id=prev_champ_id, at=time.time(), actor=current_user.id,
        )
        db.add(ev)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        # Transaction'ın herhangi bir adımı patladı → HİÇBİRİ kalıcı olmasın (yarım promotion = bozuk durum).
        db.rollback()
        logger.error(f"[promote] atomik promotion BAŞARISIZ (rollback; hiçbir değişiklik uygulanmadı) "
                     f"project={project_id} artifact={artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Promotion başarısız (rollback edildi): {e}")

    logger.info(f"[promote] champion={artifact_id} (önceki={prev_champ_id}) "
                f"project={project_id} actor={current_user.id}")
    return {
        "promoted_artifact_id": artifact_id,
        "previous_champion": prev_champ_id,
        "actor": current_user.id,
    }


@router.get("/{project_id}/artifacts/events")
def artifact_events(
    project_id: int,
    limit: int = Query(default=None),
    offset: int = Query(default=0),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Promotion olay günlüğü — champion değişimleri (kim / ne zaman / hangi sürümden hangisine).

    Champion promotion NADİR olay → basit ``limit``/``offset`` sayfalama yeterli (predictions'ın
    keyset zorunluluğu yok; head-insert drift'i pratik önemsiz). ``at`` DESC sıralı (en yeni önce).
    ``actor_email`` display için çözülür (``actor`` NULL → otomatik promotion, ileride)."""
    _require_project(db, project_id, current_user)
    default_n = settings.MONITORING_EVENTS_PAGE_SIZE
    max_n = settings.MONITORING_EVENTS_PAGE_MAX
    eff_n = default_n if limit is None else limit
    eff_n = max(1, min(int(eff_n), max_n))                  # [1, MAX]'a kırp (hata değil — days/predictions deseni)
    eff_off = max(0, int(offset or 0))

    base = db.query(ArtifactEventModel).filter(ArtifactEventModel.project_id == project_id)
    total = base.count()
    rows = (base.order_by(ArtifactEventModel.at.desc(), ArtifactEventModel.id.desc())
                .offset(eff_off).limit(eff_n).all())

    # actor (user_id) → email TEK sorguda (N+1 yok); NULL actor'ler atlanır.
    actor_ids = {r.actor for r in rows if r.actor is not None}
    email_by_id: dict = {}
    if actor_ids:
        for uid, email in (db.query(UserModel.id, UserModel.email)
                             .filter(UserModel.id.in_(actor_ids)).all()):
            email_by_id[uid] = email

    events = [{
        "id": r.id,
        "event_type": r.event_type,
        "artifact_id": r.artifact_id,             # promote EDİLEN (yeni champion)
        "from_artifact_id": r.from_artifact_id,   # önceki (ilk promotion → null)
        "at": r.at,
        "actor": r.actor,                         # user_id (null → otomatik promotion, ileride)
        "actor_email": email_by_id.get(r.actor),
    } for r in rows]
    return {
        "window_timezone": "UTC",
        "events": events,
        "total": total,
        "limit_applied": eff_n,
        "offset": eff_off,
        "has_more": (eff_off + len(rows)) < total,
    }
