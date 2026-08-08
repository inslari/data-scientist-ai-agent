"""Katman 2 — monitoring OKUMA API'si (dashboard'un veri kaynağı).

İLKE: endpoint HESAPLAMAZ, yalnız ŞEKİL verir. Aggregator (`src/services/monitoring/aggregator.py`)
her şeyi `monitoring_summary`'ye zaten yazdı; bu üç endpoint YALNIZ okur (biri `predictions` COUNT).
Tek doğruluk kaynağı aggregator kalır → dashboard mantık taşımaz. Read-only, senkron `def`, sahiplik kontrollü.

DÖRT DİSİPLİN API SINIRINDA ZORLANIR (frontend'e bırakılmaz):
  1. HER SAYIDA n → her metrik bloğu kendi ``n``'ini taşır (çıplak sayı imkânsız).
  2. TAHMİNİ ↔ GERÇEKLEŞEN AYRI → ``cbpe`` ve ``realized`` iki AYRI blok, her biri kendi ``gate_status`` +
     ``assumes`` damgası. Dashboard ikisini asla harmanlamaz ("tahmini 0.81 / gerçekleşen 0.79" iki alandan).
  3. DÜŞÜK-n / DÜŞÜK-KAPSAM DAMGASI → eşikler her yanıtta ``config`` bloğunda (settings'ten). Frontend
     hardcode etmez ("deterministik olan kod tarafından basılır" + CLAUDE.md no-hardcode).
  4. DETERMİNİSTİK DURUM ROZETİ → ``status`` / ``gate_status`` string'leri aynen geçer.

ÜÇ ENDPOINT:
  * ``GET .../monitoring/summary?days=30&artifact_id=`` → pencere listesi (trend). per_feature YOK (yanıt
    şişer; trend psi_overall/psi_max ile çizilir). ``days`` config'ten üst-sınırlı.
  * ``GET .../monitoring/summary/{window_start}?artifact_id=`` → TEK pencere DETAYI (per_feature DAHİL).
  * ``GET .../monitoring/deployments`` → artifact listesi + champion + son pencere durumu (+ total_predictions).
  * ``GET .../monitoring/latest`` → champion son penceresi (hero kartları, tek çağrı; per_feature dahil).

ZAMAN: aggregator UTC gün sınırı kullanır → tüm epoch'lar UTC. Yanıt ``window_timezone:"UTC"`` taşır →
dashboard yerel saatte gösterse bile sınırın UTC olduğunu AÇIKÇA belirtir (aksi halde "28 Tem penceresi
kullanıcının 27'sinde başlıyor" karışıklığı).
"""
from __future__ import annotations

import base64
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, not_, or_
from sqlalchemy.orm import Session, undefer

from api.core.database import get_db
from api.core.deps import get_current_user
from api.models.artifact import ArtifactModel
from api.models.monitoring_summary import MonitoringSummaryModel
from api.models.prediction import PredictionModel
from api.models.project import ProjectModel
from api.models.user import UserModel
from src.core.config import settings
from api.routes.artifacts import _promotable_candidate, is_serviceable
from src.services.inference.references import ModelFileMissing
from src.services.inference.serialize import ArtifactNotServiceable, load_for_serving

router = APIRouter(prefix="/projects", tags=["Monitoring"])

# Portfolio CROSS-PROJECT bir kaynak → `/projects/{id}` prefix'i yanıltıcı olurdu (belirli projeye
# ait değil, kullanıcının TÜM projelerini kesitler). AYRI router, prefix="/monitoring".
portfolio_router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

_DAY = 86_400


def _require_project(db: Session, project_id: int, user: UserModel) -> ProjectModel:
    project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _config_block() -> dict:
    """Eşik + sınır bloğu — HER yanıtta. Dashboard düşük-n/kapsam damgalarını ve `days` sınırını
    BURADAN okur (frontend hardcode etmez; determinizm + no-hardcode)."""
    return {
        "min_n_psi": settings.MONITORING_MIN_N_PSI,
        "min_n_cbpe": settings.MONITORING_MIN_N_CBPE,
        "min_n_quality": settings.MONITORING_MIN_N_QUALITY,
        "min_n_realized": settings.MONITORING_MIN_N_REALIZED,
        "label_coverage_trust": settings.MONITORING_LABEL_COVERAGE_TRUST,
        "default_window_days": settings.MONITORING_DEFAULT_WINDOW_DAYS,
        "max_window_days": settings.MONITORING_MAX_WINDOW_DAYS,
        # FAZ 1 endpoint sınırları (frontend hardcode etmez): schema dropdown pratik-eşiği + predictions
        # sayfa boyutu. Tek config yüzeyi → tüm yanıtlarda taşınır (schema/predictions ilgilendirir, diğerleri
        # de görür → zararsız, tek kaynak).
        "schema_max_known_values": settings.MONITORING_SCHEMA_MAX_KNOWN_VALUES,
        "predictions_page_size": settings.MONITORING_PREDICTIONS_PAGE_SIZE,
        "predictions_page_max": settings.MONITORING_PREDICTIONS_PAGE_MAX,
        # PSI drift bandı (LİTERATÜR eşikleri) — dashboard psi_max tooltip'inde "uyarı ≥ / kritik ≥"
        # bağlamını BURADAN okur (frontend hardcode etmez; "çıplak eşik yok" → eşik config'ten).
        "psi_warn": settings.MONITORING_PSI_WARN,
        "psi_critical": settings.MONITORING_PSI_CRITICAL,
    }


def _window_block(s: MonitoringSummaryModel, include_per_feature: bool = False,
                  include_distribution: bool = False) -> dict:
    """Bir `monitoring_summary` satırı → dashboard bloğu. `include_per_feature`/`include_distribution`
    YALNIZ detay/latest'te True (summary listesinde yanıt şişmesin — trend overall/max + psi_scalars ile).

    Her metrik bloğu kendi ``n``'ini taşır (disiplin 1). ``cbpe`` (tahmini) ↔ ``realized`` (gerçekleşen)
    AYRI (disiplin 2). ``realized.n`` = etiketli satır (labeled_n); metrik NULL iken bile label_coverage'tan
    türetilir (coverage HER ZAMAN dolu → dashboard "kapsam %N" satırı metrik yokken de render eder).
    ``overall_status`` triage rozeti (Faz 0) — HER blokta (aggregator yazdı, aynen taşınır)."""
    psi = {
        "overall": s.psi_overall,
        "max": s.psi_max,
        "degenerate_count": s.psi_degenerate_count,
        "importance_attributed_ratio": s.importance_attributed_ratio,
        "degenerate_importance_mass": s.degenerate_importance_mass,
        "scalars": s.psi_scalars,               # {col: psi} — HAFİF (list sparkline; per_feature YÜKLEMEZ)
        "n": s.n_sampled,                       # PSI input_row örneklem satırlarını kullanır
    }
    if include_per_feature:
        psi["per_feature"] = s.psi_per_feature   # {col: {psi, n_bins, degenerate, reason, importance_weight}}
    if include_distribution:
        # ⚠️ deferred() kolon → yalnız BURADA erişilir (detay). List bu dalı ÇALIŞTIRMAZ → N lazy-load yok.
        psi["distribution"] = s.psi_distribution  # {col: {kind, edges/categories, ref_props, prod_counts, prod_total}}
    # labeled_n: metrik varsa içinde; yoksa coverage×n_total'dan türet (coverage=round(labeled_n/n_total,6)
    # → geri kazanım tam sayı). coverage None (hiç grup yok) → 0.
    labeled_n = (round(s.label_coverage * s.n_total)
                 if s.label_coverage is not None else 0)
    return {
        "artifact_id": s.artifact_id,
        "window_start": s.window_start,
        "window_end": s.window_end,
        "window_kind": s.window_kind,
        "computed_at": s.computed_at,
        "status": s.status,                     # ok | insufficient_n | no_reference (PSI durum rozeti)
        "overall_status": s.overall_status,     # healthy | warning | critical | unknown (TRIAGE rozeti, Faz 0)
        "overall_status_reason": s.overall_status_reason,   # {driver, code(i18n), value, suppressed_axes, ...}
        "n_total": s.n_total,
        "n_sampled": s.n_sampled,
        "excluded_ui_count": s.excluded_ui_count,   # DIŞLANAN manuel deneme (source=ui_manual) — first-class (0 dahil)
        "psi": psi,
        "cbpe": {                               # TAHMİNİ perf (concept-drift altında iyimser sapar)
            "gate_status": s.cbpe_gate_status,  # ok | suppressed_calibration | suppressed_no_baseline | insufficient_n
            "estimate": s.cbpe_estimate,        # {estimated_*, baseline_*, assumes:"covariate_drift_only"} | null
            "n": s.n_total,
        },
        "realized": {                           # GERÇEKLEŞEN perf (actual_label backfill; CBPE'nin doğrulaması)
            "gate_status": s.realized_gate_status,   # ok | insufficient_labels | no_labels
            "metrics": s.realized_metrics,      # {accuracy, roc_auc, positive_rate_deviation, assumes:"labels_missing_at_random"} | null
            "label_coverage": s.label_coverage, # HER ZAMAN dolu (metrik null olsa da)
            "n": labeled_n,
        },
        "quality": s.quality_rates,             # {unseen:{count,rate}, out_of_range:{count,rate}, n, alarm_eligible}
    }


def _serving_champion(db: Session, project_id: int):
    """Serve edilen champion (scoring.py ile AYNI filtre: is_champion ∧ deployable ∧ complete)."""
    return db.query(ArtifactModel).filter(
        ArtifactModel.project_id == project_id,
        ArtifactModel.is_champion.is_(True),
        ArtifactModel.deployable.is_(True),
        ArtifactModel.phase == "complete",
    ).first()


def _no_champion_payload(db: Session, project_id: int) -> dict:
    """Champion yokken "boş-reason"u İKİYE ayır (manuel promotion): eğitilmemiş mi, canlıya-alınmamış mı?

      * ``not_trained``  — hiç serve edilebilir (complete+deployable) artifact yok → CTA "Model eğit".
      * ``not_promoted`` — var ama champion değil → CTA "Canlıya al" + ``promotable_artifact_id`` (backend
        belirler = EN YENİ *gerçekten* serve edilebilir; ``load_for_serving`` ile DOĞRULANMIŞ → kullanıcı
        butona basıp 409 almaz).

    Otomatik promotion kaldırıldığından ``not_promoted`` artık NORMAL hâl (eğitilen model uykuda bekler)."""
    cand = _promotable_candidate(db, project_id)
    if cand is None:
        return {"reason": "not_trained", "promotable_artifact_id": None}
    return {"reason": "not_promoted", "promotable_artifact_id": cand.id}


@router.get("/{project_id}/monitoring/summary")
def monitoring_summary(
    project_id: int,
    days: int = Query(default=None),
    artifact_id: str = Query(default=None),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Pencere listesi (trend/zaman-serisi). per_feature YOK. `days` config üst-sınırına kırpılır."""
    _require_project(db, project_id, current_user)
    default_days = settings.MONITORING_DEFAULT_WINDOW_DAYS
    max_days = settings.MONITORING_MAX_WINDOW_DAYS
    eff_days = default_days if days is None else days
    eff_days = max(1, min(int(eff_days), max_days))         # [1, MAX]'a kırp (hata değil)
    now_day_start = (int(time.time()) // _DAY) * _DAY
    cutoff = now_day_start - eff_days * _DAY

    q = db.query(MonitoringSummaryModel).filter(
        MonitoringSummaryModel.project_id == project_id,
        MonitoringSummaryModel.window_start >= cutoff,
    )
    if artifact_id:
        q = q.filter(MonitoringSummaryModel.artifact_id == artifact_id)
    rows = q.order_by(MonitoringSummaryModel.window_start.asc(),
                      MonitoringSummaryModel.artifact_id.asc()).all()

    return {
        "window_timezone": "UTC",
        "days_requested": days,                 # ham istek (None olabilir)
        "days_applied": eff_days,               # kırpılmış (dashboard sınırı config'ten teyit eder)
        "windows": [_window_block(s, include_per_feature=False) for s in rows],
        "config": _config_block(),
    }


@router.get("/{project_id}/monitoring/summary/{window_start}")
def monitoring_window_detail(
    project_id: int,
    window_start: int,
    artifact_id: str = Query(default=None),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """TEK pencere DETAYI — per_feature DAHİL (seçili pencerenin kolon-bazlı PSI'ı). artifact_id
    opsiyonel: champion-bölme yüzünden aynı window_start'ta birden çok artifact olabilir → verilmezse
    hepsi döner."""
    _require_project(db, project_id, current_user)
    # deferred psi_distribution'ı BU sorguda undefer et (detay onu döner; list DÖNMEZ → orada deferred kalır).
    q = db.query(MonitoringSummaryModel).options(
        undefer(MonitoringSummaryModel.psi_distribution)).filter(
        MonitoringSummaryModel.project_id == project_id,
        MonitoringSummaryModel.window_start == window_start,
    )
    if artifact_id:
        q = q.filter(MonitoringSummaryModel.artifact_id == artifact_id)
    rows = q.order_by(MonitoringSummaryModel.artifact_id.asc()).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Bu pencere için özet yok")
    return {
        "window_timezone": "UTC",
        "windows": [_window_block(s, include_per_feature=True, include_distribution=True) for s in rows],
        "config": _config_block(),
    }


@router.get("/{project_id}/monitoring/deployments")
def monitoring_deployments(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Sürüm/champion paneli. Her artifact: window_count + total_predictions (indexli COUNT) + son pencere."""
    _require_project(db, project_id, current_user)
    champ = _serving_champion(db, project_id)

    artifacts = db.query(ArtifactModel).filter(
        ArtifactModel.project_id == project_id
    ).order_by(ArtifactModel.created_at.desc()).all()

    deployments = []
    for a in artifacts:
        total_pred = (db.query(func.count(PredictionModel.prediction_id))
                        .filter(PredictionModel.project_id == project_id,
                                PredictionModel.artifact_id == a.id).scalar()) or 0
        wq = (db.query(MonitoringSummaryModel)
                .filter(MonitoringSummaryModel.project_id == project_id,
                        MonitoringSummaryModel.artifact_id == a.id))
        window_count = wq.count()
        last = wq.order_by(MonitoringSummaryModel.window_start.desc()).first()
        last_window = None
        if last is not None:
            last_window = {
                "window_start": last.window_start, "status": last.status,
                "overall_status": last.overall_status,           # portfolio/triage rozeti (Faz 0)
                "overall_status_reason": last.overall_status_reason,
                "psi_max": last.psi_max, "cbpe_gate_status": last.cbpe_gate_status,
                "realized_gate_status": last.realized_gate_status,
                "label_coverage": last.label_coverage, "n_total": last.n_total,
            }
        # promotable: champion DEĞİL + gerçekten serve edilebilir (is_serviceable = load_for_serving üç
        # kapı, disk teyitli, FAIL-CLOSED) → "Bu sürüme geç" butonu YALNIZ bunda çıkar (kullanıcı butona
        # basıp 409 almaz). Champion zaten canlı → promotable=False. Non-complete/deployable satırlar disk
        # okumadan elenir (ucuz ön-eleme).
        promotable = bool((not a.is_champion) and a.phase == "complete" and a.deployable
                          and a.model_path and is_serviceable(a.id))
        deployments.append({
            "artifact_id": a.id, "model_name": a.model_name, "phase": a.phase,
            "deployable": a.deployable, "is_champion": a.is_champion,
            "promotable": promotable,
            "schema_version": a.schema_version,
            "created_at": a.created_at, "updated_at": a.updated_at,
            "window_count": window_count, "total_predictions": total_pred,
            "last_window": last_window,
        })

    champion = None
    if champ is not None:
        champion = {
            "artifact_id": champ.id, "model_name": champ.model_name, "phase": champ.phase,
            "deployable": champ.deployable, "schema_version": champ.schema_version,
            "created_at": champ.created_at, "updated_at": champ.updated_at,
        }
    return {
        "window_timezone": "UTC",
        "champion": champion,
        "deployments": deployments,
        "config": _config_block(),
    }


@router.get("/{project_id}/monitoring/latest")
def monitoring_latest(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Hero kartları — champion'ın EN SON penceresi (per_feature dahil). "Boş" hâller AYRIŞIR (disiplin:
    "temiz ≠ ölçülmedi") → dashboard her birine farklı mesaj:
      * ``not_trained``     — hiç serve edilebilir artifact yok → CTA "Model eğit".
      * ``not_promoted``    — serve edilebilir artifact var ama champion değil → CTA "Canlıya al" +
        ``promotable_artifact_id`` (backend belirler, doğrulanmış). Manuel promotion'ın normal hâli.
      * ``no_predictions``  — champion var ama endpoint hiç çağrılmamış (tahmin yok).
      * ``no_closed_window``— tahmin var ama kapanmış pencere yok (bugün başladı; aggregator yarım günü dışlar).
    """
    _require_project(db, project_id, current_user)
    champ = _serving_champion(db, project_id)
    if champ is None:
        nc = _no_champion_payload(db, project_id)      # not_trained | not_promoted (+promotable_artifact_id)
        return {"window_timezone": "UTC", "champion_artifact_id": None,
                "has_data": False, "config": _config_block(), **nc}

    pred_count = (db.query(func.count(PredictionModel.prediction_id))
                    .filter(PredictionModel.project_id == project_id,
                            PredictionModel.artifact_id == champ.id).scalar()) or 0
    if pred_count == 0:
        return {"window_timezone": "UTC", "champion_artifact_id": champ.id,
                "has_data": False, "reason": "no_predictions", "config": _config_block()}

    latest = (db.query(MonitoringSummaryModel)
                .options(undefer(MonitoringSummaryModel.psi_distribution))
                .filter(MonitoringSummaryModel.project_id == project_id,
                        MonitoringSummaryModel.artifact_id == champ.id)
                .order_by(MonitoringSummaryModel.window_start.desc()).first())
    if latest is None:
        return {"window_timezone": "UTC", "champion_artifact_id": champ.id,
                "has_data": False, "reason": "no_closed_window", "config": _config_block()}

    return {
        "window_timezone": "UTC",
        "champion_artifact_id": champ.id,
        "has_data": True,
        "reason": None,
        "window": _window_block(latest, include_per_feature=True, include_distribution=True),
        "config": _config_block(),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════
# FAZ 1 — ÜÇ YENİ ENDPOINT: schema (scoring playground formu) · predictions (paginated log) · portfolio
# ════════════════════════════════════════════════════════════════════════════════════════════════


def _schema_column(name: str, rec: dict, required_set: set, max_known: int) -> dict:
    """Bir ``raw_input_signature.columns`` kaydı → form-dostu kolon speci.

    KATEGORİK ``known_values`` iki eşikten geçer (kullanıcı eklemesi a): imza ZATEN ``_SIG_MAX_VOCAB=50``'ye
    cap'li (üstü None saklar), ama 50 değerli bir dropdown da kullanışsız → form için pratik eşik daha düşük
    (``max_known``, config). İki durumda da ``known_values=null`` + ``known_values_truncated=true`` döner →
    form dropdown yerine serbest metin + uyarıya geçer (form karar verir, backend dayatmaz).
      * imza None (yüksek-kardinalite >50)  → truncated=true (vocab hiç saklanmamış).
      * len > max_known (ör. 21..50)        → known_values=null, truncated=true (dropdown pratik-dışı).
      * len <= max_known                    → known_values dolu, truncated=false (dropdown kullanılabilir).
    """
    kind = rec.get("kind")                      # binary | categorical | continuous | other
    out = {
        "name": name,
        "domain": kind,
        "required": name in required_set,
        "dtype_family": rec.get("dtype_family"),
        "nunique": rec.get("nunique"),
        "null_ratio": rec.get("null_ratio"),
        "allowed_values": None,                 # binary: iki değer (STRICT — form abort)
        "known_values": None,                   # categorical: dropdown seçenekleri (varsa)
        "known_values_truncated": False,        # true → form serbest metin + uyarı (dropdown yerine)
        "range_hint": None,                     # continuous: [min, max] (warn-only)
    }
    if kind == "binary":
        out["allowed_values"] = rec.get("allowed_values")
    elif kind == "categorical":
        kv = rec.get("known_values")
        if kv is None:
            out["known_values_truncated"] = True                  # imza yüksek-kardinalite (>50) → vocab yok
        elif len(kv) > max_known:
            out["known_values_truncated"] = True                  # dropdown pratik-dışı → serbest metin
        else:
            out["known_values"] = kv
    elif kind == "continuous":
        out["range_hint"] = rec.get("range")
    elif kind == "datetime":
        # DATETIME: range_hint = gözlenen [min,max] tarih (ISO). parse_rate ile birlikte form
        # tarih girişi + parse/aralık uyarısı sürer (policy datetime="parse_and_range_warn").
        out["range_hint"] = rec.get("range")
        out["parse_rate"] = rec.get("parse_rate")
    return out


@router.get("/{project_id}/monitoring/schema")
def monitoring_schema(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Scoring playground FORMU — champion'ın ``raw_input_signature``'ından form-dostu kolon şeması.

    Kaynak = ``load_for_serving(champion).raw_input_signature`` (scoring endpoint ile AYNI kapı →
    champion yoksa/serve edilemezse 409, form scoring'in kabul edeceği ŞEKLİ birebir yansıtır).
    Kolon başına {name, domain, required, allowed_values/known_values(+truncated)/range_hint, null_ratio,
    nunique} + ``domain_check_policy`` (form: binary=abort, cat/cont=warn) + ``tolerated_extra_columns``.
    """
    _require_project(db, project_id, current_user)
    champ = _serving_champion(db, project_id)
    if champ is None:
        raise HTTPException(status_code=409,
                            detail="Bu proje için serve edilebilir champion yok (önce eğit / "
                                   "son koşu complete+deployable değil) — scoring formu üretilemez.")
    try:
        artifact = load_for_serving(champ.id)   # scoring ile AYNI üç kapı (complete∧deployable∧model-dosyası)
    except (ArtifactNotServiceable, ModelFileMissing) as e:
        raise HTTPException(status_code=409, detail=f"Artifact serve edilemez: {e}")

    sig = artifact.raw_input_signature or {}
    required = list(sig.get("required") or [])
    required_set = set(required)
    max_known = settings.MONITORING_SCHEMA_MAX_KNOWN_VALUES
    columns = [_schema_column(str(name), rec or {}, required_set, max_known)
               for name, rec in (sig.get("columns") or {}).items()]
    return {
        "window_timezone": "UTC",
        "champion_artifact_id": champ.id,
        "signature_version": sig.get("signature_version"),
        "columns": columns,
        "required": sorted(required),
        # form davranış politikası (binary=strict_domain_abort → "HasCrCard=0.7 GÖNDERİLMEDEN uyar/engelle";
        # categorical=unseen_allowed_counted → uyar ama geçir; continuous=range_warn_only).
        "domain_check_policy": sig.get("policy"),
        "tolerated_extra_columns": list(artifact.tolerated_extra_columns or []),
        "config": _config_block(),
    }


# ─────────────────────────── predictions log (keyset/cursor) ───────────────────────────

def _encode_cursor(created_at, prediction_id: str) -> str:
    """``(created_at, prediction_id)`` bileşik keyset imleci → opak base64 token.

    JSON round-trip TAM temsil: int için birebir, float için de ``json.dumps`` en-kısa-round-trip repr'ı
    kullanır (`json.loads(json.dumps(x)) == x` her float'ta) → **precision kaybı YOK** (kullanıcı eklemesi:
    created_at float olsa bile base64'e tam yaz). NOT: bugün ``predictions.created_at`` INT (epoch saniye) —
    float olan alanlar ``label_backfilled_at``/``computed_at``; yine de kodlama her iki tipte de tam.
    Bileşik anahtar ŞART: aynı saniyede çok tahmin olabilir → created_at tek başına tie-break edemez;
    prediction_id (uuid) toplam-sıra sağlar (head-insert offset'i kaydırmaz)."""
    payload = json.dumps({"c": created_at, "p": prediction_id},
                         separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str):
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        return payload["c"], payload["p"]
    except Exception:
        raise HTTPException(status_code=400, detail="geçersiz cursor (base64/JSON çözülemedi)")


def _redact(row):
    """PII redaction KANCASI — v1 IDENTITY (maskeleme YOK, ``input_row_redacted=false`` ile eşleşir).

    Satır-bazlı (top-level blok DEĞİL) → ileride kolon/değer maskeleme BURAYA girer, yanıt ŞEMASI
    (``input_row``/``input_row_redacted`` alanları) değişmez → dashboard sözleşmesi kırılmaz."""
    return row


def _prediction_row(r) -> dict:
    """Bir ``PredictionModel`` satırı → log satırı. ``input_row`` yalnız örneklem satırlarda dolu
    (``input_row_sampled`` bayrakla gösterir); ``_redact`` v1-identity + ``input_row_redacted=false``."""
    has_row = r.input_row is not None
    return {
        "prediction_id": r.prediction_id,
        "created_at": r.created_at,
        "artifact_id": r.artifact_id,
        "prediction": r.prediction,
        "probability": r.probability,
        "threshold": r.threshold,
        "input_row": (_redact(r.input_row) if has_row else None),
        "input_row_sampled": r.input_row_sampled,
        "input_row_redacted": False,            # v1 identity redaction (KANCA; maskeleme gelince true olur)
        "input_warnings": r.input_warnings,
        "input_warning_flags": r.input_warning_flags,
        "actual_label": r.actual_label,
        "label_backfilled_at": r.label_backfilled_at,
        "business_key": r.business_key,
    }


@router.get("/{project_id}/monitoring/predictions")
def monitoring_predictions(
    project_id: int,
    cursor: str = Query(default=None),
    limit: int = Query(default=None),
    date_from: int = Query(default=None),       # epoch saniye (created_at >=)
    date_to: int = Query(default=None),         # epoch saniye (created_at <=)
    artifact_id: str = Query(default=None),
    has_label: bool = Query(default=None),
    has_warning: bool = Query(default=None),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Paginated tahmin log'u — **KEYSET/cursor** (offset DEĞİL). ``predictions`` TEK büyüyen tablo ve
    yeni satırlar başa girer (created_at desc) → offset'te sayfa 2'ye geçerken yeni tahmin gelirse satır
    SESSİZCE atlanır (kullanıcı log'da eksiği fark etmez). Bileşik ``(created_at, prediction_id)`` keyset
    bunu yapısal olarak kapatır: sınır satırdan KÜÇÜK olanı çeker, head-insert sayfayı kaydırmaz.

    Filtreler: date_from/to (created_at aralığı) · artifact_id · has_label (actual_label var/yok) ·
    has_warning (per-row bayrak has_unseen∨has_out_of_range). Sıralama created_at DESC, prediction_id DESC.
    """
    _require_project(db, project_id, current_user)
    default_n = settings.MONITORING_PREDICTIONS_PAGE_SIZE
    max_n = settings.MONITORING_PREDICTIONS_PAGE_MAX
    eff_n = default_n if limit is None else limit
    eff_n = max(1, min(int(eff_n), max_n))                 # [1, MAX]'a kırp (hata değil, days deseni)

    q = db.query(PredictionModel).filter(PredictionModel.project_id == project_id)
    if artifact_id:
        q = q.filter(PredictionModel.artifact_id == artifact_id)
    if date_from is not None:
        q = q.filter(PredictionModel.created_at >= date_from)
    if date_to is not None:
        q = q.filter(PredictionModel.created_at <= date_to)
    if has_label is not None:
        # actual_label GERÇEK SQL NULL'a dayanır (scoring omit + labels endpoint None-ayıklar → JSON 'null'
        # asla yazılmaz) → IS [NOT] NULL güvenilir.
        q = q.filter(PredictionModel.actual_label.isnot(None) if has_label
                     else PredictionModel.actual_label.is_(None))
    if has_warning is not None:
        flags = PredictionModel.input_warning_flags
        # per-row bayrak; ``.is_(True)`` null-güvenli (NULL flags → False, üç-değerli mantık yok) → ~P iyi tanımlı.
        warn = or_(flags["has_unseen"].as_boolean().is_(True),
                   flags["has_out_of_range"].as_boolean().is_(True))
        q = q.filter(warn if has_warning else not_(warn))
    if cursor:
        c, p = _decode_cursor(cursor)
        # DESC keyset: sınırdan KESİN küçük satırlar (created_at daha eski, ya da aynı saniyede id daha küçük).
        q = q.filter(or_(PredictionModel.created_at < c,
                         and_(PredictionModel.created_at == c,
                              PredictionModel.prediction_id < p)))

    q = q.order_by(PredictionModel.created_at.desc(), PredictionModel.prediction_id.desc())
    rows = q.limit(eff_n + 1).all()                        # +1 → has_more tespiti (fazla satır çekilmez)
    has_more = len(rows) > eff_n
    rows = rows[:eff_n]
    next_cursor = (_encode_cursor(rows[-1].created_at, rows[-1].prediction_id)
                   if (rows and has_more) else None)
    return {
        "window_timezone": "UTC",
        "rows": [_prediction_row(r) for r in rows],
        "next_cursor": next_cursor,
        "has_more": has_more,
        "limit_applied": eff_n,
        "config": _config_block(),
    }


# ─────────────────────────── portfolio (cross-project, AYRI router) ───────────────────────────

@portfolio_router.get("/portfolio")
def monitoring_portfolio(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Cross-project özet — kullanıcının HER projesi için bir kart (champion son penceresi).

    🔴 SAHİPLİK YAPISAL: her sorgu ``owned_ids`` (yalnız ``user_id==current_user``) ile çitli → başka
    kullanıcının projesi HİÇBİR dalda (champion/window/prediction join'i) yanıta giremez (negatif-testli).

    N+1 YOK — sabit sayıda toplu sorgu (owned → champions → son-pencere ROW_NUMBER → prediction COUNT),
    hepsi ``owned_ids``'e IN-filtreli. Kart HAFİF: ``overall_status`` + reason **{driver, code}'e KIRPILI**
    (kullanıcı eklemesi b: portfolio 20 proje × tam reason gereksiz yük; value bloğu — feature adı/psi_max/
    suppressed_axes — detay/latest endpoint'inde) + psi_max ayrı + window_start + n_total + total_predictions.
    per_feature/distribution/scalars YÜKLENMEZ (yalnız hafif kolonlar seçilir). Üç boş-reason ayrışır."""
    owned = db.query(ProjectModel.id, ProjectModel.title).filter(
        ProjectModel.user_id == current_user.id).all()
    owned_ids = [p.id for p in owned]
    if not owned_ids:
        return {"window_timezone": "UTC", "projects": [], "config": _config_block()}

    # champions (owned-only) → {project_id: artifact}
    champs = db.query(ArtifactModel).filter(
        ArtifactModel.project_id.in_(owned_ids),
        ArtifactModel.is_champion.is_(True),
        ArtifactModel.deployable.is_(True),
        ArtifactModel.phase == "complete",
    ).all()
    champ_by_project = {a.project_id: a for a in champs}
    champ_ids = [a.id for a in champs]                      # artifact_id uuid → globalde tekil

    # not_trained ↔ not_promoted ayrımı (manuel promotion): champion'ı OLMAYAN projelerde serve
    # edilebilir (complete+deployable+model_path) artifact VAR mı? TEK toplu sorgu (N+1 yok, owned-çitli).
    # ⚠️ DB-seviyesi sinyal (disk load_for_serving DEĞİL) — portfolio HAFİF kalsın; kesin doğrulama
    # (model dosyası fiilen var mı) DETAY sayfasında, promote butonunun olduğu yerde yapılır.
    championless_ids = [pid for pid in owned_ids if pid not in champ_by_project]
    promotable_project_ids: set = set()
    if championless_ids:
        for (pid,) in (db.query(ArtifactModel.project_id)
                         .filter(ArtifactModel.project_id.in_(championless_ids),
                                 ArtifactModel.phase == "complete",
                                 ArtifactModel.deployable.is_(True),
                                 ArtifactModel.model_path.isnot(None))
                         .distinct().all()):
            promotable_project_ids.add(pid)

    # son pencere / champion — TEK sorgu (ROW_NUMBER), YALNIZ hafif kolonlar (per_feature/distribution DEĞİL).
    latest_by_project: dict = {}
    if champ_ids:
        rn = func.row_number().over(
            partition_by=[MonitoringSummaryModel.project_id, MonitoringSummaryModel.artifact_id],
            order_by=MonitoringSummaryModel.window_start.desc()).label("rn")
        sub = db.query(
            MonitoringSummaryModel.project_id.label("project_id"),
            MonitoringSummaryModel.artifact_id.label("artifact_id"),
            MonitoringSummaryModel.window_start.label("window_start"),
            MonitoringSummaryModel.computed_at.label("computed_at"),   # header "son güncelleme" (tazelik)
            MonitoringSummaryModel.status.label("status"),
            MonitoringSummaryModel.overall_status.label("overall_status"),
            MonitoringSummaryModel.overall_status_reason.label("overall_status_reason"),
            MonitoringSummaryModel.psi_max.label("psi_max"),
            MonitoringSummaryModel.n_total.label("n_total"),
            MonitoringSummaryModel.cbpe_gate_status.label("cbpe_gate_status"),
            MonitoringSummaryModel.realized_gate_status.label("realized_gate_status"),
            MonitoringSummaryModel.label_coverage.label("label_coverage"),
            rn,
        ).filter(
            MonitoringSummaryModel.project_id.in_(owned_ids),      # SAHİPLİK çiti (join her dalda owned)
            MonitoringSummaryModel.artifact_id.in_(champ_ids),
        ).subquery()
        for r in db.query(sub).filter(sub.c.rn == 1).all():
            latest_by_project[r.project_id] = r

    # total_predictions / champion — TEK grouped COUNT (owned-only).
    pred_counts: dict = {}
    if champ_ids:
        for aid, cnt in (db.query(PredictionModel.artifact_id, func.count(PredictionModel.prediction_id))
                           .filter(PredictionModel.project_id.in_(owned_ids),
                                   PredictionModel.artifact_id.in_(champ_ids))
                           .group_by(PredictionModel.artifact_id).all()):
            pred_counts[aid] = cnt

    projects_out = []
    for p in owned:
        champ = champ_by_project.get(p.id)
        base = {"project_id": p.id, "title": p.title}
        if champ is None:                                  # champion yok → not_trained | not_promoted
            reason = "not_promoted" if p.id in promotable_project_ids else "not_trained"
            projects_out.append({**base, "champion_artifact_id": None,
                                 "has_data": False, "reason": reason, "card": None})
            continue
        total_pred = pred_counts.get(champ.id, 0)
        base["champion_artifact_id"] = champ.id
        if total_pred == 0:                                # boş-reason 2: champion var, tahmin yok
            projects_out.append({**base, "has_data": False, "reason": "no_predictions", "card": None})
            continue
        latest = latest_by_project.get(p.id)
        if latest is None:                                 # boş-reason 3: tahmin var, kapanmış pencere yok
            projects_out.append({**base, "has_data": False, "reason": "no_closed_window", "card": None})
            continue
        reason = latest.overall_status_reason or {}
        projects_out.append({
            **base, "has_data": True, "reason": None,
            "card": {
                "overall_status": latest.overall_status,
                # KIRPILI reason (kullanıcı eklemesi b): kart-düzeyi için driver+code yeter; value bloğu detayda.
                "reason": {"driver": reason.get("driver"), "code": reason.get("code")},
                "psi_max": latest.psi_max,                 # kart psi_max'ı AYRI taşır (reason'da tekrar etmez)
                "window_start": latest.window_start,
                "computed_at": latest.computed_at,         # dashboard header "son güncelleme" (tazelik)
                "n_total": latest.n_total,
                "total_predictions": total_pred,
                "status": latest.status,
                "cbpe_gate_status": latest.cbpe_gate_status,
                "realized_gate_status": latest.realized_gate_status,
                "label_coverage": latest.label_coverage,
            },
        })
    return {"window_timezone": "UTC", "projects": projects_out, "config": _config_block()}
