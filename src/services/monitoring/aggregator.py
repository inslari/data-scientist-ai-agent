"""Katman 2 Parça 2 — monitoring aggregator İSKELETİ (metrik hesabı YOK).

Amaç: pencereleme (UTC günlük tumbling, YALNIZ kapanmış gün) + champion-bölme
(`predictions.artifact_id`) + idempotanslık (dirty-kontrol) mantığını METRİK
karmaşıklığı olmadan doğrulamak. Çekirdek alanları (n_total/n_sampled/status/
window_*/computed_at) yazar; psi_*/cbpe_*/quality_* NULL bırakır → Parça 3/4
yalnız alan doldurur, ŞEMA değiştirmez.

`db_maintenance.run_maintenance()` alt-görevi olarak koşar → per-subtask try/except
ile İZOLE (patlarsa GC/checkpoint/log budaması etkilenmez).
"""
from __future__ import annotations

import time

from sqlalchemy import and_, case, func

from api.core.database import SessionLocal
from api.models.monitoring_summary import MonitoringSummaryModel
from api.models.prediction import PredictionModel
from src.core.config import settings
from src.core.logging_config import get_logger
from src.services.monitoring.cbpe import estimate_performance
from src.services.monitoring.provenance import attribute_importance
from src.services.monitoring.psi import compute_psi_summary
from src.services.monitoring.realized import compute_realized_metrics
from src.services.monitoring.triage import compute_overall_status

logger = get_logger(__name__)

_DAY = 86_400

# KAYNAK FİLTRESİ (Katman 2): dashboard Scoring playground'ın MANUEL denemeleri (source="ui_manual")
# monitoring'i KİRLETMEZ → tüm metrik sorguları bu filtreyle koşar (PSI/kalite/CBPE/gerçekleşen).
# ``coalesce(source,'api')``: alan-öncesi NULL kayıt "api" (üretim) sayılır → körleşme yok, geçmiş
# üretim tahminleri dışlanmaz. ``_IS_UI`` simetrik (source == 'ui_manual'; NULL → sayılmaz) → dışlanan sayacı.
_NOT_UI = func.coalesce(PredictionModel.source, "api") != "ui_manual"
_IS_UI = PredictionModel.source == "ui_manual"


def _day_floor(col):
    """created_at (INT epoch) → UTC gün başına AŞAĞI yuvarla. Modulo ile (bölme/cast YOK) →
    SQLite ve Postgres'te AYNI (floor). ⚠️ Eski ``cast(col / _DAY, Integer) * _DAY`` Postgres'te
    float-bölme + cast YUVARLIYORDU (öğleden sonraki tahminleri sonraki güne itip pencereyi bozuyordu);
    SQLite floor verdiğinden testler yeşildi. Bkz. test_day_floor_uses_modulo_not_rounding_cast."""
    return col - (col % _DAY)

_EMPTY_PSI = {"psi_overall": None, "psi_max": None, "psi_degenerate_count": None,
              "degenerate_importance_mass": None, "psi_per_feature": None,
              "importance_attributed_ratio": None, "psi_scalars": None, "psi_distribution": None}


def _load_reference(artifact_id: str, cache: dict):
    """Artifact (per_feature dolu) döner; yok / training_reference boş / per_feature boş → None.

    None = no_reference kaynağı: retention GC champion'ı silmiş VEYA eski/yarım artifact (per_feature boş)."""
    if artifact_id in cache:
        return cache[artifact_id]
    art = None
    try:
        from src.services.inference.serialize import load_artifact
        a = load_artifact(artifact_id)
        if (a.training_reference or {}).get("per_feature"):
            art = a
    except Exception as e:
        logger.info(f"[monitoring] artifact referansı yüklenemedi (no_reference) {artifact_id}: "
                    f"{type(e).__name__}: {str(e)[:120]}")
    cache[artifact_id] = art
    return art


def _fetch_prod_rows(db, project_id, artifact_id, window_start, limit) -> list:
    """Pencerenin örneklem input_row'ları (PSI için). input_row_sampled=True → JSON dolu. LIMIT=cap."""
    rows = (db.query(PredictionModel.input_row)
              .filter(PredictionModel.project_id == project_id,
                      PredictionModel.artifact_id == artifact_id,
                      PredictionModel.created_at >= window_start,
                      PredictionModel.created_at < window_start + _DAY,
                      _NOT_UI,
                      PredictionModel.input_row_sampled.is_(True))
              .limit(limit).all())
    return [r[0] for r in rows if isinstance(r[0], dict)]


def _compute_quality(db, project_id, artifact_id, window_start) -> "dict | None":
    """Veri kalitesi — per-row `input_warning_flags`'ten (TÜM satırlar, PSI referansından BAĞIMSIZ).

    Sayaç HER ZAMAN (alarm n≥MIN_N_QUALITY, dashboard/config uygular). Batch-agregatı input_warnings
    KULLANILMAZ (pencerede toplanamaz). Bayrağı olmayan satır (eski/fail-safe None) sayıma girmez;
    hiç yoksa None. ⚠️ ÖLÇEK: satır-başı JSON python'da sayılıyor (hacim sıfırken kabul); büyüyünce
    portable SQL JSON-extract'a taşınır (SQLite json_extract / Postgres ->>)."""
    rows = (db.query(PredictionModel.input_warning_flags)
              .filter(PredictionModel.project_id == project_id,
                      PredictionModel.artifact_id == artifact_id,
                      PredictionModel.created_at >= window_start,
                      PredictionModel.created_at < window_start + _DAY,
                      _NOT_UI,
                      PredictionModel.input_warning_flags.isnot(None))
              .all())
    flags = [r[0] for r in rows if isinstance(r[0], dict)]
    nq = len(flags)
    if nq == 0:
        return None
    unseen = sum(1 for f in flags if f.get("has_unseen"))
    oor = sum(1 for f in flags if f.get("has_out_of_range"))
    return {
        "unseen": {"count": unseen, "rate": round(unseen / nq, 6)},
        "out_of_range": {"count": oor, "rate": round(oor / nq, 6)},
        "n": nq,
        "alarm_eligible": nq >= settings.MONITORING_MIN_N_QUALITY,
    }


def _compute_cbpe(db, art, project_id, artifact_id, window_start, n_total):
    """CBPE (tahmini performans) — per-row `probability`'den, GERÇEKLEŞEN etiket olmadan.

    KAPI (cbpe_gate_status): kalibrasyon şart (Faz C `is_well_calibrated`). Bastırma → cbpe_estimate
    NULL (güvenilmez sayı saklama → dashboard kazara göstermesin). Regresyon/multiclass → calibration
    None → suppressed_no_baseline. CONCEPT DRIFT altında İYİMSER sapar (bkz. cbpe.py) → "assumes" damgası."""
    if art is None:
        return "suppressed_no_baseline", None
    baseline = (art.training_reference or {}).get("baseline_metrics") or {}
    cal = baseline.get("calibration")
    if not baseline or cal is None:
        return "suppressed_no_baseline", None                  # eski artifact / regresyon / multiclass
    if not cal.get("is_well_calibrated"):
        return "suppressed_calibration", None                  # kalibre değil → CBPE geçersiz
    if n_total < settings.MONITORING_MIN_N_CBPE:
        return "insufficient_n", None
    rows = (db.query(PredictionModel.probability)
              .filter(PredictionModel.project_id == project_id,
                      PredictionModel.artifact_id == artifact_id,
                      PredictionModel.created_at >= window_start,
                      PredictionModel.created_at < window_start + _DAY,
                      _NOT_UI)
              .all())
    probs = [r[0] for r in rows if isinstance(r[0], (int, float))]   # binary float (multiclass dict düşer)
    if len(probs) < settings.MONITORING_MIN_N_CBPE:
        return "insufficient_n", None
    threshold = (art.model_ref or {}).get("optimal_threshold") or 0.5
    est = estimate_performance(probs, threshold)
    return "ok", {
        "estimated_roc_auc": (round(est["roc_auc"], 6) if est["roc_auc"] is not None else None),
        "estimated_accuracy": round(est["accuracy"], 6),
        "baseline_roc_auc": baseline.get("roc_auc"),
        "baseline_accuracy": baseline.get("accuracy"),
        # ⚠️ concept drift altında İYİMSER sapar → dashboard "tahmini" damgası bu alandan
        "assumes": "covariate_drift_only",
    }


def _infer_task(probs: list) -> str:
    """Pencerenin probability tiplerinden görevi çıkar: dict→multiclass, sayı→binary, hiçbiri→regression.

    Scoring: binary→float, multiclass→dict, regresyon→None yazar. İlk None-olmayan belirler."""
    for p in probs:
        if isinstance(p, dict):
            return "multiclass"
        if isinstance(p, (int, float)):
            return "binary"
    return "regression"


def _compute_realized(db, art, project_id, artifact_id, window_start, n_total):
    """GERÇEKLEŞEN performans — ``actual_label`` DOLU satırlardan (CBPE'nin doğrulaması).

    Döner: ``(gate, metrics|None, label_coverage)``. ``label_coverage`` HER ZAMAN dolu (0 dahil,
    min-n altında dahil) → dashboard "etiket kapsamı %N" satırı realized NULL iken bile render eder.
    GATE (kalibrasyon değil, ETİKET-yeterliliği): no_labels (0) → insufficient_labels (<min) → ok.
    Not: pozitif-sınıf yoksa roc_auc None ama accuracy geçerli → gate YİNE "ok" (status yalan söyleyemez;
    "suppressed_no_positive_class" accuracy'nin varlığını yalanlardı → 3 değerde toplandı, roc_auc içeride None).

    ⚠️ INVARIANT: ``actual_label.isnot(None)`` GERÇEK SQL NULL'a dayanır. Scoring bu alanı HİÇ yazmaz
    (omit → SQL NULL) ve labels endpoint None etiketi AYIKLAR → JSON 'null' asla yazılmaz. Açıkça None
    bind edilseydi none_as_null=False yüzünden JSON 'null' saklanır ve isnot(None) tüm satırları etiketli
    sayardı (n_sampled'ın input_row_sampled boolean'ıyla çözdüğü gotcha'nın ikizi)."""
    labeled_n_rows = (db.query(PredictionModel.prediction,
                               PredictionModel.actual_label,
                               PredictionModel.probability)
                        .filter(PredictionModel.project_id == project_id,
                                PredictionModel.artifact_id == artifact_id,
                                PredictionModel.created_at >= window_start,
                                PredictionModel.created_at < window_start + _DAY,
                                _NOT_UI,
                                PredictionModel.actual_label.isnot(None))
                        .all())
    labeled_n = len(labeled_n_rows)
    coverage = round(labeled_n / n_total, 6) if n_total else 0.0
    if labeled_n == 0:
        return "no_labels", None, 0.0
    if labeled_n < settings.MONITORING_MIN_N_REALIZED:
        return "insufficient_labels", None, coverage    # etiket az → metrik gürültülü (label_coverage yine görünür)

    preds = [r[0] for r in labeled_n_rows]
    actuals = [r[1] for r in labeled_n_rows]
    probs = [r[2] for r in labeled_n_rows]
    baseline = (art.training_reference or {}).get("baseline_metrics") if art is not None else None
    positive_label = (baseline or {}).get("positive_label")
    task = _infer_task(probs)
    metrics = compute_realized_metrics(task, preds, actuals, probs, positive_label, baseline)
    return "ok", metrics, coverage


def run_monitoring_aggregation() -> int:
    """Kapanmış UTC günlerinde (project, artifact) gruplarına özet yazar/günceller (PSI dahil).

    Döner: yazılan/güncellenen özet satırı sayısı (atlananlar sayılmaz).
    """
    now = time.time()                                # FLOAT epoch (computed_at çapası; 1-sn yarışı kapalı)
    latest_closed_day_end = (int(now) // _DAY) * _DAY     # bugünkü (yarım) gün DIŞLANIR
    written = 0

    db = SessionLocal()
    try:
        # window_start = UTC gün başı epoch = created_at'i gün sınırına AŞAĞI yuvarla.
        # ⚠️ BUG (2026-08-06, gerçek-veri populate turunda bulundu): `cast(created_at / _DAY, Integer)`
        # Postgres'te FLOAT bölme + cast YUVARLAR (floor DEĞİL) → öğleden sonraki (gün-kesri ≥0.5)
        # tahminler bir SONRAKİ güne bucket'lanıyordu; window_start alt-sorguların (`created_at >=
        # window_start`) ALTINA düşen satırları dışlıyor → PSI boş-referansa karşı saçma yüksek,
        # realized/quality boş. SQLite int/int→floor verdiğinden testler yeşildi (SQLite/PG sapması).
        # DÜZELTME: modulo ile floor (bkz. _day_floor) — bölme/cast yok, İKİ dial'de de aynı.
        bucket = _day_floor(PredictionModel.created_at)

        # ⚠️ TAM-TARAMA: tek filtre "kapanmış gün" (created_at < latest_closed_day_end); ALT-SINIR
        # YOK → her koşu TÜM geçmiş kapanmış günleri grup olarak getirir. Hacim sıfırken kabul
        # (YAGNI). Büyüyünce buraya `MONITORING_LOOKBACK_DAYS` alt-sınırı eklenir (son K günü tara);
        # idempotent UPSERT + dirty-kontrol bununla uyumlu — nereye bakılacağı burası.
        groups = (
            db.query(
                PredictionModel.project_id.label("project_id"),
                PredictionModel.artifact_id.label("artifact_id"),
                bucket.label("window_start"),
                # n_total = ÜRETİM satırları (ui_manual HARİÇ). Koşullu toplam: aynı grup içinde hem
                # üretim hem manuel satır olabilir → count(*) yerine _NOT_UI-gated sum (ui'ler ayrı sayılır).
                func.sum(case((_NOT_UI, 1), else_=0)).label("n_total"),
                # n_sampled = input_row örneklem ÜRETİM satırları. `input_row_sampled` boolean'ı KANIT —
                # `count(input_row)` GÜVENİLMEZ: JSON tipi Python None'ı SQL NULL değil JSON 'null'
                # saklar (none_as_null=False) → non-sampled satırları da sayardı. _NOT_UI de kapılı.
                func.sum(case((and_(_NOT_UI, PredictionModel.input_row_sampled.is_(True)), 1),
                              else_=0)).label("n_sampled"),
                # DIŞLANAN manuel deneme sayısı (first-class görünürlük — sessizce yutulmaz).
                func.sum(case((_IS_UI, 1), else_=0)).label("excluded_ui_count"),
                func.max(func.coalesce(PredictionModel.label_backfilled_at,
                                       PredictionModel.created_at)).label("max_change"),
            )
            .filter(PredictionModel.created_at < latest_closed_day_end)
            .group_by(PredictionModel.project_id, PredictionModel.artifact_id, bucket)
            .all()
        )

        alpha = settings.MONITORING_PSI_LAPLACE_ALPHA
        ref_cache: dict = {}
        for g in groups:
            window_start = int(g.window_start)
            # PK sırası = (project_id, artifact_id, window_start)
            existing = db.get(MonitoringSummaryModel,
                              (g.project_id, g.artifact_id, window_start))

            # DIRTY-KONTROL: veri son hesaptan SONRA değişti mi? Geç `actual_label` backfill'i eski
            # pencereyi kirletir (label_backfilled_at > computed_at) → yeniden hesap. Parça 2'de
            # backfill YOK → hep inert: zaten-güncel grup ATLANIR, computed_at KORUNUR (= "son
            # işlenenden devam"; monitoring_summary'nin kendisi durum, ayrı watermark yok).
            # FLOAT karşılaştırma (int() YOK): label_backfilled_at + computed_at float → geç etiketin
            # 1-saniye sınır yarışı kapalı (run-başı now < sonra-commit'lenen backfilled_at).
            if existing is not None and g.max_change is not None \
                    and float(g.max_change) <= float(existing.computed_at):
                continue

            n_total = int(g.n_total or 0)
            n_sampled = int(g.n_sampled or 0)
            excluded_ui_count = int(g.excluded_ui_count or 0)
            # Pencerede YALNIZ ui_manual (playground) varsa izlenecek üretim trafiği yok → özet YAZMA
            # (sahte boş satır üretme). Dışlanan sayacını taşıyacak üretim penceresi zaten yok.
            if n_total == 0:
                continue

            # STATÜS SIRASI: önce REFERANS (no_reference reddi n'den bağımsız), sonra n. PSI input_row
            # DOLU satırları kullanır → n_SAMPLED eşiği (n_total değil; p<1'de ayrışır).
            art = _load_reference(g.artifact_id, ref_cache)
            psi = dict(_EMPTY_PSI)
            if art is None:
                status = "no_reference"                       # referans yok → PSI hesaplanamaz (≠ yetersiz n)
            elif n_sampled < settings.MONITORING_MIN_N_PSI:
                status = "insufficient_n"                     # referans var ama örneklem az → sayı üretme
            else:
                status = "ok"
                tref = art.training_reference or {}
                # 3b: importance provenance PSI-anında manifest'ten (train'de yazılmaz; tek doğruluk kaynağı
                # manifest). importance yoksa raw_imp={} → PSI düz ortalamaya düşer (fallback).
                raw_imp, attributed_ratio = {}, None
                fi = tref.get("feature_importance")
                if fi:
                    raw_imp, attributed_ratio = attribute_importance(
                        fi, art.manifest, (art.raw_input_signature or {}).get("required"), art.param_store)
                prod_rows = _fetch_prod_rows(db, g.project_id, g.artifact_id, window_start,
                                             settings.SERVING_INPUT_ROW_SAMPLE_TARGET)
                psi = compute_psi_summary(tref.get("per_feature"), prod_rows, tref.get("n_train") or 0,
                                          alpha, raw_importance=raw_imp)
                psi["importance_attributed_ratio"] = attributed_ratio

            # VERİ KALİTESİ — PSI durumundan BAĞIMSIZ (referans gerekmez, per-row bayraktan). Her grupta hesaplanır.
            quality = _compute_quality(db, g.project_id, g.artifact_id, window_start)
            # CBPE — per-row probability'den; kendi kapısı (kalibrasyon), PSI durumundan bağımsız.
            cbpe_gate, cbpe_est = _compute_cbpe(db, art, g.project_id, g.artifact_id, window_start, n_total)
            # GERÇEKLEŞEN — actual_label DOLU satırlardan (backfill'e bağlı; yoksa no_labels). PSI/CBPE'den
            # bağımsız. Geç etiket bu pencereyi kirletince (label_backfilled_at>computed_at) tüm satır
            # baştan hesaplanır — PSI/CBPE donmuş pencerede BİREBİR aynıya çıkar (israf ama zararsız, basit).
            realized_gate, realized_metrics, label_coverage = _compute_realized(
                db, art, g.project_id, g.artifact_id, window_start, n_total)

            # ── TRIAGE (Faz 0): dört monitörü tek rozete indir. INVARIANT: realized AYNI PENCEREDE
            # ölçüldü (actual_label o pencerenin tahminlerine yazılır) → çapraz-uzlaşma güvenli.
            overall_status, overall_reason = compute_overall_status(
                psi_status=status, psi_max=psi["psi_max"],
                importance_attributed_ratio=psi["importance_attributed_ratio"],
                degenerate_importance_mass=psi["degenerate_importance_mass"],
                cbpe_gate_status=cbpe_gate, cbpe_estimate=cbpe_est,
                realized_gate_status=realized_gate, realized_metrics=realized_metrics,
                label_coverage=label_coverage, quality_rates=quality, settings=settings)

            if existing is None:
                db.add(MonitoringSummaryModel(
                    project_id=g.project_id, artifact_id=g.artifact_id,
                    window_start=window_start, window_end=window_start + _DAY,
                    window_kind="daily", computed_at=now,
                    n_total=n_total, n_sampled=n_sampled,
                    excluded_ui_count=excluded_ui_count, status=status,
                    psi_overall=psi["psi_overall"], psi_max=psi["psi_max"],
                    psi_degenerate_count=psi["psi_degenerate_count"],
                    degenerate_importance_mass=psi["degenerate_importance_mass"],
                    importance_attributed_ratio=psi["importance_attributed_ratio"],
                    psi_per_feature=psi["psi_per_feature"],
                    psi_scalars=psi["psi_scalars"], psi_distribution=psi["psi_distribution"],
                    quality_rates=quality,
                    cbpe_estimate=cbpe_est, cbpe_gate_status=cbpe_gate,
                    label_coverage=label_coverage,
                    realized_metrics=realized_metrics, realized_gate_status=realized_gate,
                    overall_status=overall_status, overall_status_reason=overall_reason,
                ))
            else:
                existing.window_end = window_start + _DAY
                existing.n_total = n_total
                existing.n_sampled = n_sampled
                existing.excluded_ui_count = excluded_ui_count
                existing.status = status
                existing.computed_at = now
                existing.psi_overall = psi["psi_overall"]
                existing.psi_max = psi["psi_max"]
                existing.psi_degenerate_count = psi["psi_degenerate_count"]
                existing.degenerate_importance_mass = psi["degenerate_importance_mass"]
                existing.importance_attributed_ratio = psi["importance_attributed_ratio"]
                existing.psi_per_feature = psi["psi_per_feature"]
                existing.psi_scalars = psi["psi_scalars"]
                existing.psi_distribution = psi["psi_distribution"]
                existing.quality_rates = quality
                existing.cbpe_estimate = cbpe_est
                existing.cbpe_gate_status = cbpe_gate
                existing.label_coverage = label_coverage
                existing.realized_metrics = realized_metrics
                existing.realized_gate_status = realized_gate
                existing.overall_status = overall_status
                existing.overall_status_reason = overall_reason
            written += 1

        db.commit()
    finally:
        db.close()

    logger.info(f"[monitoring] aggregate: {written} özet satırı yazıldı/güncellendi "
                f"(kapanmış gün < {latest_closed_day_end}).")
    return written
