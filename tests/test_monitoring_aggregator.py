"""Katman 2 Parça 2+3 — monitoring aggregator (pencereleme + PSI).

Doğrular: UTC günlük tumbling + yalnız-kapanmış-gün + champion-bölme (aynı günde 2 artifact →
2 satır) + n_total/n_sampled ayrımı + status ÜÇ değer (ok/insufficient_n/no_reference) + PSI alanları
(ok→dolu, aksi→null) + idempotanslık (yeniden koş → atlanır) + dirty-kontrol (label_backfilled_at >
computed_at → yeniden hesap).

Sentetik predictions doğrudan tabloya; created_at kontrollü. Referans = diske yazılan STUB artifact.json.
"""
import json
import os
import shutil
import uuid

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_monitoring_test.sqlite")

import pytest

from api.core.database import Base, SessionLocal, engine
from api.models.user import UserModel
from api.models.project import ProjectModel
from api.models.prediction import PredictionModel
from api.models.monitoring_summary import MonitoringSummaryModel

from src.services.monitoring.aggregator import run_monitoring_aggregation

_DAY = 86_400
# "f" için dejenere-OLMAYAN sürekli referans (5 distinct quantile) → PSI ölçülür, degenerate_count=0
_PF = {"f": {"kind": "continuous", "mean": 175.0, "std": 100.0,
             "quantiles": {"0.01": 5.0, "0.25": 90.0, "0.5": 175.0, "0.75": 260.0, "0.99": 345.0},
             "n": 1000}}


def _write_stub_artifact(aid):
    d = os.path.join("data", "artifacts", aid)
    os.makedirs(d, exist_ok=True)
    # feature_importance + raw_input_signature → 3b yolu (importance-ağırlık) tetiklenir. "f" raw
    # (identity) → attribute_importance %100 atfeder.
    json.dump({"artifact_schema_version": "1.2", "param_store": {}, "manifest": [],
               "raw_input_signature": {"required": ["f"], "columns": {}},
               "model_ref": {"optimal_threshold": 0.5, "model_name": "x"},
               "training_reference": {"per_feature": _PF, "n_train": 1000,
                                      "feature_importance": {"f": 2.0},
                                      "baseline_metrics": {"roc_auc": 0.86, "accuracy": 0.80,
                                          "positive_label": 1, "positive_rate": 0.30,
                                          "calibration": {"is_well_calibrated": True,
                                                          "brier_score": 0.05, "brier_baseline": 0.2}}}},
              open(os.path.join(d, "artifact.json"), "w"))


def _rows(project_id, artifact_id, day_start, n, n_sampled, unseen_every=0,
          labeled=0, pos_of_labeled=0):
    out = []
    for i in range(n):
        sampled = i < n_sampled
        # unseen_every>0: her N'inci satır has_unseen (kalite oranı doğrulaması için)
        flag_unseen = bool(unseen_every) and (i % unseen_every == 0)
        # labeled>0: ilk `labeled` satıra actual_label (gerçekleşen perf). İlk `pos_of_labeled` pozitif(1).
        # label_backfilled_at = geçmiş epoch → dirty tetiklemez (idempotans korunur), yalnız veri.
        # ⚠️ ÜRETİME SADIK: etiketsiz satırda actual_label anahtarı OMIT edilir (SQL NULL) — açıkça
        # None bind etmek JSON 'null' saklardı (none_as_null=False) → isnot(None) hepsini etiketli
        # sayardı (n_sampled gotcha'sının ikizi). Scoring bu alanı hiç yazmaz → üretimde gerçek NULL.
        is_labeled = i < labeled
        row = {
            "prediction_id": uuid.uuid4().hex, "project_id": project_id, "artifact_id": artifact_id,
            "created_at": day_start + 100 + (i % 1000),
            "input_row": ({"f": i} if sampled else None), "input_row_sampled": sampled,
            "prediction": 1, "probability": 0.5,
            "input_warning_flags": {"has_unseen": flag_unseen, "has_out_of_range": False},
        }
        if is_labeled:
            row["actual_label"] = 1 if i < pos_of_labeled else 0
            row["label_backfilled_at"] = day_start + 50
        out.append(row)
    return out


@pytest.fixture(scope="module")
def seeded():
    Base.metadata.create_all(bind=engine)
    import time
    today_start = (int(time.time()) // _DAY) * _DAY
    dayA = today_start - 3 * _DAY
    dayB = today_start - 2 * _DAY

    _write_stub_artifact("artA")
    _write_stub_artifact("artB")
    # artZ: STUB YOK → no_reference

    db = SessionLocal()
    try:
        u = UserModel(email="mon-test@local.test", hashed_password="x")
        db.add(u); db.commit(); db.refresh(u)
        p = ProjectModel(title="mon-test", status="completed", user_id=u.id)
        db.add(p); db.commit(); db.refresh(p)
        pid = p.id
        m = []
        # Gün A: artA 600/600 → ok(PSI)+ok(CBPE, ≥500)  ·  artB 50/50 → insufficient (champion-bölme)
        # artA: her 10'uncu satır has_unseen → 60 satır → kalite oranı 0.1
        # artA gerçekleşen: 250 etiketli (≥200 → ok), 200 pozitif → accuracy 0.8, labeled_pos_rate 0.8
        m += _rows(pid, "artA", dayA, 600, 600, unseen_every=10, labeled=250, pos_of_labeled=200)
        # artB gerçekleşen: 10 etiketli (<200 → insufficient_labels) ama label_coverage 0.2 GÖRÜNÜR
        m += _rows(pid, "artB", dayA, 50, 50, labeled=10, pos_of_labeled=6)
        # Gün B: artA 320/200 → n_total≠n_sampled, n_sampled<300 → insufficient  ·  artZ 350/350 → no_reference
        m += _rows(pid, "artA", dayB, 320, 200)
        m += _rows(pid, "artZ", dayB, 350, 350)
        # BUGÜN (yarım): işlenMEmeli
        m += _rows(pid, "artA", today_start, 400, 400)
        db.bulk_insert_mappings(PredictionModel, m)
        db.commit()
    finally:
        db.close()

    yield {"pid": pid, "dayA": dayA, "dayB": dayB, "today": today_start}

    db = SessionLocal()
    try:
        db.query(MonitoringSummaryModel).filter(MonitoringSummaryModel.project_id == pid).delete()
        db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.query(UserModel).filter(UserModel.email == "mon-test@local.test").delete()
        db.commit()
    finally:
        db.close()
    for aid in ("artA", "artB"):
        shutil.rmtree(os.path.join("data", "artifacts", aid), ignore_errors=True)


def _summaries(pid):
    db = SessionLocal()
    try:
        return {(s.artifact_id, s.window_start): s
                for s in db.query(MonitoringSummaryModel)
                          .filter(MonitoringSummaryModel.project_id == pid).all()}
    finally:
        db.close()


def test_windowing_champion_split_status_psi(seeded):
    written = run_monitoring_aggregation()
    assert written == 4, f"4 kapanmış-gün grubu beklenirdi (bugün hariç), yazılan={written}"

    s = _summaries(seeded["pid"])
    assert len(s) == 4
    assert ("artA", seeded["today"]) not in s              # yarım-gün işlenMEmeli

    # Champion-bölme: gün A'da iki ayrı artifact, her biri kendi n/status'u
    a = s[("artA", seeded["dayA"])]
    b = s[("artB", seeded["dayA"])]
    assert (a.n_total, a.n_sampled, a.status) == (600, 600, "ok")
    assert (b.n_total, b.n_sampled, b.status) == (50, 50, "insufficient_n")

    # OK grup → PSI DOLU; "f" ölçüldü (dejenere değil)
    assert a.psi_overall is not None and a.psi_max is not None
    assert a.psi_degenerate_count == 0
    assert "f" in a.psi_per_feature and a.psi_per_feature["f"]["degenerate"] is False
    # 3b: importance (f raw, %100 atıf) → weight + attributed_ratio dolu, dejenere kütle 0
    assert a.importance_attributed_ratio == 1.0
    assert a.psi_per_feature["f"]["importance_weight"] == 1.0
    assert a.degenerate_importance_mass == 0.0
    # Parça 4 kalite: per-row bayraktan (PSI'dan bağımsız). 600 satırın 60'ı unseen → oran 0.1
    assert a.quality_rates["unseen"]["count"] == 60
    assert a.quality_rates["unseen"]["rate"] == 0.1
    assert a.quality_rates["out_of_range"]["count"] == 0
    assert a.quality_rates["n"] == 600 and a.quality_rates["alarm_eligible"] is True
    # insufficient_n PSI grubu bile kalite ALIR (bağımsız); artB 50 satır → alarm_eligible False (<100)
    assert b.quality_rates is not None and b.quality_rates["alarm_eligible"] is False

    # Parça 4 CBPE: artA kalibre + n≥500 → ok; probability hep 0.5 → tahmini AUC 0.5, baseline korunur
    assert a.cbpe_gate_status == "ok" and a.cbpe_estimate is not None
    assert a.cbpe_estimate["estimated_roc_auc"] == 0.5
    assert a.cbpe_estimate["baseline_roc_auc"] == 0.86
    assert a.cbpe_estimate["assumes"] == "covariate_drift_only"
    # artB kalibre AMA n=50<500 → insufficient_n, estimate NULL
    assert b.cbpe_gate_status == "insufficient_n" and b.cbpe_estimate is None

    # GERÇEKLEŞEN performans: artA 250 etiketli (≥200) → ok; accuracy/pozitif-oran/sapma deterministik
    assert a.realized_gate_status == "ok" and a.realized_metrics is not None
    assert a.label_coverage == round(250 / 600, 6)         # HER ZAMAN dolu (dashboard "kapsam" satırı)
    assert a.realized_metrics["labeled_n"] == 250
    assert a.realized_metrics["accuracy"] == 0.8           # pred=1, 200/250 actual=1
    assert a.realized_metrics["labeled_positive_rate"] == 0.8
    assert a.realized_metrics["baseline_positive_rate"] == 0.30
    assert a.realized_metrics["positive_rate_deviation"] == 0.5   # SEÇİLİM YANLILIĞI kanıtı (kod basar)
    assert a.realized_metrics["roc_auc"] == 0.5            # tüm prob 0.5 (bağ) → AUC 0.5
    assert a.realized_metrics["assumes"] == "labels_missing_at_random"
    # artB 10 etiketli (<200) → insufficient_labels; metrik NULL AMA label_coverage GÖRÜNÜR (%20)
    assert b.realized_gate_status == "insufficient_labels" and b.realized_metrics is None
    assert b.label_coverage == round(10 / 50, 6)

    # insufficient_n → PSI NULL (referans var ama örneklem az; n_total 320 olsa da n_sampled 200<300)
    d = s[("artA", seeded["dayB"])]
    assert (d.n_total, d.n_sampled, d.status) == (320, 200, "insufficient_n")
    assert d.psi_overall is None and d.psi_max is None and d.psi_per_feature is None

    # no_reference → PSI NULL (artZ diskte yok; insufficient_n'den AYRI, n yeterli olsa da)
    z = s[("artZ", seeded["dayB"])]
    assert z.status == "no_reference" and z.n_sampled == 350
    assert z.psi_overall is None and z.psi_per_feature is None
    assert z.cbpe_gate_status == "suppressed_no_baseline" and z.cbpe_estimate is None   # artifact yok
    # gerçekleşen: etiket yok → no_labels, coverage 0.0 (referans yokluğundan BAĞIMSIZ — etikete bağlı)
    assert z.realized_gate_status == "no_labels" and z.realized_metrics is None
    assert z.label_coverage == 0.0

    for row in s.values():
        assert row.window_end == row.window_start + _DAY
        assert row.window_kind == "daily" and row.window_start % _DAY == 0


def test_idempotent_then_dirty_reprocess(seeded):
    run_monitoring_aggregation()                           # taban (test sırası bağımsız)
    before = _summaries(seeded["pid"])
    assert len(before) == 4

    # (1) İDEMPOTANSLIK: değişiklik yok → hepsi atlanır (written=0), satır sabit
    assert run_monitoring_aggregation() == 0
    assert len(_summaries(seeded["pid"])) == 4

    # (2) DIRTY: gün A / artA'da bir satırın label_backfilled_at'ini computed_at üstüne çek → kirlenir
    a_before = before[("artA", seeded["dayA"])]
    db = SessionLocal()
    try:
        row = (db.query(PredictionModel)
                 .filter(PredictionModel.project_id == seeded["pid"],
                         PredictionModel.artifact_id == "artA",
                         PredictionModel.created_at >= seeded["dayA"],
                         PredictionModel.created_at < seeded["dayA"] + _DAY)
                 .first())
        row.label_backfilled_at = a_before.computed_at + 10_000
        db.commit()
    finally:
        db.close()

    assert run_monitoring_aggregation() == 1               # yalnız kirlenen pencere
    after = _summaries(seeded["pid"])
    assert len(after) == 4                                 # UPSERT, yeni satır yok
    assert after[("artA", seeded["dayA"])].computed_at >= a_before.computed_at


def test_label_backfill_flips_realized(seeded):
    """Geç etiket backfill → dirty → yeniden hesap: no_labels → ok. PSI/CBPE donmuş pencerede değişmez."""
    run_monitoring_aggregation()                           # taban
    before = _summaries(seeded["pid"])
    d0 = before[("artA", seeded["dayB"])]
    assert d0.realized_gate_status == "no_labels" and d0.label_coverage == 0.0

    import time as _t
    now = _t.time()
    db = SessionLocal()
    try:
        rows = (db.query(PredictionModel)
                  .filter(PredictionModel.project_id == seeded["pid"],
                          PredictionModel.artifact_id == "artA",
                          PredictionModel.created_at >= seeded["dayB"],
                          PredictionModel.created_at < seeded["dayB"] + _DAY)
                  .limit(250).all())
        for idx, r in enumerate(rows):
            r.actual_label = 1 if idx < 200 else 0         # 200 pozitif → accuracy 0.8 (pred hep 1)
            r.label_backfilled_at = now + 10_000           # > computed_at → pencere kirlenir
        db.commit()
    finally:
        db.close()

    # ≥1: dayB kesin kirlendi. (dayA, önceki testte label_backfilled_at=computed_at+10000 UZAK GELECEĞE
    # çekildiğinden kalıcı-dirty kaldı → o da sayılabilir; bu çapraz-test artefaktı, dayB kanıtı d1'de.)
    assert run_monitoring_aggregation() >= 1
    after = _summaries(seeded["pid"])
    d1 = after[("artA", seeded["dayB"])]
    assert d1.realized_gate_status == "ok"
    assert d1.realized_metrics["labeled_n"] == 250
    assert d1.realized_metrics["accuracy"] == 0.8
    assert d1.label_coverage == round(250 / 320, 6)
    # PSI/CBPE donmuş pencerede yeniden hesap → BİREBİR aynı (israf ama zararsız): durum korunur
    assert d1.status == "insufficient_n"                   # n_sampled 200<300
    assert d1.cbpe_gate_status == "insufficient_n"         # n_total 320<500


def test_triage_and_distribution_columns(seeded):
    """Faz 0: aggregator overall_status + psi_scalars + psi_distribution (deferred) yazar.
    ok pencere → measured rozet + scalars/distribution dolu; no_reference/insufficient → unknown + scalars null."""
    from sqlalchemy.orm import undefer
    run_monitoring_aggregation()
    db = SessionLocal()
    try:
        rows = {(s.artifact_id, s.window_start): s
                for s in db.query(MonitoringSummaryModel)
                          .options(undefer(MonitoringSummaryModel.psi_distribution))
                          .filter(MonitoringSummaryModel.project_id == seeded["pid"]).all()}

        # artA gün A (ok): çekirdek ölçüldü → rozet measured (warning/critical), reason yapısı dolu
        a = rows[("artA", seeded["dayA"])]
        assert a.overall_status in ("healthy", "warning", "critical")
        assert a.overall_status_reason["driver"] is not None
        assert "measurement_coverage" in a.overall_status_reason
        # psi_scalars = hafif projeksiyon; psi_distribution = overlay (deferred, undefer ile yüklendi)
        assert a.psi_scalars == {"f": a.psi_max}                 # tek kolon "f"
        assert a.psi_distribution["f"]["kind"] == "continuous"
        assert a.psi_distribution["f"]["edges"][0] is None       # ±inf → null
        assert sum(a.psi_distribution["f"]["prod_counts"]) == a.psi_distribution["f"]["prod_total"]

        # artZ (no_reference): üç çekirdek eksen unmeasured, kalite alarmsız → unknown
        z = rows[("artZ", seeded["dayB"])]
        assert z.overall_status == "unknown"
        assert z.overall_status_reason["code"] == "all_core_unmeasured_unknown"
        assert z.psi_scalars is None and z.psi_distribution is None

        # artB (insufficient_n): PSI/CBPE/realized hepsi yetersiz, kalite n<100 → unknown
        b = rows[("artB", seeded["dayA"])]
        assert b.overall_status == "unknown"
        assert b.psi_scalars is None
    finally:
        db.close()


def _src_row(project_id, artifact_id, day_start, i, source, sampled=True):
    """Tek prediction mapping — `source` AÇIK (bulk_insert_mappings Python-default UYGULAMAZ →
    omit edilen kolon NULL olur; bu yüzden NULL/'api'/'ui_manual' vakalarını açıkça kuruyoruz)."""
    r = {
        "prediction_id": uuid.uuid4().hex, "project_id": project_id, "artifact_id": artifact_id,
        "created_at": day_start + 100 + (i % 1000),
        "input_row": ({"f": i} if sampled else None), "input_row_sampled": sampled,
        "prediction": 1, "probability": 0.5,
        "input_warning_flags": {"has_unseen": False, "has_out_of_range": False},
    }
    if source is not None:                 # None → kolon OMIT (SQL NULL = alan-öncesi kayıt)
        r["source"] = source
    return r


@pytest.fixture(scope="module")
def seeded_source():
    """Kaynak-ayrımı için AYRI proje (module `seeded`'e dokunmaz): karışık pencere + pure-ui pencere."""
    Base.metadata.create_all(bind=engine)
    import time
    today_start = (int(time.time()) // _DAY) * _DAY
    dayMix = today_start - 5 * _DAY        # karışık: api + NULL(legacy) + ui_manual
    dayUI = today_start - 4 * _DAY         # YALNIZ ui_manual → özet YAZILMAMALI

    _write_stub_artifact("artS")
    db = SessionLocal()
    try:
        u = UserModel(email="mon-src@local.test", hashed_password="x")
        db.add(u); db.commit(); db.refresh(u)
        p = ProjectModel(title="mon-src", status="completed", user_id=u.id)
        db.add(p); db.commit(); db.refresh(p)
        pid = p.id
        m = []
        # Karışık gün: 400 "api" + 200 NULL(legacy) = 600 ÜRETİM (PSI ok, ≥500 CBPE) + 150 "ui_manual" DIŞLANIR
        for i in range(400):
            m.append(_src_row(pid, "artS", dayMix, i, "api"))
        for i in range(400, 600):
            m.append(_src_row(pid, "artS", dayMix, i, None))          # legacy NULL → coalesce api
        for i in range(600, 750):
            m.append(_src_row(pid, "artS", dayMix, i, "ui_manual"))   # dışlanır (n_total saymaz)
        # Pure-ui gün: yalnız 300 ui_manual → n_total 0 → özet satırı YOK
        for i in range(300):
            m.append(_src_row(pid, "artS", dayUI, i, "ui_manual"))
        db.bulk_insert_mappings(PredictionModel, m)
        db.commit()
    finally:
        db.close()

    yield {"pid": pid, "dayMix": dayMix, "dayUI": dayUI}

    db = SessionLocal()
    try:
        db.query(MonitoringSummaryModel).filter(MonitoringSummaryModel.project_id == pid).delete()
        db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.query(UserModel).filter(UserModel.email == "mon-src@local.test").delete()
        db.commit()
    finally:
        db.close()
    shutil.rmtree(os.path.join("data", "artifacts", "artS"), ignore_errors=True)


def test_source_excludes_ui_manual_and_counts_it(seeded_source):
    """ui_manual monitoring'ten DIŞLANIR (n_total saymaz) ama excluded_ui_count ile GÖRÜNÜR;
    NULL(legacy) coalesce ile üretim sayılır (körleşme yok); pure-ui pencere özet YAZMAZ."""
    run_monitoring_aggregation()
    s = _summaries(seeded_source["pid"])

    # Karışık gün: 400 api + 200 NULL = 600 üretim; 150 ui DIŞLANDI ama sayıldı
    mix = s[("artS", seeded_source["dayMix"])]
    assert mix.n_total == 600, f"NULL(legacy) coalesce api sayılmalı; ui hariç 600 beklendi, {mix.n_total}"
    assert mix.n_sampled == 600                                   # örneklem de ui hariç
    assert mix.excluded_ui_count == 150                           # first-class dışlanan sayacı
    assert mix.status == "ok"                                     # 600 üretim → PSI ölçülür
    # PSI/CBPE ui_manual satırlarını GÖRMEDİ → üretim-yalnız (600, ≥500) hesaplandı
    assert mix.psi_overall is not None
    assert mix.cbpe_gate_status == "ok" and mix.cbpe_estimate is not None

    # Pure-ui gün: üretim trafiği yok → ÖZET SATIRI YAZILMAMALI (sahte boş satır üretilmez)
    assert ("artS", seeded_source["dayUI"]) not in s


def test_day_floor_uses_modulo_not_rounding_cast():
    """REGRESYON KİLİDİ (2026-08-06 bucketing bug). Eski ``cast(created_at / _DAY, Integer) * _DAY``
    Postgres'te FLOAT bölme + cast YUVARLIYORDU → gün-kesri ≥0.5 tahminler bir SONRAKİ güne kayıyordu;
    SQLite int/int→floor verdiğinden 26 test YEŞİL kalıp bug'ı sakladı (SQLite/PG semantik sapması).
    Modulo (`created_at - created_at % _DAY`) dial-agnostik. Bu test DERLENEN Postgres SQL'ini denetler:
    cast-round deseni geri gelirse (veya `%` kaybolursa) KIRILIR → sınıf testte GÖRÜNÜR olur."""
    from sqlalchemy.dialects import postgresql
    from src.services.monitoring.aggregator import _day_floor
    sql = str(_day_floor(PredictionModel.created_at).compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "%" in sql, f"bucket modulo-floor kullanmalı (Postgres cast-round tuzağı), bulunan: {sql}"
    assert "CAST" not in sql.upper(), f"cast(round) deseni geri gelmiş: {sql}"


def test_aggregation_floors_afternoon_predictions_to_same_day():
    """İNVARYANT: gün-kesri ≥0.5 (öğleden SONRA) tahminler AYNI UTC gününe bucket'lanır, sonrakine DEĞİL —
    ve window_start alt-sorguların [ws, ws+DAY) sınırını KAPSAR (bug'da satırlar sınırın altına düşüyordu).
    SQLite'ta zaten floor; Postgres'te eski cast-round bunu kırardı. Self-contained (module fixture'a dokunmaz)."""
    Base.metadata.create_all(bind=engine)
    import time
    day = (int(time.time()) // _DAY) * _DAY - 3 * _DAY        # 3 gün önce (kapanmış)
    afternoon = day + 82800                                   # 23:00 (gün-kesri 0.958 → eski kod yuvarlardı)
    db = SessionLocal()
    try:
        u = UserModel(email="mon-floor@local.test", hashed_password="x"); db.add(u); db.commit(); db.refresh(u)
        p = ProjectModel(title="mon-floor", status="completed", user_id=u.id); db.add(p); db.commit(); db.refresh(p)
        pid = p.id
        db.bulk_insert_mappings(PredictionModel, [
            {"prediction_id": uuid.uuid4().hex, "project_id": pid, "artifact_id": "artF",
             "created_at": afternoon, "input_row": None, "input_row_sampled": False,
             "prediction": 1, "probability": 0.5} for _ in range(5)])
        db.commit()
    finally:
        db.close()
    try:
        run_monitoring_aggregation()
        s = _summaries(pid)
        assert len(s) == 1, f"tek pencere beklenirdi, bulunan: {list(s)}"
        (aid, ws), row = next(iter(s.items()))
        assert ws == day, f"öğleden sonraki tahmin AYNI güne floor'lanmalı: ws={ws} beklenen={day} (sonraki gün={day + _DAY})"
        assert ws <= afternoon < ws + _DAY, "window_start alt-sorgu sınırı created_at'i KAPSAMALI"
    finally:
        db = SessionLocal()
        try:
            db.query(MonitoringSummaryModel).filter(MonitoringSummaryModel.project_id == pid).delete()
            db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
            db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
            db.query(UserModel).filter(UserModel.email == "mon-floor@local.test").delete()
            db.commit()
        finally:
            db.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
