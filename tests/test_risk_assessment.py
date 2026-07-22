"""Faz 2 Grup B: deterministik risk listesi.

- churn'de en az bir gerçek risk (confident-error warn) — "risk yok" demek yanlıştı.
- credit/temiz veride false-positive üretmez; n/CI kapısı örtüşen farkı eler.
- fairness (#12): DI band-dışı + CI örtüşmez iken amplification=DI/base_rate_ratio ile
  model-warn (verinin ötesine geçer) vs veri-info (model sadık, veride eşitsizlik) ayrılır;
  fairness yoksa GERÇEK sebeple koşmadı (rule 3 — koşmayan temiz sayılmaz, "n/CI yok" yalanı yok).
- critical YALNIZ veri bütünlüğünden; model kalitesi asla critical (yapısal assert).
"""
import pytest

from src.services.risk_assessment import build_risk_list, RiskCheck, _SEVERITY


def _ids(checks):
    return {c.id for c in checks}


def _by_id(checks, cid):
    return next((c for c in checks if c.id == cid), None)


# Confusion: class1 (yes) recall 272/407=0.668, class0 (no) recall 1400/1593=0.879 → oran 0.76<0.8,
# CI'lar (n büyük) örtüşmüyor → minority_recall RİSK.
_CM_GAP = {"labels": [0, 1], "matrix": [[1400, 193], [135, 272]]}


def _churn_state(**over):
    ev = {
        "class_balance": {"distribution": {0: 0.7965, 1: 0.2035}, "minority_class": 0,
                          "minority_rate": 0.2035, "majority_class": 0, "majority_rate": 0.7965,
                          "level": "imbalanced", "positive_class": 1, "interest_classes": [1],
                          "n_classes": 2},
        "error_analysis": {"n_total": 2000, "n_errors": 323,
                           "confidence_analysis": {"high_confidence_errors": 28,
                                                   "high_confidence_errors_pct": 8.67}},
        "confusion_matrix": _CM_GAP,
        "per_class_metrics": {"0": {"recall": 0.879, "support": 1593},
                              "1": {"recall": 0.668, "support": 407}},
        "overfitting": {"status": "normal", "gap": -0.006},
        "calibration": {"brier_score": 0.1448, "brier_baseline": 0.1621},
        "feature_importance_summary": {"leakage_suspects": []},
        "target_lock_leakage": [],
    }
    ev.update(over.pop("evaluation_report", {}))
    st = {"intent": "classification", "evaluation_report": ev,
          "relations_report": {"redundant_pairs": [{"a": "x", "b": "y", "corr": 0.99}] * 8},
          "feature_report": {"normalization_collisions": []},
          "data_quality_status": "ok"}
    st.update(over)
    return st


def test_churn_catches_confident_errors_as_warn():
    """Denetimin başında 'risk yok' demenin yanlış olduğu kanıt: 28 confident-error → WARN."""
    rl = build_risk_list(_churn_state())
    ce = _by_id(rl["risks"], "confident_errors")
    assert ce is not None and ce.severity == "warn", rl["risks"]
    # rate sürücüsü: 28/2000=%1.4 ≥ %1 taban + sayı ≥ 10
    assert ce.value == 28


def test_churn_catches_minority_recall_gap():
    """0.668/0.879=0.76 < 0.8 ve CI'lar örtüşmüyor → azınlık recall riski."""
    rl = build_risk_list(_churn_state())
    mr = _by_id(rl["risks"], "minority_recall")
    assert mr is not None and mr.severity == "warn", [r.id for r in rl["risks"]]


def test_churn_is_not_risk_free():
    """Asıl bug: churn 'belirgin risk yok' diyordu. Artık DEMEZ."""
    rl = build_risk_list(_churn_state())
    assert len(rl["risks"]) >= 1


# ── #12 fairness plug ────────────────────────────────────────────────────────

def _fair(segments, **kw):
    base = {"status": "ok", "task": "classification", "di_reference_band": [0.8, 1.25],
            "segments": segments}
    base.update(kw)
    return base


def _seg(ref, ref_ci, g, g_ci, di, brr, n=500):
    """Tek-grup segment: referans + bir karşılaştırma grubu (seçim oranı CI'larıyla)."""
    return {
        "groups": {ref: {"selection_rate_ci": ref_ci}, g: {"selection_rate_ci": g_ci}},
        "comparisons": {"reference_group": ref,
                        "groups": {g: {"disparate_impact": di, "base_rate_ratio": brr, "n": n}}},
    }


def test_fairness_absent_not_run_with_real_reason():
    """Fairness üretilmediyse KOŞMADI + GERÇEK sebep — eski 'n/CI yok' yalanı KALKTI (rule 3)."""
    rl = build_risk_list(_churn_state())          # fixture'da fairness yok
    f = _by_id(rl["not_run"], "fairness")
    assert f is not None and not f.ran
    assert "n/CI" not in f.not_run_reason["en"]
    assert "not produced" in f.not_run_reason["en"]
    assert f.id not in _ids(rl["clean"])          # temiz sayılmaz


def test_fairness_status_not_ok_carries_real_reason():
    """status != ok → fairness'ın dil-nötr sebep KODU iki dile çevrilerek taşınır."""
    fair = {"status": "not_computed", "task": "classification",
            "reason": {"code": "fr_no_carrier"}, "segments": {}}
    rl = build_risk_list(_churn_state(evaluation_report={"fairness": fair}))
    f = _by_id(rl["not_run"], "fairness")
    assert f is not None and not f.ran
    assert "carrier" in f.not_run_reason["en"].lower()   # gerçek sebep, yer tutucu değil


def test_fairness_model_amplification_warn():
    """country=Germany: DI 2.44 / base_rate_ratio 1.748 = 1.40 band-dışı → MODEL warn."""
    fair = _fair({"country": _seg("France", {"lower": 0.143, "upper": 0.189},
                                  "Germany", {"lower": 0.359, "upper": 0.444},
                                  di=2.44, brr=1.748)})
    rl = build_risk_list(_churn_state(evaluation_report={"fairness": fair}))
    fr = [r for r in rl["risks"] if r.id == "fairness"]
    assert len(fr) == 1 and fr[0].severity == "warn", rl["risks"]
    assert fr[0].value == pytest.approx(1.396, abs=0.01)   # amplification


def test_fairness_data_disparity_info_not_silent():
    """gender=Female: DI 1.62 / brr 1.54 = 1.05 band-İÇİ → model sadık ama veri eşitsiz → INFO."""
    fair = _fair({"gender": _seg("Male", {"lower": 0.151, "upper": 0.197},
                                 "Female", {"lower": 0.253, "upper": 0.310},
                                 di=1.62, brr=1.54)})
    rl = build_risk_list(_churn_state(evaluation_report={"fairness": fair}))
    fr = [r for r in rl["risks"] if r.id == "fairness"]
    assert len(fr) == 1 and fr[0].severity == "info", rl["risks"]
    # metin model/veri ayrımını net yapar
    assert "VERİ" in fr[0].detail["tr"] and "DATA" in fr[0].detail["en"]


def test_fairness_model_and_data_both_surface():
    """Germany (model) + gender (veri) aynı raporda → warn VE info ikisi de görünür (biri diğerini gizlemez)."""
    fair = _fair({
        "country": _seg("France", {"lower": 0.143, "upper": 0.189},
                        "Germany", {"lower": 0.359, "upper": 0.444}, di=2.44, brr=1.748),
        "gender": _seg("Male", {"lower": 0.151, "upper": 0.197},
                       "Female", {"lower": 0.253, "upper": 0.310}, di=1.62, brr=1.54),
    })
    rl = build_risk_list(_churn_state(evaluation_report={"fairness": fair}))
    sevs = {r.severity for r in rl["risks"] if r.id == "fairness"}
    assert sevs == {"warn", "info"}, rl["risks"]


def test_fairness_ci_overlap_not_flagged():
    """DI band-dışı ama seçim oranı CI'ları örtüşüyor → gürültü → risk DEĞİL (temiz)."""
    fair = _fair({"country": _seg("France", {"lower": 0.10, "upper": 0.30},
                                  "Germany", {"lower": 0.20, "upper": 0.45},
                                  di=1.6, brr=1.1)})   # CI'lar örtüşür (0.20 < 0.30)
    rl = build_risk_list(_churn_state(evaluation_report={"fairness": fair}))
    assert not [r for r in rl["risks"] if r.id == "fairness"]
    assert _by_id(rl["clean"], "fairness") is not None


# ── D3 aksiyon yoğunlaşması plug ─────────────────────────────────────────────

def _ac(groups, status="ok"):
    return {"status": status, "n_top": 200, "columns": {"g": {"groups": groups, "skipped_groups": {}}}}


def _acg(rep, base, kind, sev):
    return {"rep_ratio": rep, "base_rate_ratio": base, "kind": kind, "flag_severity": sev,
            "n_decile": 40, "n_pop": 200}


def test_ac_coverage_gap_warn():
    ac = _ac({"P": _acg(0.0, 3.0, "coverage_gap", "warn"),
              "A": _acg(1.1, 0.9, "proportional", None)})
    rl = build_risk_list(_churn_state(evaluation_report={"action_concentration": ac}))
    a = [r for r in rl["risks"] if r.id == "action_concentration"]
    assert len(a) == 1 and a[0].severity == "warn", rl["risks"]
    assert "kayıp" in a[0].detail["tr"] and "loss" in a[0].detail["en"]


def test_ac_over_concentration_info():
    ac = _ac({"Q": _acg(3.75, 0.45, "over_unexpected", "info")})
    rl = build_risk_list(_churn_state(evaluation_report={"action_concentration": ac}))
    a = [r for r in rl["risks"] if r.id == "action_concentration"]
    assert len(a) == 1 and a[0].severity == "info"


def test_ac_expected_concentration_not_flagged():
    """Grade D/E vakası: expected → risk DEĞİL → temiz (yön netliği)."""
    ac = _ac({"D": _acg(5.0, 2.7, "expected", None), "E": _acg(10.0, 4.2, "expected", None)})
    rl = build_risk_list(_churn_state(evaluation_report={"action_concentration": ac}))
    assert not [r for r in rl["risks"] if r.id == "action_concentration"]
    assert _by_id(rl["clean"], "action_concentration") is not None


def test_ac_absent_not_run_real_reason():
    rl = build_risk_list(_churn_state())          # fixture'da action_concentration yok
    a = _by_id(rl["not_run"], "action_concentration")
    assert a is not None and not a.ran and "not produced" in a.not_run_reason["en"]


def test_fairness_within_band_clean():
    """DI band içinde (0.97) → değerlendirmeye girmez → temiz."""
    fair = _fair({"country": _seg("France", {"lower": 0.14, "upper": 0.20},
                                  "Spain", {"lower": 0.13, "upper": 0.21},
                                  di=0.97, brr=0.99)})
    rl = build_risk_list(_churn_state(evaluation_report={"fairness": fair}))
    assert not [r for r in rl["risks"] if r.id == "fairness"]
    assert _by_id(rl["clean"], "fairness") is not None


def test_ci_gate_overlapping_intervals_not_flagged():
    """Aynı recall farkı ama n KÜÇÜK → CI'lar örtüşür → risk DEĞİL (false-positive önlenir)."""
    # class1 recall 6/9=0.667, class0 34/40=0.85 — küçük n → geniş, örtüşen Wilson CI
    small_cm = {"labels": [0, 1], "matrix": [[34, 6], [3, 6]]}
    rl = build_risk_list(_churn_state(evaluation_report={
        "confusion_matrix": small_cm,
        "class_balance": {"distribution": {0: 0.82, 1: 0.18}, "minority_class": 0,
                          "minority_rate": 0.18, "majority_class": 0, "majority_rate": 0.82,
                          "level": "imbalanced", "positive_class": 1, "interest_classes": [1],
                          "n_classes": 2}}))
    assert _by_id(rl["risks"], "minority_recall") is None
    assert _by_id(rl["clean"], "minority_recall") is not None   # koştu, temiz


def test_clean_data_says_checks_ran_all_clean():
    """Temiz veri: 'N kontrol koştu, hepsi temiz' — boş 'risk yok' değil (ran_detectors)."""
    clean_cm = {"labels": [0, 1], "matrix": [[900, 30], [25, 45]]}  # dengeli-ish, recall yakın
    st = _churn_state(evaluation_report={
        "confusion_matrix": clean_cm,
        "class_balance": {"distribution": {0: 0.5, 1: 0.5}, "minority_class": 0, "minority_rate": 0.5,
                          "majority_class": 1, "majority_rate": 0.5, "level": "balanced",
                          "positive_class": 1, "interest_classes": [1], "n_classes": 2},
        "error_analysis": {"n_total": 1000, "n_errors": 55,
                           "confidence_analysis": {"high_confidence_errors": 2,
                                                   "high_confidence_errors_pct": 3.6}},
    })
    st["relations_report"] = {"redundant_pairs": []}
    rl = build_risk_list(st)
    assert rl["risks"] == []
    assert len(rl["clean"]) >= 5      # birçok kontrol koştu ve temiz
    assert _by_id(rl["not_run"], "fairness") is not None   # fairness yine not-run


def test_critical_only_from_data_integrity():
    """data_quality=failed → critical (source=data_quality). Model kalitesi ASLA critical."""
    rl = build_risk_list(_churn_state(data_quality_status="failed"))
    dq = _by_id(rl["risks"], "data_quality")
    assert dq.severity == "critical" and dq.source == "data_quality"
    # Model kaynaklı hiçbir risk critical değil (yapısal assert build_risk_list içinde de var)
    for r in rl["risks"]:
        if r.source != "data_quality":
            assert r.severity in ("info", "warn")


def test_model_critical_assert_fires():
    """Yapısal garanti: model kaynaklı critical konstrüksiyonu assert'i tetikler."""
    import src.services.risk_assessment as ra
    orig = ra._check_overfitting
    ra._check_overfitting = lambda ev: RiskCheck("overfitting", ran=True, risk=True,
                                                 severity="critical", source="model")
    try:
        with pytest.raises(AssertionError):
            build_risk_list(_churn_state())
    finally:
        ra._check_overfitting = orig


def test_regression_skips_class_based_checks():
    """Regresyon: sınıf-tabanlı kontroller (imbalance/recall/confident/brier) hiç koşmaz."""
    rl = build_risk_list({"intent": "regression", "evaluation_report": {},
                          "feature_report": {}, "relations_report": {}})
    ran_or_notrun = _ids(rl["clean"]) | _ids(rl["not_run"]) | _ids(rl["risks"])
    assert "severe_imbalance" not in ran_or_notrun
    assert "confident_errors" not in ran_or_notrun


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
