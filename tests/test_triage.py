"""Katman 2 Faz 0 — TRIAGE roll-up (SAF, DB yok). overall_status = dört monitör → tek rozet.

Doğrular: dört değer (healthy/warning/critical/**unknown**) · eksen tavanları (CBPE asla critical, quality
warning) · realized critical üretir · ÇAPRAZ UZLAŞMA tek-yön + suppressed_axes bilgi korunur + r_trusted çiti
(invariant) · ölçüm-kapsamı healthy engeli (degen_mass>0.15) · unknown tabanı + quality-alarm ayrı kod ·
regresyon rmse bandı · tek-yön (healthy hiçbir şeyi YUKARI çekmez).
"""
import pytest

from src.core.config import settings
from src.services.monitoring.triage import compute_overall_status


def _call(**over):
    """Varsayılan: hepsi unmeasured. Test override eder."""
    base = dict(psi_status="no_reference", psi_max=None,
                importance_attributed_ratio=None, degenerate_importance_mass=None,
                cbpe_gate_status="suppressed_no_baseline", cbpe_estimate=None,
                realized_gate_status="no_labels", realized_metrics=None,
                label_coverage=0.0, quality_rates=None, settings=settings)
    base.update(over)
    return compute_overall_status(**base)


def _psi_ok(psi_max, attr=1.0, degen=0.0):
    return dict(psi_status="ok", psi_max=psi_max, importance_attributed_ratio=attr,
                degenerate_importance_mass=degen)


def _realized_ok(acc, bacc, coverage=0.8):
    return dict(realized_gate_status="ok", label_coverage=coverage,
                realized_metrics={"accuracy": acc, "baseline_accuracy": bacc})


def _cbpe_ok(ea, ba):
    return dict(cbpe_gate_status="ok",
                cbpe_estimate={"estimated_accuracy": ea, "baseline_accuracy": ba})


def _quality(unseen_rate, oor_rate=0.0, eligible=True):
    return dict(quality_rates={"unseen": {"rate": unseen_rate}, "out_of_range": {"rate": oor_rate},
                               "alarm_eligible": eligible})


def test_all_measured_healthy():
    st, r = _call(**_psi_ok(0.03), **_realized_ok(0.80, 0.80), **_cbpe_ok(0.79, 0.80), **_quality(0.01))
    assert st == "healthy" and r["code"] == "all_measured_healthy"
    assert r["core_unmeasured"] is False and r["suppressed_axes"] == {}


def test_realized_drop_critical_and_warning():
    # drop 0.12 ≥ 0.10 → critical
    st, r = _call(**_psi_ok(0.03), **_realized_ok(0.68, 0.80))
    assert st == "critical" and r["driver"] == "realized" and r["code"] == "realized_drop_critical"
    # drop 0.06 ∈ [0.05,0.10) → warning
    st2, r2 = _call(**_psi_ok(0.03), **_realized_ok(0.74, 0.80))
    assert st2 == "warning" and r2["code"] == "realized_drop_warning"


def test_cbpe_capped_at_warning():
    # CBPE tahmini %20 düşse bile CRITICAL DEĞİL — belirsiz tahmin kırmızı üretmez
    st, r = _call(**_cbpe_ok(0.60, 0.80))          # drop 0.20
    assert st == "warning" and r["driver"] == "cbpe" and r["code"] == "cbpe_drop_warning"


def test_psi_critical_when_realized_absent():
    st, r = _call(**_psi_ok(0.31))                 # realized unmeasured → PSI kritik hakim
    assert st == "critical" and r["driver"] == "drift" and r["code"] == "psi_high_impact_critical"


def test_realized_healthy_suppresses_drift_critical_one_directional():
    # realized GÜVENİLİR + healthy, PSI kritik → gerçek-doğru öncüyü EZER (critical→warning),
    # ham şiddet suppressed_axes'ta KORUNUR (dashboard "kritik ama etki yok")
    st, r = _call(**_psi_ok(0.31), **_realized_ok(0.80, 0.80))
    assert st == "warning" and r["driver"] == "drift"
    assert r["code"] == "psi_drift_no_impact_warning"
    assert r["suppressed_axes"] == {"drift": "critical"}      # bilgi gizlenmedi


def test_suppression_requires_trusted_realized_invariant():
    # realized ok AMA coverage<trust (0.3<0.5) → GÜVENİLİR DEĞİL → bastırma TETİKLENMEZ, PSI kritik kalır.
    # INVARIANT çiti: yanlı/başka-pencere realized öncü göstergeyi sessizce ezemez.
    st, r = _call(**_psi_ok(0.31),
                  realized_gate_status="ok", label_coverage=0.3,
                  realized_metrics={"accuracy": 0.80, "baseline_accuracy": 0.80})
    assert st == "critical" and r["code"] == "psi_high_impact_critical"
    assert r["suppressed_axes"] == {}


def test_quality_warning_with_healthy_core():
    st, r = _call(**_psi_ok(0.03), **_realized_ok(0.80, 0.80), **_quality(0.07))   # unseen 0.07 ≥ 0.05
    assert st == "warning" and r["driver"] == "quality" and r["code"] == "quality_alarm_warning"


def test_drift_coverage_inadequate_blocks_healthy():
    # PSI düşük (0.03) AMA degen_mass 0.20 > 0.15 → modelin %20'si ölçülemez → drift healthy OLAMAZ
    st, r = _call(**_psi_ok(0.03, degen=0.20))
    assert st == "warning" and r["driver"] == "drift" and r["code"] == "psi_drift_warning"
    assert r["measurement_coverage"]["adequate"] is False


def test_unknown_floor_no_quality():
    st, r = _call()                                # her şey unmeasured
    assert st == "unknown" and r["code"] == "all_core_unmeasured_unknown"
    assert r["core_unmeasured"] is True and r["driver"] is None


def test_unknown_floor_with_quality_alarm_distinct_code():
    st, r = _call(**_quality(0.09))                # çekirdek yok AMA kalite alarmı
    assert st == "unknown" and r["code"] == "all_core_unmeasured_with_quality_alarm"
    assert r["driver"] == "quality" and r["core_unmeasured"] is True
    assert r["value"]["unseen_rate"] == 0.09       # UI unknown çipin yanına kalite uyarısı basar


def test_regression_rmse_increase_bands():
    # göreli artış (rmse−baseline)/baseline: 0.30 ≥ 0.25 → critical
    st, r = _call(**_psi_ok(0.03), realized_gate_status="ok", label_coverage=0.8,
                  realized_metrics={"rmse": 13.0, "baseline_rmse": 10.0})
    assert st == "critical" and r["driver"] == "realized" and r["value"]["metric"] == "rmse_rel"


def test_one_directional_healthy_never_upgrades():
    # realized healthy + drift healthy + cbpe healthy → healthy (yukarı çekme YOK, false-critical yok)
    st, r = _call(**_psi_ok(0.03), **_realized_ok(0.81, 0.80), **_cbpe_ok(0.80, 0.80))
    assert st == "healthy" and r["suppressed_axes"] == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
