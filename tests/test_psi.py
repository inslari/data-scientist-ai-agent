"""Katman 2 Parça 3 — PSI çekirdeği (saf fonksiyon, DB'siz, hızlı).

Kilit iddialar: binary özel-kılıfsız ölçülür (dejenere DEĞİL) · gerçek sabit → dejenere+null+reason ·
PROD tek-değer DRİFT'tir (dejenere değil, PSI yüksek+finite) · Laplace inf'i önler · kategorik unseen→other ·
psi_overall/max/degenerate_count.
"""
import math

from src.services.monitoring.psi import compute_psi_summary, _continuous_bins, _categorical_bins

ALPHA = 0.5


def _cont(q, n=1000):
    return {"kind": "continuous", "mean": 0.0, "std": 1.0, "quantiles": dict(q), "n": n}


# ── kova inşası / dejenerasyon (yalnız REFERANS özelliği) ──────────────────────

def test_binary_reference_measured_not_degenerate():
    # quantile [0,0,0,1,1] → distinct interior {0,1}=2 → ÖLÇÜLÜR (Ölçüm 2'nin adj-collision'ı DEJENERE DEĞİL)
    res, reason = _continuous_bins(_cont({"0.01": 0.0, "0.25": 0.0, "0.5": 0.0, "0.75": 1.0, "0.99": 1.0}))
    assert reason is None
    edges, props = res
    assert edges == [-math.inf, 0.0, 1.0, math.inf]      # unique-edge → 3 kova
    assert len(props) == 3 and all(p > 0 for p in props) and abs(sum(props) - 1.0) < 1e-9


def test_constant_reference_degenerate_single_value():
    res, reason = _continuous_bins(_cont({"0.01": 5.0, "0.25": 5.0, "0.5": 5.0, "0.75": 5.0, "0.99": 5.0}))
    assert res is None and reason == "single_value"


def test_missing_quantiles_degenerate():
    res, reason = _continuous_bins(_cont({}))
    assert res is None and reason == "insufficient_distinct_edges"


# ── pencere özeti ─────────────────────────────────────────────────────────────

def test_binary_summary_finite():
    stat = _cont({"0.01": 0.0, "0.25": 0.0, "0.5": 0.0, "0.75": 1.0, "0.99": 1.0})
    prod = [{"x": 0} for _ in range(300)] + [{"x": 1} for _ in range(300)]
    out = compute_psi_summary({"x": stat}, prod, 1000, ALPHA)
    pf = out["psi_per_feature"]["x"]
    assert pf["degenerate"] is False and pf["n_bins"] == 3
    assert pf["psi"] is not None and math.isfinite(pf["psi"]) and pf["psi"] >= 0
    assert out["psi_degenerate_count"] == 0


def test_constant_summary_null_recorded():
    stat = _cont({"0.01": 5.0, "0.25": 5.0, "0.5": 5.0, "0.75": 5.0, "0.99": 5.0})
    out = compute_psi_summary({"x": stat}, [{"x": 5} for _ in range(10)], 1000, ALPHA)
    pf = out["psi_per_feature"]["x"]
    assert pf["degenerate"] is True and pf["psi"] is None and pf["reason"] == "single_value"
    assert out["psi_overall"] is None and out["psi_max"] is None
    assert out["psi_degenerate_count"] == 1               # ölçülemeyen first-class ("temiz ≠ ölçülmedi")


def test_prod_single_value_is_drift_not_degenerate():
    # Referans 5 distinct (sağlam); prod tek-değer = DRİFT → dejenere DEĞİL, PSI yüksek+finite (Laplace)
    stat = _cont({"0.01": 1.0, "0.25": 2.0, "0.5": 3.0, "0.75": 4.0, "0.99": 5.0})
    drift = compute_psi_summary({"x": stat}, [{"x": 3.0} for _ in range(500)], 5000, ALPHA)
    match = compute_psi_summary({"x": stat}, [{"x": v} for v in (1, 2, 3, 4, 5) for _ in range(100)], 5000, ALPHA)
    pd = drift["psi_per_feature"]["x"]
    assert pd["degenerate"] is False and math.isfinite(pd["psi"])   # inf YOK
    assert pd["psi"] > match["psi_per_feature"]["x"]["psi"]         # drift daha yüksek


def test_categorical_unseen_folds_to_other_finite():
    stat = {"kind": "categorical", "value_counts_normalized": {"a": 0.5, "b": 0.5}, "n_unique": 2}
    prod = [{"c": v} for v in ("a", "b", "zzz") for _ in range(100)]   # zzz = unseen → other
    out = compute_psi_summary({"c": stat}, prod, 1000, ALPHA)
    pf = out["psi_per_feature"]["c"]
    assert pf["degenerate"] is False and pf["n_bins"] == 3          # a, b, other
    assert math.isfinite(pf["psi"]) and pf["psi"] > 0              # tail_mass=0 ama Laplace inf'i önler


def test_overall_mean_and_max():
    s = _cont({"0.01": 1.0, "0.25": 2.0, "0.5": 3.0, "0.75": 4.0, "0.99": 5.0})
    # a: prod orta değerde (düşük drift), b: prod kenarda (yüksek drift)
    prod = [{"a": 3.0, "b": 1.0} for _ in range(400)]
    out = compute_psi_summary({"a": _cont(s["quantiles"]), "b": _cont(s["quantiles"])}, prod, 5000, ALPHA)
    pa = out["psi_per_feature"]["a"]["psi"]
    pb = out["psi_per_feature"]["b"]["psi"]
    assert out["psi_overall"] == round((pa + pb) / 2, 6)
    assert out["psi_max"] == round(max(pa, pb), 6)
    assert out["psi_degenerate_count"] == 0


def test_scalars_and_distribution_continuous(seeded_or_not=None):
    """Faz 0 item 2/3: compute_psi_summary psi_scalars (hafif) + psi_distribution (overlay) döndürür.
    Sürekli kolonda ±inf kenarları JSON-güvenli `null` (psycopg2 inf reddeder)."""
    stat = _cont({"0.01": 1.0, "0.25": 2.0, "0.5": 3.0, "0.75": 4.0, "0.99": 5.0})
    prod = [{"x": v} for v in (1, 2, 3, 4, 5) for _ in range(100)]
    out = compute_psi_summary({"x": stat}, prod, 5000, ALPHA)
    # scalars = per_feature psi'nin hafif projeksiyonu
    assert out["psi_scalars"]["x"] == out["psi_per_feature"]["x"]["psi"]
    dist = out["psi_distribution"]["x"]
    assert dist["kind"] == "continuous"
    assert dist["edges"][0] is None and dist["edges"][-1] is None      # ±inf → null (JSON-güvenli)
    assert all(isinstance(e, float) for e in dist["edges"][1:-1])       # iç kenarlar sonlu
    assert len(dist["ref_props"]) == len(dist["prod_counts"])           # kova hizası
    assert sum(dist["prod_counts"]) == dist["prod_total"] == 500


def test_distribution_categorical_has_other_bucket():
    stat = {"kind": "categorical", "value_counts_normalized": {"a": 0.5, "b": 0.5}, "n_unique": 2}
    prod = [{"c": v} for v in ("a", "b", "zzz") for _ in range(100)]
    out = compute_psi_summary({"c": stat}, prod, 1000, ALPHA)
    dist = out["psi_distribution"]["c"]
    assert dist["kind"] == "categorical"
    assert dist["categories"][-1] == "__other__"                       # unseen kovası
    assert dist["prod_counts"][-1] == 100                              # zzz → other
    assert dist["prod_total"] == 300


def test_degenerate_excluded_from_scalars_and_distribution():
    stat = _cont({"0.01": 5.0, "0.25": 5.0, "0.5": 5.0, "0.75": 5.0, "0.99": 5.0})   # sabit → dejenere
    out = compute_psi_summary({"x": stat}, [{"x": 5} for _ in range(10)], 1000, ALPHA)
    assert out["psi_scalars"] == {} and out["psi_distribution"] == {}  # ölçülemeyen dağılım/skaler taşımaz


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
