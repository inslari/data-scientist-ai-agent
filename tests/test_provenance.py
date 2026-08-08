"""Katman 2 Parça 3b — importance provenance + ağırlıklı PSI (saf, DB'siz).

Kilit: cycle-guard (bozuk/zincirli manifest sonsuz döngüye girmez) · terminal SIKI (raw ne producer
ise orphan, sessizce raw sayılmaz) · eşit bölme + attributed_ratio · importance-ağırlıklı psi_overall
+ fallback · degenerate_importance_mass (ölçülemeyenin model ağırlığı).
"""
import math

from src.services.monitoring.provenance import (
    attribute_importance, build_producer_map, resolve_to_raw)
from src.services.monitoring.psi import compute_psi_summary

ALPHA = 0.5


def _cont(q):
    return {"kind": "continuous", "mean": 0.0, "std": 1.0, "quantiles": dict(q), "n": 1000}


# ── provenance / resolve_to_raw ───────────────────────────────────────────────

def test_resolve_chain_to_raw():
    producer = {"x": ["y"], "y": ["r"]}                    # x←y←r (chained)
    assert resolve_to_raw("x", producer, {"r"}) == {"r"}


def test_cycle_guard_terminates_orphan():
    producer = {"a": ["b"], "b": ["a"]}                    # DÖNGÜ, hiçbiri raw değil
    assert resolve_to_raw("a", producer, {"r"}) == set()   # sonsuz döngü YOK → orphan (boş)


def test_terminal_strict_unknown_is_orphan():
    # ne raw ne producer'da → orphan (sessizce raw SAYILMAZ)
    assert resolve_to_raw("mystery", {}, {"r"}) == set()


def test_attribute_equal_split_and_ratio():
    manifest = [{"delta": {"added": ["ratio_a_b"]},
                 "frozen_choice": {"selection": [
                     {"output": "ratio_a_b", "numerator": "a", "denominator": "b"}]}}]
    fi = {"ratio_a_b": 10.0, "a": 5.0, "orphan_feat": 5.0}  # toplam 20
    raw_imp, ratio = attribute_importance(fi, manifest, ["a", "b"])
    # ratio_a_b(10) → a:5,b:5 (EŞİT) · a(5, raw identity) → a:5 · orphan_feat → orphan
    assert abs(raw_imp["a"] - 0.5) < 1e-9                   # (5+5)/20
    assert abs(raw_imp["b"] - 0.25) < 1e-9                  # 5/20
    assert abs(ratio - 0.75) < 1e-9                         # 15/20 atfedildi, %25 orphan


def test_attribute_onehot_and_count_from_params():
    manifest = [{"delta": {"added": ["city_NY", "city_LA", "count_grade"]}, "frozen_choice": {}}]
    ps = {"onehot_encode": {"columns": ["city"]}}
    fi = {"city_NY": 4.0, "city_LA": 2.0, "count_grade": 4.0}
    raw_imp, ratio = attribute_importance(fi, manifest, ["city", "grade"], ps)
    assert abs(raw_imp["city"] - 0.6) < 1e-9                # (4+2)/10
    assert abs(raw_imp["grade"] - 0.4) < 1e-9               # 4/10
    assert abs(ratio - 1.0) < 1e-9


# ── ağırlıklı psi_overall + degenerate_importance_mass ────────────────────────

def test_weighted_overall_differs_from_plain():
    s = _cont({"0.01": 1.0, "0.25": 2.0, "0.5": 3.0, "0.75": 4.0, "0.99": 5.0})
    prod = [{"a": 5.0, "b": 3.0} for _ in range(400)]       # a kenarda (yüksek psi), b ortada (düşük)
    per_feature = {"a": _cont(s["quantiles"]), "b": _cont(s["quantiles"])}
    plain = compute_psi_summary(per_feature, prod, 5000, ALPHA)
    weighted = compute_psi_summary(per_feature, prod, 5000, ALPHA, raw_importance={"a": 0.9, "b": 0.1})
    pa = plain["psi_per_feature"]["a"]["psi"]
    pb = plain["psi_per_feature"]["b"]["psi"]
    assert plain["psi_overall"] == round((pa + pb) / 2, 6)                 # düz
    assert weighted["psi_overall"] == round((0.9 * pa + 0.1 * pb) / 1.0, 6)  # ağırlık toplamı 1
    assert weighted["psi_per_feature"]["a"]["importance_weight"] == 0.9
    assert weighted["psi_max"] == plain["psi_max"]                          # max ağırlıktan bağımsız


def test_degenerate_importance_mass_recorded():
    per_feature = {
        "const": _cont({"0.01": 5.0, "0.25": 5.0, "0.5": 5.0, "0.75": 5.0, "0.99": 5.0}),  # dejenere
        "good": _cont({"0.01": 1.0, "0.25": 2.0, "0.5": 3.0, "0.75": 4.0, "0.99": 5.0}),
    }
    prod = [{"const": 5.0, "good": 3.0} for _ in range(400)]
    out = compute_psi_summary(per_feature, prod, 5000, ALPHA, raw_importance={"const": 0.4, "good": 0.6})
    # "PSI 0.03 iyi" derken ölçülemeyen const'ın modelde %40 ağırlığı olduğu GÖRÜNÜR
    assert out["psi_degenerate_count"] == 1
    assert out["degenerate_importance_mass"] == 0.4
    assert out["psi_per_feature"]["const"]["degenerate"] is True
    assert out["psi_per_feature"]["const"]["importance_weight"] == 0.4
    # psi_overall yalnız ölçülen "good" üzerinden (renormalize) → good'un psi'si
    assert math.isfinite(out["psi_overall"])
    assert out["psi_overall"] == out["psi_per_feature"]["good"]["psi"]


def test_no_importance_fallback_plain():
    out = compute_psi_summary({"a": _cont({"0.01": 1.0, "0.25": 2.0, "0.5": 3.0, "0.75": 4.0, "0.99": 5.0})},
                              [{"a": 3.0} for _ in range(300)], 3000, ALPHA)   # raw_importance yok
    assert out["psi_per_feature"]["a"]["importance_weight"] is None
    assert out["degenerate_importance_mass"] is None


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
