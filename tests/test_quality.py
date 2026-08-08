"""Katman 2 Parça 4 — veri kalitesi per-row bayrak + PSI çapraz-kontrol (saf, DB'siz).

Kilit: check_raw_input row_flags satır-hizalı + doğru · PSI kategorik "other" kütlesi == kalite
unseen sayısı (iki BAĞIMSIZ yol aynı kriteri kullanır → çapraz doğrulama; tutarsızlık = bug işareti).
"""
import pandas as pd

from src.services.inference.orchestrator import check_raw_input
from src.services.monitoring.psi import _categorical_bins

_SIG = {
    "required": ["cat", "num"],
    "columns": {
        "cat": {"kind": "categorical", "known_values": ["a", "b"]},
        "num": {"kind": "continuous", "range": [0.0, 10.0]},
    },
}


def test_row_flags_aligned_and_correct():
    raw = pd.DataFrame({"cat": ["a", "b", "z", "a", "z"],       # z = unseen (idx 2,4)
                        "num": [1, 5, 20, -3, 5]})              # 20,-3 aralık-dışı (idx 2,3)
    rf = check_raw_input(raw, _SIG)["row_flags"]
    assert rf["has_unseen"] == [False, False, True, False, True]
    assert rf["has_out_of_range"] == [False, False, True, True, False]


def test_no_warnings_all_false():
    raw = pd.DataFrame({"cat": ["a", "b"], "num": [1, 5]})
    rf = check_raw_input(raw, _SIG)["row_flags"]
    assert rf["has_unseen"] == [False, False]
    assert rf["has_out_of_range"] == [False, False]


def test_cross_check_unseen_rate_equals_psi_other_mass():
    """Kalite unseen sayısı == PSI kategorik 'other' kova sayısı (iki bağımsız yol, aynı kriter)."""
    raw = pd.DataFrame({"cat": ["a", "b", "z", "q", "a", "b"], "num": [1, 2, 3, 4, 5, 6]})
    rf = check_raw_input(raw, _SIG)["row_flags"]
    quality_unseen = sum(rf["has_unseen"])                     # z,q → 2

    # PSI yolu: aynı vocab'la kategorik kovala, "other" (son kova) sayısı
    stat = {"kind": "categorical", "value_counts_normalized": {"a": 0.5, "b": 0.5}, "n_unique": 2}
    (cats, _props, _tail), _reason = _categorical_bins(stat)
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats) + 1                                          # +1 = "other"
    counts = [0] * k
    for v in raw["cat"]:
        counts[idx.get(str(v), k - 1)] += 1
    psi_other = counts[-1]

    assert quality_unseen == psi_other == 2                   # TUTARLI (tutarsızlık = bug işareti)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
