"""Offline tests for the autonomous leakage guards (no LLM/DB):

A) `_single_feature_target_lock` / `_detect_leakage` — catches NON-LINEAR single-feature
   target lock (e.g. ALERT_ID: -1 vs positive ids) that linear correlation misses.
B) auto-feature own-goal fix — `_add_missing_indicators` / `_count_encode_tt` /
   `_groupby_deviation_tt` must NOT derive features from leaky/target/identifier columns.
C) `detect_target_lock` — binned target-lock check must be immune to CLASS IMBALANCE
   (a rare-event base rate must not by itself look like a lock) while still catching
   a real lock, in both imbalanced and balanced data.
"""

import unittest

import numpy as np
import pandas as pd

from src.services.feature_builder import (
    _single_feature_target_lock,
    _binned_max_purity,
    _detect_leakage,
    _add_missing_indicators,
    _count_encode_tt,
    _groupby_deviation_tt,
    _leakage_exclude_cols,
)
from src.services.leakage_detector import detect_target_lock
from src.services.critic_advisor import _diagnose_suspicious_performance, advise


def _alert_id_frame(n=400, seed=0):
    """ALERT_ID = -1 for non-fraud, a positive alert id for fraud → threshold-perfect
    separator with only MODERATE linear correlation (what corr-based checks miss)."""
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n))
    alert_id = pd.Series(np.where(y == 1, rng.integers(100, 500, n), -1))
    amount = pd.Series(rng.normal(50, 10, n))
    return pd.DataFrame({"ALERT_ID": alert_id, "amount": amount}), y


class SingleFeatureTargetLockTest(unittest.TestCase):
    def test_catches_nonlinear_separator(self):
        df, y = _alert_id_frame()
        # Linear correlation is only moderate → the OLD path would not drop it.
        self.assertLess(abs(df["ALERT_ID"].corr(y)), 0.95)
        drop, _ = _single_feature_target_lock(df, y)
        self.assertIn("ALERT_ID", [d["column"] for d in drop])

    def test_noise_feature_not_flagged(self):
        df, y = _alert_id_frame()
        drop, suspect = _single_feature_target_lock(df, y)
        flagged = {d["column"] for d in drop} | {s["column"] for s in suspect}
        self.assertNotIn("amount", flagged)

    def test_detect_leakage_drops_alert_id(self):
        df, y = _alert_id_frame()
        cleaned, report = _detect_leakage(df.copy(), y)
        self.assertNotIn("ALERT_ID", cleaned.columns)
        self.assertIn("ALERT_ID", [d["column"] for d in report["dropped"]])

    def test_small_sample_is_noop(self):
        df, y = _alert_id_frame(n=20)
        self.assertEqual(_single_feature_target_lock(df, y), ([], []))

    def test_regression_monotonic_leak(self):
        # y continuous; a feature that is a monotonic transform of y → |Spearman|≈1
        n = 300
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(0, 1, n))
        df = pd.DataFrame({"leak": np.exp(y), "noise": rng.normal(0, 1, n)})
        drop, _ = _single_feature_target_lock(df, y)
        names = [d["column"] for d in drop]
        self.assertIn("leak", names)
        self.assertNotIn("noise", names)


class ContinuousFeatureNotDroppedTest(unittest.TestCase):
    """Cardinality 'ID-like' rule must NOT nuke continuous float/metric features
    (creditcard V1-V28 regression: every PCA value unique → wrongly dropped → only
    Amount left → AUC collapsed to 0.73)."""

    def test_float_pca_features_survive_int_id_dropped(self):
        rng = np.random.default_rng(0)
        n = 2000
        y = pd.Series(rng.integers(0, 2, n))
        df = pd.DataFrame({
            "V1": rng.normal(0, 1, n),          # continuous float, near-unique
            "V2": rng.normal(0, 1, n),
            "Amount": rng.exponential(50, n),
            "row_id": np.arange(n),             # genuine integer identifier
        })
        clean, report = _detect_leakage(df.copy(), y)
        for c in ("V1", "V2", "Amount"):
            self.assertIn(c, clean.columns)
        self.assertNotIn("row_id", clean.columns)
        self.assertIn("row_id", [d["column"] for d in report["dropped"]])

    def test_metric_semantic_exempts_even_if_int(self):
        # A high-cardinality integer column labeled 'metric' is a measurement, not an ID
        rng = np.random.default_rng(1)
        n = 1500
        y = pd.Series(rng.integers(0, 2, n))
        df = pd.DataFrame({"sensor_count": rng.integers(0, 100000, n)})

        class _Meta:
            columns = {"sensor_count": {"semantic_type": "metric"}}

        clean, _ = _detect_leakage(df.copy(), y, meta=_Meta())
        self.assertIn("sensor_count", clean.columns)


class AutoFeatureExcludeTest(unittest.TestCase):
    def test_missing_indicator_respects_exclude(self):
        # ALERT_TYPE present only for the positive class → its _isna would be a perfect leak
        s = pd.Series(["fan_in", None, "cycle", None])
        df = pd.DataFrame({"ALERT_TYPE": s, "amount": [1.0, 2.0, 3.0, 4.0]})
        self.assertEqual(_add_missing_indicators(df.copy()), ["ALERT_TYPE_isna"])
        self.assertEqual(_add_missing_indicators(df.copy(), exclude={"ALERT_TYPE"}), [])

    def test_count_encode_respects_exclude(self):
        Xtr = pd.DataFrame({"cat": ["a", "b", "a", "b"], "leakcat": ["x", "y", "x", "y"]})
        Xte = Xtr.copy()
        _, _, added = _count_encode_tt(Xtr.copy(), Xte.copy(), exclude={"leakcat"})
        self.assertIn("count_cat", added)
        self.assertNotIn("count_leakcat", added)

    def test_groupby_deviation_respects_exclude(self):
        Xtr = pd.DataFrame({
            "grp": ["a", "b", "a", "b", "a", "b"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "leakval": [0.0, 9.0, 0.0, 9.0, 0.0, 9.0],
        })
        Xte = Xtr.copy()
        _, _, added = _groupby_deviation_tt(Xtr.copy(), Xte.copy(), exclude={"leakval"})
        # leakval must be used neither as a numeric input nor produce a dev_ column
        self.assertFalse(any("leakval" in c for c in added))

    def test_exclude_set_builder(self):
        plan = {"target": {"column_name": "IS_FRAUD"}}

        class _Meta:
            columns = {
                "TX_ID": {"semantic_type": "identifier"},
                "TIMESTAMP": {"semantic_type": "timestamp"},
                "TX_AMOUNT": {"semantic_type": "numeric"},
            }

        dlr = {"concerns": [{"column_name": "ALERT_ID"}]}
        excl = _leakage_exclude_cols(plan, _Meta(), dlr)
        self.assertEqual(excl, {"IS_FRAUD", "TX_ID", "TIMESTAMP", "ALERT_ID"})
        self.assertNotIn("TX_AMOUNT", excl)


class MulticlassTargetLockTest(unittest.TestCase):
    """C — multiclass single-feature target lock via binned purity."""

    def _multiclass_frame(self, n=450, seed=3):
        rng = np.random.default_rng(seed)
        y = pd.Series(rng.integers(0, 3, n))           # 3 classes
        # leak: feature value range uniquely encodes the class → bin purity ~1
        leak = pd.Series(y * 1000 + rng.normal(0, 5, n))
        noise = pd.Series(rng.normal(0, 1, n))
        return pd.DataFrame({"leak": leak, "noise": noise}), y

    def test_binned_purity_perfect_separator(self):
        df, y = self._multiclass_frame()
        self.assertGreaterEqual(_binned_max_purity(df["leak"], y), 0.99)
        self.assertLess(_binned_max_purity(df["noise"], y), 0.9)

    def test_multiclass_drops_leak(self):
        df, y = self._multiclass_frame()
        drop, _ = _single_feature_target_lock(df, y)
        names = [d["column"] for d in drop]
        self.assertIn("leak", names)
        self.assertNotIn("noise", names)


class SuspiciousPerformanceTest(unittest.TestCase):
    """D — critic 'too-good' leakage alarm."""

    def test_near_perfect_with_dominant_feature(self):
        d = _diagnose_suspicious_performance(
            "classification",
            {"roc_auc": 0.999},
            {"leakage_suspects": [{"column": "ALERT_ID", "severity": "high"}]},
        )
        self.assertIsNotNone(d)
        self.assertEqual(d["type"], "suspicious_performance")
        self.assertEqual(d["severity"], 9)
        self.assertEqual(d["action"], "feature_plan_node")
        self.assertIn("ALERT_ID", d["leakage_suspects"])

    def test_dominant_feature_without_near_perfect(self):
        d = _diagnose_suspicious_performance(
            "classification",
            {"roc_auc": 0.94},
            {"leakage_suspects": [{"column": "X", "severity": "high"}]},
        )
        self.assertIsNotNone(d)
        self.assertEqual(d["severity"], 8)

    def test_combined_suspect_names_split(self):
        d = _diagnose_suspicious_performance(
            "classification",
            {"roc_auc": 0.999},
            {"leakage_suspects": [{"column": "a + b", "severity": "medium"}]},
        )
        self.assertEqual(set(d["leakage_suspects"]), {"a", "b"})

    def test_clean_model_not_flagged(self):
        d = _diagnose_suspicious_performance(
            "classification", {"roc_auc": 0.88}, {"leakage_suspects": []},
        )
        self.assertIsNone(d)

    def test_advise_routes_and_forces_drops(self):
        state = {
            "evaluation_report": {
                "task": "classification",
                "test_metrics": {"roc_auc": 0.999},
                "feature_importance_summary": {
                    "leakage_suspects": [{"column": "ALERT_ID", "severity": "high"}],
                },
            },
            "retry_count": 0, "max_retries": 3, "retry_history": [],
        }
        decision = advise(state)
        self.assertTrue(decision["should_retry"])
        self.assertEqual(decision["next_node"], "feature_plan_node")
        self.assertEqual(decision["primary_type"], "suspicious_performance")
        self.assertIn("ALERT_ID", decision["forced_drops"])


class TargetLockImbalanceTest(unittest.TestCase):
    """`detect_target_lock` must measure the FEATURE, not the class imbalance.

    The old rule flagged a bin whose majority-class share was >= 0.97. In a rare-event
    problem (fraud at ~2%) every bin is trivially ~98% non-fraud, so features with zero
    relationship to the target were reported as leakage. The check now scores a bin by
    how far it moves a class BEYOND its global prior, so it stays honest at any base rate.
    """

    IMPORTANT = {"feat": 0.40}  # above the 0.05 importance gate

    def _imbalanced(self, n=6000, rate=0.02, seed=0):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < rate).astype(int)
        return rng, y

    def test_noise_not_flagged_in_rare_event_data(self):
        """A feature with NO relationship to a 2%-positive target must stay clean."""
        rng, y = self._imbalanced()
        X = pd.DataFrame({"feat": rng.normal(size=len(y))})
        self.assertEqual(detect_target_lock(X, y, self.IMPORTANT), [])

    def test_strong_but_legitimate_signal_not_flagged(self):
        """A genuinely predictive feature is NOT a lock: it shifts the odds, it does not
        make the outcome certain (fraud's `amount` / `account_age_days` pattern)."""
        rng, y = self._imbalanced()
        # positives skew high, but no bin gets anywhere near pure
        feat = rng.normal(size=len(y)) + y * 1.5
        X = pd.DataFrame({"feat": feat})
        self.assertEqual(detect_target_lock(X, y, self.IMPORTANT), [])

    def test_real_leak_still_caught_in_rare_event_data(self):
        """A target-derived column must still be flagged despite the 2% base rate.

        Uses a realistic sample size (bins of ~6k rows, like the fraud dataset): a bin
        with zero positives is only strong evidence of a lock when the bin is big enough
        for zero to be surprising. At small n the check deliberately stays silent — the
        AUC-based `_single_feature_target_lock` guard covers that regime instead.
        """
        rng, y = self._imbalanced(n=60000)
        X = pd.DataFrame({"feat": y + rng.normal(0, 0.05, len(y))})
        found = detect_target_lock(X, y, self.IMPORTANT)
        self.assertEqual([c["feature"] for c in found], ["feat"])
        # Sample-size ceiling: a zero-positive bin of ~6k rows tops out near 0.97 —
        # comfortably over the 0.95 gate, while legitimate features stay under 0.70.
        self.assertGreaterEqual(found[0]["purity_gain"], 0.95)

    def test_real_leak_caught_at_moderate_imbalance(self):
        """15% positives: the leak makes a whole bin pure POSITIVE — unambiguous lock."""
        rng = np.random.default_rng(7)
        y = (rng.random(6000) < 0.15).astype(int)
        X = pd.DataFrame({"feat": y + rng.normal(0, 0.05, len(y))})
        found = detect_target_lock(X, y, self.IMPORTANT)
        self.assertEqual([c["feature"] for c in found], ["feat"])

    def test_small_bin_zero_positives_is_not_a_lock(self):
        """Regression guard for the subtle failure the fix had to survive: in rare-event
        data a merely-predictive feature empties its bottom bin of positives. Raw purity
        reads 1.0 ('certain'), but with only ~12 expected positives per bin that is a
        resolution limit, not determinism. Must NOT be reported as leakage."""
        rng, y = self._imbalanced(n=6000, rate=0.02)
        X = pd.DataFrame({"feat": rng.normal(size=len(y)) + y * 1.5})
        self.assertEqual(detect_target_lock(X, y, self.IMPORTANT), [])

    def test_balanced_data_lock_still_caught(self):
        """Balanced target: the classic near-pure bin must still be reported.
        Values are kept DISTINCT — identical values make qcut collapse bins and dilute
        the lock, which would hide it from any binned check."""
        rng = np.random.default_rng(1)
        n = 6000
        y = rng.integers(0, 2, n)
        feat = rng.normal(size=n)
        neg = np.flatnonzero(y == 0)
        picked = neg[rng.random(neg.size) < 0.30]
        feat[picked] = rng.uniform(5.0, 6.0, picked.size)  # top decile → all class 0
        found = detect_target_lock(pd.DataFrame({"feat": feat}), y, self.IMPORTANT)
        self.assertEqual([c["feature"] for c in found], ["feat"])

    def test_low_importance_feature_is_ignored(self):
        rng, y = self._imbalanced()
        X = pd.DataFrame({"feat": y + rng.normal(0, 0.05, len(y))})
        self.assertEqual(detect_target_lock(X, y, {"feat": 0.01}), [])

    def test_single_class_target_is_noop(self):
        """No uncertainty to lock → nothing to report (and no divide-by-zero)."""
        rng = np.random.default_rng(2)
        y = np.zeros(500, dtype=int)
        X = pd.DataFrame({"feat": rng.normal(size=500)})
        self.assertEqual(detect_target_lock(X, y, self.IMPORTANT), [])

    def test_regression_branch_unaffected(self):
        """Variance-collapse branch is already normalized by global std — untouched."""
        rng = np.random.default_rng(3)
        n = 3000
        feat = rng.normal(size=n)
        y = rng.normal(0, 10, n)
        y[feat > np.quantile(feat, 0.9)] = 7.0  # variance collapses in the top decile
        found = detect_target_lock(pd.DataFrame({"feat": feat}), y, self.IMPORTANT,
                                   intent="regression")
        self.assertEqual([c["feature"] for c in found], ["feat"])


if __name__ == "__main__":
    unittest.main()
