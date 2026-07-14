"""Job 2 — PR-AUC as the selection metric under extreme imbalance (offline, no LLM/DB).

Why this exists: model selection used to run on `roc_auc` at ANY class balance. Under
extreme imbalance the huge negative mass barely moves the FPR, so ROC is flattering and
poor at separating candidates — it selected a model that was measurably worse at the
question the business actually asks ("of the positives I flag, how many are real").

Measured on the reference datasets: ROC and PR-AUC only disagree on the winner BELOW ~5%
minority share (credit 21.8% and churn 20.4% → same winner; e-commerce fraud 2.2% and
ULB 0.17% → winner changes). Hence the switch threshold is 5%, and the balanced /
moderately-imbalanced datasets must keep their existing roc_auc behaviour untouched.
"""

import unittest

import numpy as np

from src.core.config import settings
from src.services.ml_trainer import _select_scoring
from src.services.evaluator import _analyze_overfitting


def _y(minority_share: float, n: int = 20000) -> np.ndarray:
    k = int(round(n * minority_share))
    return np.array([1] * k + [0] * (n - k))


class SelectScoringTest(unittest.TestCase):
    """The scoring metric is chosen from the DATA at runtime — no dataset-specific rule."""

    def test_extreme_imbalance_switches_to_pr_auc(self):
        for share in (0.0017, 0.0221, 0.049):  # ULB, e-commerce fraud, just under the gate
            scoring, metric = _select_scoring(_y(share), is_binary=True)
            self.assertEqual(scoring, "average_precision", f"share={share}")
            self.assertEqual(metric, "pr_auc")

    def test_reference_datasets_keep_roc_auc(self):
        """credit (21.8%) and churn (20.4%) must be untouched — measured: same winner
        under either metric, so switching would move reference scores for nothing."""
        for share in (0.204, 0.218, 0.5):
            scoring, metric = _select_scoring(_y(share), is_binary=True)
            self.assertEqual(scoring, "roc_auc", f"share={share}")
            self.assertEqual(metric, "roc_auc")

    def test_threshold_boundary_is_exclusive(self):
        """Exactly at the threshold we stay on ROC (switch is strictly below)."""
        scoring, _ = _select_scoring(_y(settings.PR_AUC_SCORING_THRESHOLD), is_binary=True)
        self.assertEqual(scoring, "roc_auc")

    def test_multiclass_untouched(self):
        scoring, metric = _select_scoring(np.array([0, 1, 2] * 100), is_binary=False)
        self.assertEqual(scoring, "roc_auc_ovr_weighted")
        self.assertEqual(metric, "roc_auc_ovr")

    def test_threshold_is_configurable_not_hardcoded(self):
        self.assertIsInstance(settings.PR_AUC_SCORING_THRESHOLD, float)
        self.assertEqual(settings.PR_AUC_SCORING_THRESHOLD, 0.05)


class OverfittingMetricMatchTest(unittest.TestCase):
    """The gap is only meaningful between two measurements of the SAME metric.

    Once CV can run on PR-AUC, comparing CV-PR-AUC against test-ROC would invent a
    difference out of thin air and report a fake overfitting verdict.
    """

    TRAIN = {"roc_auc": 0.99, "pr_auc": 0.95, "accuracy": 0.98}
    TEST = {"roc_auc": 0.93, "pr_auc": 0.71, "accuracy": 0.96}

    def test_pr_auc_cv_compares_against_pr_auc_test(self):
        out = _analyze_overfitting(self.TRAIN, self.TEST, cv_score=0.80, cv_metric="pr_auc")
        self.assertAlmostEqual(out["gap"], 0.80 - 0.71, places=4)

    def test_roc_cv_compares_against_roc_test(self):
        out = _analyze_overfitting(self.TRAIN, self.TEST, cv_score=0.9791, cv_metric="roc_auc")
        self.assertAlmostEqual(out["gap"], 0.9791 - 0.93, places=4)

    def test_mismatched_metrics_would_have_lied(self):
        """Guard the actual bug: PR-AUC CV vs ROC test would look like a NEGATIVE gap."""
        matched = _analyze_overfitting(self.TRAIN, self.TEST, cv_score=0.80, cv_metric="pr_auc")
        naive_mismatch = 0.80 - self.TEST["roc_auc"]  # what the old code would have done
        self.assertLess(naive_mismatch, 0)
        self.assertGreater(matched["gap"], 0)

    def test_default_metric_is_roc_auc(self):
        out = _analyze_overfitting(self.TRAIN, self.TEST, cv_score=0.95)
        self.assertAlmostEqual(out["gap"], 0.95 - 0.93, places=4)


if __name__ == "__main__":
    unittest.main()
