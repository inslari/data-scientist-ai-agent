"""Offline tests for the quality-gate domain-leakage warning (drop-awareness).

The gate must not warn about high-confidence leakage suspects that the feature
builder already dropped (recorded in feature_report.leakage_dropped) — they never
entered the model, so "predictions may be unreliable" would be misleading.
"""

import unittest

from src.services.quality_gate import evaluate_quality_gate


def _base_state(concerns, dropped):
    """A clean classification run whose ONLY possible warning is domain leakage."""
    return {
        "evaluation_report": {
            "task": "binary_classification",
            "test_metrics": {"roc_auc": 0.87},     # healthy → no perf failure
            "overfitting": {"gap": 0.0, "status": "normal"},
            "reliability_score": {"score": 80, "grade": "B"},
            "feature_importance_summary": {},
            "target_lock_leakage": [],
        },
        "domain_leakage_report": {"concerns": concerns},
        "feature_report": {"leakage_dropped": dropped},
        # no target_col → target sanity-check skipped
    }


def _high(col):
    return {"column_name": col, "confidence": "high", "relationship_type": "identifier",
            "reasoning": "identifier-like"}


class DomainLeakageDropAwarenessTest(unittest.TestCase):
    def test_dropped_concerns_do_not_warn(self):
        # RowNumber/CustomerId/Surname flagged high AND dropped → no warning, pass.
        state = _base_state(
            concerns=[_high("RowNumber"), _high("CustomerId"), _high("Surname")],
            dropped=[{"column": "RowNumber"}, {"column": "CustomerId"}, {"column": "Surname"}],
        )
        result = evaluate_quality_gate(state)
        codes = {w["code"] for w in result["warnings"]}
        self.assertNotIn("domain_leakage_high", codes)
        self.assertEqual(result["status"], "pass")

    def test_concern_still_in_model_warns(self):
        # Flagged high but NOT dropped → genuine concern → warn.
        state = _base_state(concerns=[_high("balance_at_churn")], dropped=[])
        result = evaluate_quality_gate(state)
        leak = [w for w in result["warnings"] if w["code"] == "domain_leakage_high"]
        self.assertEqual(len(leak), 1)
        self.assertIn("balance_at_churn", leak[0]["message"])
        self.assertEqual(result["status"], "warn")

    def test_partial_only_warns_for_undropped(self):
        state = _base_state(
            concerns=[_high("CustomerId"), _high("leaky_feature")],
            dropped=[{"column": "CustomerId"}],
        )
        result = evaluate_quality_gate(state)
        leak = [w for w in result["warnings"] if w["code"] == "domain_leakage_high"][0]
        self.assertIn("leaky_feature", leak["message"])
        self.assertNotIn("CustomerId", leak["message"])      # dropped → excluded
        self.assertEqual([c["column_name"] for c in leak["details"]], ["leaky_feature"])

    def test_low_confidence_never_warns(self):
        state = _base_state(
            concerns=[{"column_name": "x", "confidence": "low"}],
            dropped=[],
        )
        result = evaluate_quality_gate(state)
        self.assertNotIn("domain_leakage_high", {w["code"] for w in result["warnings"]})
        self.assertEqual(result["status"], "pass")

    def test_missing_feature_report_is_safe(self):
        # No feature_report at all → falls back to warning (nothing known dropped).
        state = _base_state(concerns=[_high("CustomerId")], dropped=[])
        state.pop("feature_report")
        result = evaluate_quality_gate(state)
        self.assertIn("domain_leakage_high", {w["code"] for w in result["warnings"]})


if __name__ == "__main__":
    unittest.main()
