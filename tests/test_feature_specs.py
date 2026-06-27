"""Offline tests for the agent-decided feature-spec path (no LLM/DB).

Covers: spec parsing from tool args, the deterministic validating executor in
the feature builder, schema roundtrip, and transcript extraction.
"""

import types
import unittest

import numpy as np
import pandas as pd

from src.domains.schemas import (
    FeatureSpec, ExpertFinding, MerchantAnalysis,
    FEATURE_OP_SPEC, validate_feature_spec_shape,
)
from src.domains.financial import agents
from src.services.feature_builder import (
    _apply_feature_specs, _apply_plan_derived_features, _FEATURE_OPS,
)


def _df():
    return pd.DataFrame({
        "amount":  [10.0, 20.0, 30.0, 40.0],
        "balance": [100.0, 0.0, 50.0, 25.0],
        "is_fraud": [0, 1, 0, 1],
    })


def _state_with_specs(specs):
    analysis = MerchantAnalysis(
        pack_name="financial", primary_expert="fraud", council=["fraud"],
        findings=[ExpertFinding(expert="fraud", summary="s", feature_specs=specs)],
    )
    return {"merchant_analysis": analysis.to_dict()}


class ParseSpecTest(unittest.TestCase):
    def test_valid_ratio(self):
        s = agents._parse_feature_spec({"name": "r", "op": "ratio", "inputs": ["amount", "balance"]})
        self.assertIsNotNone(s)
        self.assertEqual(s.op, "ratio")

    def test_unknown_op_rejected(self):
        self.assertIsNone(agents._parse_feature_spec({"name": "x", "op": "divide", "inputs": ["a", "b"]}))

    def test_wrong_arity_rejected(self):
        # ratio needs 2-3 inputs
        self.assertIsNone(agents._parse_feature_spec({"name": "x", "op": "ratio", "inputs": ["a"]}))
        # zscore needs 1
        self.assertIsNone(agents._parse_feature_spec({"name": "x", "op": "zscore", "inputs": ["a", "b"]}))

    def test_empty_name_rejected(self):
        self.assertIsNone(agents._parse_feature_spec({"name": "", "op": "log", "inputs": ["a"]}))

    def test_three_input_ratio_accepted(self):
        s = agents._parse_feature_spec({"name": "r3", "op": "ratio", "inputs": ["a", "b", "c"]})
        self.assertIsNotNone(s)
        self.assertEqual(len(s.inputs), 3)

    def test_power_requires_one_param(self):
        # power needs exactly 1 param (exponent)
        self.assertIsNone(agents._parse_feature_spec({"name": "p", "op": "power", "inputs": ["a"]}))
        s = agents._parse_feature_spec({"name": "p", "op": "power", "inputs": ["a"], "params": [2]})
        self.assertIsNotNone(s)
        self.assertEqual(s.params, [2.0])

    def test_weighted_sum_param_per_input(self):
        # weighted_sum: one weight per input
        self.assertIsNone(agents._parse_feature_spec(
            {"name": "w", "op": "weighted_sum", "inputs": ["a", "b"], "params": [1.0]}))
        s = agents._parse_feature_spec(
            {"name": "w", "op": "weighted_sum", "inputs": ["a", "b"], "params": [0.5, 0.5]})
        self.assertIsNotNone(s)

    def test_non_numeric_params_rejected(self):
        self.assertIsNone(agents._parse_feature_spec(
            {"name": "p", "op": "power", "inputs": ["a"], "params": ["x"]}))


class ExecutorTest(unittest.TestCase):
    def test_applies_valid_ratio_zero_safe(self):
        df = _df()
        spec = FeatureSpec(name="amt_to_bal", op="ratio", inputs=["amount", "balance"])
        added = _apply_feature_specs(df, _state_with_specs([spec]), disallowed=set())
        self.assertEqual(added, ["amt_to_bal"])
        # row 1 has balance=0 → eps-guarded, finite (no inf/NaN)
        self.assertTrue(np.isfinite(df["amt_to_bal"]).all())
        self.assertAlmostEqual(df["amt_to_bal"].iloc[0], 10.0 / 100.0, places=4)

    def test_zscore_arity_one(self):
        df = _df()
        spec = FeatureSpec(name="amt_z", op="zscore", inputs=["amount"])
        added = _apply_feature_specs(df, _state_with_specs([spec]), disallowed=set())
        self.assertEqual(added, ["amt_z"])
        self.assertAlmostEqual(df["amt_z"].mean(), 0.0, places=6)

    def test_missing_column_skipped(self):
        df = _df()
        spec = FeatureSpec(name="x", op="ratio", inputs=["amount", "nonexistent"])
        added = _apply_feature_specs(df, _state_with_specs([spec]), disallowed=set())
        self.assertEqual(added, [])
        self.assertNotIn("x", df.columns)

    def test_leaky_input_skipped(self):
        df = _df()
        # is_fraud is the target → must never become a feature input
        spec = FeatureSpec(name="leaky", op="ratio", inputs=["is_fraud", "amount"])
        added = _apply_feature_specs(df, _state_with_specs([spec]), disallowed={"is_fraud"})
        self.assertEqual(added, [])
        self.assertNotIn("leaky", df.columns)

    def test_no_analysis_is_noop(self):
        df = _df()
        self.assertEqual(_apply_feature_specs(df, {}, disallowed=set()), [])

    def test_duplicate_name_not_overwritten(self):
        df = _df()
        spec = FeatureSpec(name="amount", op="zscore", inputs=["amount"])  # collides with existing col
        added = _apply_feature_specs(df, _state_with_specs([spec]), disallowed=set())
        self.assertEqual(added, [])


class NewOpsExecutorTest(unittest.TestCase):
    def _apply(self, spec):
        df = _df()
        added = _apply_feature_specs(df, _state_with_specs([spec]), disallowed=set())
        return df, added

    def test_power_squared(self):
        df, added = self._apply(FeatureSpec(name="amt_sq", op="power", inputs=["amount"], params=[2.0]))
        self.assertEqual(added, ["amt_sq"])
        self.assertAlmostEqual(df["amt_sq"].iloc[0], 100.0, places=4)

    def test_power_exponent_clamped(self):
        # exponent 99 is clamped into _POWER_EXP_BOUNDS → stays finite, no overflow
        df, added = self._apply(FeatureSpec(name="amt_big", op="power", inputs=["amount"], params=[99.0]))
        self.assertEqual(added, ["amt_big"])
        self.assertTrue(np.isfinite(df["amt_big"]).all())

    def test_weighted_sum(self):
        df, added = self._apply(FeatureSpec(
            name="combo", op="weighted_sum", inputs=["amount", "balance"], params=[1.0, 2.0]))
        self.assertEqual(added, ["combo"])
        self.assertAlmostEqual(df["combo"].iloc[0], 10.0 + 2.0 * 100.0, places=4)

    def test_three_input_product(self):
        df, added = self._apply(FeatureSpec(name="prod3", op="product", inputs=["amount", "balance", "amount"]))
        self.assertEqual(added, ["prod3"])
        self.assertAlmostEqual(df["prod3"].iloc[0], 10.0 * 100.0 * 10.0, places=4)

    def test_three_input_ratio_zero_safe(self):
        df, added = self._apply(FeatureSpec(name="ratio3", op="ratio", inputs=["amount", "balance", "balance"]))
        self.assertEqual(added, ["ratio3"])
        self.assertTrue(np.isfinite(df["ratio3"]).all())  # balance=0 row → eps-guarded

    def test_normalize_to_unit_range(self):
        df, added = self._apply(FeatureSpec(name="amt_n", op="normalize", inputs=["amount"]))
        self.assertEqual(added, ["amt_n"])
        self.assertAlmostEqual(df["amt_n"].min(), 0.0, places=6)
        self.assertAlmostEqual(df["amt_n"].max(), 1.0, places=6)


class PlanDerivedFeaturesTest(unittest.TestCase):
    def test_plan_derived_applied(self):
        df = _df()
        plan = {"derived_features": [
            {"name": "amt_to_bal", "op": "ratio", "inputs": ["amount", "balance"]},
        ]}
        added = _apply_plan_derived_features(df, plan, disallowed=set())
        self.assertEqual(added, ["amt_to_bal"])
        self.assertIn("amt_to_bal", df.columns)

    def test_plan_derived_leakage_guard(self):
        df = _df()
        plan = {"derived_features": [
            {"name": "leaky", "op": "ratio", "inputs": ["is_fraud", "amount"]},
        ]}
        added = _apply_plan_derived_features(df, plan, disallowed={"is_fraud"})
        self.assertEqual(added, [])
        self.assertNotIn("leaky", df.columns)

    def test_plan_derived_empty_noop(self):
        self.assertEqual(_apply_plan_derived_features(_df(), {}, disallowed=set()), [])


class PaletteTest(unittest.TestCase):
    def test_all_ops_callable_and_in_spec(self):
        for op, fn in _FEATURE_OPS.items():
            self.assertTrue(callable(fn))
            self.assertIn(op, FEATURE_OP_SPEC)

    def test_spec_and_ops_in_sync(self):
        self.assertEqual(set(_FEATURE_OPS.keys()), set(FEATURE_OP_SPEC.keys()))

    def test_validate_shape(self):
        self.assertTrue(validate_feature_spec_shape("ratio", 2, 0))
        self.assertTrue(validate_feature_spec_shape("ratio", 3, 0))
        self.assertFalse(validate_feature_spec_shape("ratio", 4, 0))
        self.assertTrue(validate_feature_spec_shape("power", 1, 1))
        self.assertFalse(validate_feature_spec_shape("power", 1, 0))
        self.assertTrue(validate_feature_spec_shape("weighted_sum", 3, 3))
        self.assertFalse(validate_feature_spec_shape("weighted_sum", 3, 2))


class SchemaRoundtripTest(unittest.TestCase):
    def test_feature_specs_survive_roundtrip(self):
        a = MerchantAnalysis(
            pack_name="financial", primary_expert="fraud", council=["fraud"],
            findings=[ExpertFinding(expert="fraud", summary="s", feature_specs=[
                FeatureSpec(name="amt_z", op="zscore", inputs=["amount"], rationale="spread"),
            ])],
        )
        b = MerchantAnalysis.from_dict(a.to_dict())
        self.assertEqual(b.findings[0].feature_specs[0].name, "amt_z")
        self.assertEqual(b.findings[0].feature_specs[0].op, "zscore")


class ExtractionTest(unittest.TestCase):
    def test_extract_picks_up_propose_feature(self):
        def _msg(tcs):
            return types.SimpleNamespace(tool_calls=tcs, content="")

        result = {"messages": [
            _msg([{"name": "propose_feature", "args": {
                "name": "amt_to_bal", "op": "ratio", "inputs": ["amount", "balance"], "rationale": "risk"}}]),
            _msg([{"name": "propose_feature", "args": {  # invalid → dropped
                "name": "bad", "op": "divide", "inputs": ["a", "b"]}}]),
            _msg([{"name": "record_findings", "args": {"summary": "ok"}}]),
        ]}
        f = agents._extract_finding("fraud", result)
        names = [s.name for s in f.feature_specs]
        self.assertEqual(names, ["amt_to_bal"])


if __name__ == "__main__":
    unittest.main()
