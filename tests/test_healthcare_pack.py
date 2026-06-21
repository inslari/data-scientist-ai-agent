"""Offline unit tests for the healthcare pack + the generic data-driven base.

No LLM/network/DB: routing is verified through the deterministic keyword fallback,
hooks through their static text, and the deep-analysis modules on synthetic data.

Run: uv run python -m unittest tests.test_healthcare_pack
"""

import unittest

import numpy as np
import pandas as pd

from src.domains import get_active_pack, NullDomainPack, list_packs
from src.domains.base import MerchantContext
from src.domains.healthcare.pack import HealthcarePack
from src.domains.generic import router as grouter
from src.services.analysis_healthcare import (
    compute_survival_summary,
    compute_readmission_rate,
)


def _route(pack, query):
    return grouter.keyword_route(
        query,
        pack_name=pack.name,
        experts=pack.experts,
        expert_keywords=pack.expert_keywords,
        default_expert=pack.default_expert,
    )


class RegistryTest(unittest.TestCase):
    def test_resolves_healthcare(self):
        pack = get_active_pack("healthcare")
        self.assertIsInstance(pack, HealthcarePack)
        self.assertEqual(pack.name, "healthcare")
        self.assertEqual(len(pack.experts), 5)
        self.assertEqual(pack.analysis_type, "healthcare")

    def test_listed_with_label(self):
        names = {p["name"]: p["label"] for p in list_packs()}
        self.assertIn("healthcare", names)
        self.assertTrue(names["healthcare"])
        self.assertIn("ecommerce", names)  # other packs still present

    def test_instance_cached(self):
        self.assertIs(get_active_pack("healthcare"), get_active_pack("healthcare"))


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.pack = HealthcarePack()

    def test_survival(self):
        self.assertEqual(_route(self.pack, "kaplan-meier sağkalım analizi").primary_expert, "survival")

    def test_risk(self):
        self.assertEqual(_route(self.pack, "hasta için risk skoru hesapla").primary_expert, "risk")

    def test_coding(self):
        self.assertEqual(_route(self.pack, "icd-10 tanı kodu eşleme").primary_expert, "coding")

    def test_readmission(self):
        self.assertEqual(_route(self.pack, "30 günlük yeniden yatış tahmini").primary_expert, "readmission")

    def test_no_keyword_defaults_to_cohort(self):
        ctx = _route(self.pack, "xyzzy plugh foobar")
        self.assertFalse(ctx.is_council)
        self.assertEqual(ctx.primary_expert, "cohort")


class HooksTest(unittest.TestCase):
    def setUp(self):
        self.pack = HealthcarePack()

    def _ctx(self, primary, council=None, is_council=False):
        return MerchantContext(pack_name="healthcare", primary_expert=primary,
                               council=council or [primary], is_council=is_council)

    def test_static_hooks_nonempty_and_labeled(self):
        ctx = self._ctx("survival")
        intent = self.pack.enrich_intent_prompt(ctx)
        self.assertIn("HEALTHCARE DOMAIN CONTEXT", intent)
        self.assertIn("Survival Analysis Expert", intent)
        self.assertIn("HEALTHCARE DOMAIN QUERY GUIDANCE", self.pack.query_hints(ctx))
        self.assertTrue(self.pack.feature_hints(ctx).strip())

    def test_council_lens_has_synthesis(self):
        ctx = self._ctx("risk", ["risk", "readmission"], is_council=True)
        lens = self.pack.interpret_lens(ctx, "classification")
        self.assertTrue(lens.is_council)
        self.assertIn("SYNTHESIS", lens.system_suffix)
        self.assertIn("Readmission & LOS Expert", lens.system_suffix)

    def test_unknown_expert_degrades_to_empty(self):
        ctx = self._ctx("does_not_exist")
        self.assertEqual(self.pack.query_hints(ctx), "")
        self.assertEqual(self.pack.feature_hints(ctx), "")


class NullParityTest(unittest.TestCase):
    def test_null_pack_unaffected(self):
        np_pack = NullDomainPack()
        ctx = np_pack.select_experts("anything")
        self.assertEqual(ctx.primary_expert, "")
        self.assertEqual(np_pack.query_hints(ctx), "")


class DeepAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 3000
        cls.df = pd.DataFrame({
            "patient_id": np.arange(n),
            "los_days": np.round(rng.exponential(6, n) + 1, 1),
            "death": rng.integers(0, 2, n),
            "readmitted": rng.integers(0, 2, n),
            "department": rng.choice(["Cardiology", "Oncology", "Surgery", "ICU"], n),
        })
        cls.profile = {"summary": {
            "numeric_columns": ["los_days", "death", "readmitted"],
            "categorical_columns": ["department"],
            "datetime_columns": [],
        }}

    def test_survival(self):
        out = compute_survival_summary(self.df, self.profile)
        self.assertIsNotNone(out)
        s = out["survival_summary"]
        self.assertEqual(s["time_col"], "los_days")
        self.assertEqual(s["event_col"], "death")
        self.assertGreater(s["n"], 100)
        self.assertEqual(len(s["survival_curve"]), 3)

    def test_readmission(self):
        out = compute_readmission_rate(self.df, self.profile)
        self.assertIsNotNone(out)
        r = out["readmission_rate"]
        self.assertEqual(r["flag_col"], "readmitted")
        self.assertTrue(0.0 <= r["overall_rate"] <= 1.0)
        self.assertIsNotNone(r["by_group"])
        self.assertEqual(r["by_group"]["column"], "department")

    def test_returns_none_when_irrelevant(self):
        df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) * 2.0})
        prof = {"summary": {"numeric_columns": ["x", "y"], "categorical_columns": [], "datetime_columns": []}}
        self.assertIsNone(compute_survival_summary(df, prof))
        self.assertIsNone(compute_readmission_rate(df, prof))


if __name__ == "__main__":
    unittest.main()
