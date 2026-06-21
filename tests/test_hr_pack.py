"""Offline unit tests for the HR pack + the generic data-driven base.

No LLM/network/DB: routing is verified through the deterministic keyword fallback,
hooks through their static text, and the deep-analysis modules on synthetic data.

Run: uv run python -m unittest tests.test_hr_pack
"""

import unittest

import numpy as np
import pandas as pd

from src.domains import get_active_pack, NullDomainPack, list_packs
from src.domains.base import MerchantContext
from src.domains.hr.pack import HRPack
from src.domains.generic import router as grouter
from src.services.analysis_hr import (
    compute_attrition_overview,
    compute_compensation_gaps,
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
    def test_resolves_hr(self):
        pack = get_active_pack("hr")
        self.assertIsInstance(pack, HRPack)
        self.assertEqual(pack.name, "hr")
        self.assertEqual(len(pack.experts), 5)
        self.assertEqual(pack.analysis_type, "hr")

    def test_listed_with_label(self):
        names = {p["name"]: p["label"] for p in list_packs()}
        self.assertIn("hr", names)
        self.assertTrue(names["hr"])
        self.assertIn("ecommerce", names)

    def test_instance_cached(self):
        self.assertIs(get_active_pack("hr"), get_active_pack("hr"))


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.pack = HRPack()

    def test_attrition(self):
        self.assertEqual(_route(self.pack, "çalışan işten ayrılma tahmini").primary_expert, "attrition")

    def test_performance(self):
        self.assertEqual(_route(self.pack, "çalışan performans değerlendirmesi").primary_expert, "performance")

    def test_recruitment(self):
        self.assertEqual(_route(self.pack, "işe alım hunisi analizi").primary_expert, "recruitment")

    def test_compensation(self):
        self.assertEqual(_route(self.pack, "maaş ve ücret farkı analizi").primary_expert, "compensation")

    def test_engagement(self):
        self.assertEqual(_route(self.pack, "çalışan memnuniyeti anketi").primary_expert, "engagement")

    def test_no_keyword_defaults_to_attrition(self):
        ctx = _route(self.pack, "xyzzy plugh foobar")
        self.assertFalse(ctx.is_council)
        self.assertEqual(ctx.primary_expert, "attrition")


class HooksTest(unittest.TestCase):
    def setUp(self):
        self.pack = HRPack()

    def _ctx(self, primary, council=None, is_council=False):
        return MerchantContext(pack_name="hr", primary_expert=primary,
                               council=council or [primary], is_council=is_council)

    def test_static_hooks_nonempty_and_labeled(self):
        ctx = self._ctx("attrition")
        intent = self.pack.enrich_intent_prompt(ctx)
        self.assertIn("HR DOMAIN CONTEXT", intent)
        self.assertIn("Attrition & Turnover Expert", intent)
        self.assertIn("HR DOMAIN QUERY GUIDANCE", self.pack.query_hints(ctx))
        self.assertTrue(self.pack.feature_hints(ctx).strip())

    def test_council_lens_has_synthesis(self):
        ctx = self._ctx("attrition", ["attrition", "compensation"], is_council=True)
        lens = self.pack.interpret_lens(ctx, "classification")
        self.assertTrue(lens.is_council)
        self.assertIn("SYNTHESIS", lens.system_suffix)
        self.assertIn("Compensation & Pay-Equity Expert", lens.system_suffix)

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
        rng = np.random.default_rng(2)
        n = 2000
        dept = rng.choice(["Sales", "Engineering", "Support", "Finance"], n)
        base = {"Sales": 6000, "Engineering": 9000, "Support": 4500, "Finance": 7000}
        salary = np.array([base[d] for d in dept]) + rng.normal(0, 500, n)
        cls.df = pd.DataFrame({
            "employee_id": np.arange(n),
            "attrition": rng.integers(0, 2, n),
            "monthly_salary": np.round(salary, 2),
            "department": dept,
            "age": rng.integers(22, 60, n),
        })
        cls.profile = {"summary": {
            "numeric_columns": ["attrition", "monthly_salary", "age"],
            "categorical_columns": ["department"],
            "datetime_columns": [],
        }}

    def test_attrition(self):
        out = compute_attrition_overview(self.df, self.profile)
        self.assertIsNotNone(out)
        a = out["attrition_overview"]
        self.assertEqual(a["flag_col"], "attrition")
        self.assertTrue(0.0 <= a["overall_rate"] <= 1.0)
        self.assertIsNotNone(a["by_group"])
        self.assertEqual(a["by_group"]["column"], "department")

    def test_compensation(self):
        out = compute_compensation_gaps(self.df, self.profile)
        self.assertIsNotNone(out)
        c = out["compensation_gaps"]
        self.assertEqual(c["salary_col"], "monthly_salary")
        self.assertEqual(c["group_col"], "department")
        self.assertEqual(c["highest_group"], "Engineering")  # highest base pay
        self.assertEqual(c["lowest_group"], "Support")

    def test_returns_none_when_irrelevant(self):
        df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) * 2.0})
        prof = {"summary": {"numeric_columns": ["x", "y"], "categorical_columns": [], "datetime_columns": []}}
        self.assertIsNone(compute_attrition_overview(df, prof))
        self.assertIsNone(compute_compensation_gaps(df, prof))


if __name__ == "__main__":
    unittest.main()
