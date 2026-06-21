"""Offline unit tests for the e-commerce pack + the generic data-driven base.

No LLM/network/DB: routing is verified through the deterministic keyword fallback,
hooks through their static text, and the deep-analysis modules on synthetic data.

Run: uv run python -m unittest tests.test_ecommerce_pack
"""

import unittest

import numpy as np
import pandas as pd

from src.domains import get_active_pack, NullDomainPack, list_packs
from src.domains.base import MerchantContext
from src.domains.ecommerce.pack import EcommercePack
from src.domains.generic import router as grouter
from src.services.analysis_ecommerce import compute_rfm, compute_basket_affinity


def _route(pack, query):
    return grouter.keyword_route(
        query,
        pack_name=pack.name,
        experts=pack.experts,
        expert_keywords=pack.expert_keywords,
        default_expert=pack.default_expert,
    )


class RegistryTest(unittest.TestCase):
    def test_resolves_ecommerce(self):
        pack = get_active_pack("ecommerce")
        self.assertIsInstance(pack, EcommercePack)
        self.assertEqual(pack.name, "ecommerce")
        self.assertEqual(len(pack.experts), 5)
        self.assertEqual(pack.analysis_type, "ecommerce")

    def test_listed_with_label(self):
        names = {p["name"]: p["label"] for p in list_packs()}
        self.assertIn("ecommerce", names)
        self.assertTrue(names["ecommerce"])
        self.assertIn("financial", names)  # still present after refactor

    def test_instance_cached(self):
        self.assertIs(get_active_pack("ecommerce"), get_active_pack("ecommerce"))


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.pack = EcommercePack()

    def test_basket(self):
        self.assertEqual(_route(self.pack, "sepette birlikte alınan ürünler").primary_expert, "basket")

    def test_rfm_customer(self):
        self.assertEqual(_route(self.pack, "rfm segmentasyonu yap").primary_expert, "customer")

    def test_recommendation(self):
        self.assertEqual(_route(self.pack, "ürün tavsiye sistemi kur").primary_expert, "recommendation")

    def test_conversion(self):
        self.assertEqual(_route(self.pack, "checkout dönüşüm hunisi analizi").primary_expert, "conversion")

    def test_no_keyword_defaults_to_customer(self):
        ctx = _route(self.pack, "xyzzy plugh foobar")
        self.assertFalse(ctx.is_council)
        self.assertEqual(ctx.primary_expert, "customer")


class HooksTest(unittest.TestCase):
    def setUp(self):
        self.pack = EcommercePack()

    def _ctx(self, primary, council=None, is_council=False):
        return MerchantContext(pack_name="ecommerce", primary_expert=primary,
                               council=council or [primary], is_council=is_council)

    def test_static_hooks_nonempty_and_labeled(self):
        ctx = self._ctx("basket")
        intent = self.pack.enrich_intent_prompt(ctx)
        self.assertIn("E-COMMERCE DOMAIN CONTEXT", intent)
        self.assertIn("Basket Analysis Expert", intent)
        self.assertIn("E-COMMERCE DOMAIN QUERY GUIDANCE", self.pack.query_hints(ctx))
        self.assertTrue(self.pack.feature_hints(ctx).strip())

    def test_council_lens_has_synthesis(self):
        ctx = self._ctx("customer", ["customer", "churn"], is_council=True)
        lens = self.pack.interpret_lens(ctx, "classification")
        self.assertTrue(lens.is_council)
        self.assertIn("SYNTHESIS", lens.system_suffix)
        self.assertIn("Retention & Churn Expert", lens.system_suffix)

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
        n = 4000
        cls.df = pd.DataFrame({
            "customer_id": rng.integers(1, 400, n),
            "order_id": rng.integers(1, 1200, n),
            "product_category": rng.choice(
                ["Electronics", "Books", "Home", "Toys", "Garden"], n),
            "line_total": np.round(rng.exponential(50, n) + 5, 2),
            "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        })
        cls.profile = {"summary": {
            "numeric_columns": ["line_total"],
            "categorical_columns": ["product_category"],
            "datetime_columns": ["order_date"],
        }}

    def test_rfm(self):
        out = compute_rfm(self.df, self.profile)
        self.assertIsNotNone(out)
        rfm = out["rfm"]
        self.assertEqual(rfm["customer_col"], "customer_id")
        self.assertGreater(rfm["n_customers"], 50)
        self.assertEqual(sum(s["count"] for s in rfm["segments"]), rfm["n_customers"])

    def test_basket(self):
        out = compute_basket_affinity(self.df, self.profile)
        self.assertIsNotNone(out)
        ba = out["basket_affinity"]
        self.assertEqual(ba["item_col"], "product_category")
        self.assertTrue(ba["pairs"])
        for p in ba["pairs"]:
            self.assertIn("lift", p)
            self.assertGreaterEqual(p["count"], 1)

    def test_returns_none_when_irrelevant(self):
        df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) * 2.0})
        prof = {"summary": {"numeric_columns": ["x", "y"], "categorical_columns": [], "datetime_columns": []}}
        self.assertIsNone(compute_rfm(df, prof))
        self.assertIsNone(compute_basket_affinity(df, prof))


if __name__ == "__main__":
    unittest.main()
