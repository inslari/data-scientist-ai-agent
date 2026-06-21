"""Offline unit tests for the retail pack + the generic data-driven base.

No LLM/network/DB: routing is verified through the deterministic keyword fallback,
hooks through their static text, and the deep-analysis modules on synthetic data.

Run: uv run python -m unittest tests.test_retail_pack
"""

import unittest

import numpy as np
import pandas as pd

from src.domains import get_active_pack, NullDomainPack, list_packs
from src.domains.base import MerchantContext
from src.domains.retail.pack import RetailPack
from src.domains.generic import router as grouter
from src.services.analysis_retail import (
    compute_demand_trend,
    compute_price_elasticity,
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
    def test_resolves_retail(self):
        pack = get_active_pack("retail")
        self.assertIsInstance(pack, RetailPack)
        self.assertEqual(pack.name, "retail")
        self.assertEqual(len(pack.experts), 5)
        self.assertEqual(pack.analysis_type, "retail")

    def test_listed_with_label(self):
        names = {p["name"]: p["label"] for p in list_packs()}
        self.assertIn("retail", names)
        self.assertTrue(names["retail"])
        self.assertIn("ecommerce", names)

    def test_instance_cached(self):
        self.assertIs(get_active_pack("retail"), get_active_pack("retail"))


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.pack = RetailPack()

    def test_inventory(self):
        self.assertEqual(_route(self.pack, "güvenlik stoğu ve yeniden sipariş noktası").primary_expert, "inventory")

    def test_pricing(self):
        self.assertEqual(_route(self.pack, "fiyat esnekliği analizi").primary_expert, "pricing")

    def test_demand(self):
        self.assertEqual(_route(self.pack, "talep tahmini modeli kur").primary_expert, "demand")

    def test_assortment(self):
        self.assertEqual(_route(self.pack, "ürün çeşitliliği ve raf alanı optimizasyonu").primary_expert, "assortment")

    def test_no_keyword_defaults_to_demand(self):
        ctx = _route(self.pack, "xyzzy plugh foobar")
        self.assertFalse(ctx.is_council)
        self.assertEqual(ctx.primary_expert, "demand")


class HooksTest(unittest.TestCase):
    def setUp(self):
        self.pack = RetailPack()

    def _ctx(self, primary, council=None, is_council=False):
        return MerchantContext(pack_name="retail", primary_expert=primary,
                               council=council or [primary], is_council=is_council)

    def test_static_hooks_nonempty_and_labeled(self):
        ctx = self._ctx("inventory")
        intent = self.pack.enrich_intent_prompt(ctx)
        self.assertIn("RETAIL DOMAIN CONTEXT", intent)
        self.assertIn("Inventory Optimization Expert", intent)
        self.assertIn("RETAIL DOMAIN QUERY GUIDANCE", self.pack.query_hints(ctx))
        self.assertTrue(self.pack.feature_hints(ctx).strip())

    def test_council_lens_has_synthesis(self):
        ctx = self._ctx("demand", ["demand", "pricing"], is_council=True)
        lens = self.pack.interpret_lens(ctx, "regression")
        self.assertTrue(lens.is_council)
        self.assertIn("SYNTHESIS", lens.system_suffix)
        self.assertIn("Pricing & Elasticity Expert", lens.system_suffix)

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
        rng = np.random.default_rng(1)
        n = 4000
        dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
        price = np.round(rng.uniform(5, 50, n), 2)
        # downward-sloping demand in price + noise
        units = np.maximum(1, np.round(200 / price + rng.normal(0, 1, n))).astype(int)
        cls.df = pd.DataFrame({
            "order_date": dates,
            "unit_price": price,
            "units_sold": units,
            "store": rng.choice(["A", "B", "C"], n),
        })
        cls.profile = {"summary": {
            "numeric_columns": ["unit_price", "units_sold"],
            "categorical_columns": ["store"],
            "datetime_columns": ["order_date"],
        }}

    def test_demand_trend(self):
        out = compute_demand_trend(self.df, self.profile)
        self.assertIsNotNone(out)
        d = out["demand_trend"]
        self.assertEqual(d["date_col"], "order_date")
        self.assertEqual(d["quantity_col"], "units_sold")
        self.assertIn(d["trend"], ("up", "down", "flat"))
        self.assertGreater(d["n_days"], 10)

    def test_price_elasticity(self):
        out = compute_price_elasticity(self.df, self.profile)
        self.assertIsNotNone(out)
        e = out["price_elasticity"]
        self.assertEqual(e["price_col"], "unit_price")
        self.assertEqual(e["quantity_col"], "units_sold")
        self.assertLess(e["elasticity"], 0)  # demand falls as price rises

    def test_returns_none_when_irrelevant(self):
        df = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) * 2.0})
        prof = {"summary": {"numeric_columns": ["a", "b"], "categorical_columns": [], "datetime_columns": []}}
        self.assertIsNone(compute_demand_trend(df, prof))
        self.assertIsNone(compute_price_elasticity(df, prof))


if __name__ == "__main__":
    unittest.main()
