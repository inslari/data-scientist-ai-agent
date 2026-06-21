"""Offline unit tests for the marketing pack + the generic data-driven base.

No LLM/network/DB: routing is verified through the deterministic keyword fallback,
hooks through their static text, and the deep-analysis modules on synthetic data.

Run: uv run python -m unittest tests.test_marketing_pack
"""

import unittest

import numpy as np
import pandas as pd

from src.domains import get_active_pack, NullDomainPack, list_packs
from src.domains.base import MerchantContext
from src.domains.marketing.pack import MarketingPack
from src.domains.generic import router as grouter
from src.services.analysis_marketing import (
    compute_channel_performance,
    compute_campaign_roi,
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
    def test_resolves_marketing(self):
        pack = get_active_pack("marketing")
        self.assertIsInstance(pack, MarketingPack)
        self.assertEqual(pack.name, "marketing")
        self.assertEqual(len(pack.experts), 5)
        self.assertEqual(pack.analysis_type, "marketing")

    def test_listed_with_label(self):
        names = {p["name"]: p["label"] for p in list_packs()}
        self.assertIn("marketing", names)
        self.assertTrue(names["marketing"])
        self.assertIn("ecommerce", names)

    def test_instance_cached(self):
        self.assertIs(get_active_pack("marketing"), get_active_pack("marketing"))


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.pack = MarketingPack()

    def test_campaign(self):
        self.assertEqual(_route(self.pack, "kampanya roi analizi").primary_expert, "campaign")

    def test_attribution(self):
        self.assertEqual(_route(self.pack, "çok temaslı atıf modeli").primary_expert, "attribution")

    def test_ltv(self):
        self.assertEqual(_route(self.pack, "ltv cac geri ödeme analizi").primary_expert, "ltv")

    def test_segmentation(self):
        self.assertEqual(_route(self.pack, "müşteri segmentasyonu yap").primary_expert, "segmentation")

    def test_channel(self):
        self.assertEqual(_route(self.pack, "kanal karması optimizasyonu").primary_expert, "channel")

    def test_no_keyword_defaults_to_segmentation(self):
        ctx = _route(self.pack, "xyzzy plugh foobar")
        self.assertFalse(ctx.is_council)
        self.assertEqual(ctx.primary_expert, "segmentation")


class HooksTest(unittest.TestCase):
    def setUp(self):
        self.pack = MarketingPack()

    def _ctx(self, primary, council=None, is_council=False):
        return MerchantContext(pack_name="marketing", primary_expert=primary,
                               council=council or [primary], is_council=is_council)

    def test_static_hooks_nonempty_and_labeled(self):
        ctx = self._ctx("campaign")
        intent = self.pack.enrich_intent_prompt(ctx)
        self.assertIn("MARKETING DOMAIN CONTEXT", intent)
        self.assertIn("Campaign ROI Expert", intent)
        self.assertIn("MARKETING DOMAIN QUERY GUIDANCE", self.pack.query_hints(ctx))
        self.assertTrue(self.pack.feature_hints(ctx).strip())

    def test_council_lens_has_synthesis(self):
        ctx = self._ctx("campaign", ["campaign", "channel"], is_council=True)
        lens = self.pack.interpret_lens(ctx, "regression")
        self.assertTrue(lens.is_council)
        self.assertIn("SYNTHESIS", lens.system_suffix)
        self.assertIn("Channel Mix Expert", lens.system_suffix)

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
        rng = np.random.default_rng(3)
        n = 2400
        channel = rng.choice(["Paid Search", "Social", "Email", "Organic"], n)
        spend = np.round(rng.uniform(10, 200, n), 2)
        revenue = np.round(spend * rng.uniform(0.5, 3.0, n), 2)
        cls.df = pd.DataFrame({
            "channel": channel,
            "spend": spend,
            "revenue": revenue,
            "conversions": rng.integers(0, 20, n),
        })
        cls.profile = {"summary": {
            "numeric_columns": ["spend", "revenue", "conversions"],
            "categorical_columns": ["channel"],
            "datetime_columns": [],
        }}

    def test_channel_performance(self):
        out = compute_channel_performance(self.df, self.profile)
        self.assertIsNotNone(out)
        c = out["channel_performance"]
        self.assertEqual(c["channel_col"], "channel")
        self.assertEqual(c["metric_col"], "revenue")
        self.assertGreaterEqual(c["n_channels"], 2)

    def test_campaign_roi(self):
        out = compute_campaign_roi(self.df, self.profile)
        self.assertIsNotNone(out)
        r = out["campaign_roi"]
        self.assertEqual(r["cost_col"], "spend")
        self.assertEqual(r["revenue_col"], "revenue")
        self.assertIsNotNone(r["overall_roi"])
        self.assertIsNotNone(r["by_channel"])

    def test_returns_none_when_irrelevant(self):
        df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) * 2.0})
        prof = {"summary": {"numeric_columns": ["x", "y"], "categorical_columns": [], "datetime_columns": []}}
        self.assertIsNone(compute_channel_performance(df, prof))
        self.assertIsNone(compute_campaign_roi(df, prof))