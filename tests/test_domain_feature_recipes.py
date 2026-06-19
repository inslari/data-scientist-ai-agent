"""Offline tests for the deterministic feature-recipe layer (no LLM/DB)."""

import unittest

import pandas as pd

from src.domains import NullDomainPack
from src.domains.base import MerchantContext
from src.domains.financial.pack import FinancialPack
from src.domains.financial.feature_recipes import (
    VolatilityRecipe,
    SpikeRatioRecipe,
    recipes_for,
)


def _agg_df():
    """Mimic the feature builder's post-aggregation naming convention."""
    return pd.DataFrame({
        "avg_amount": [10.0, 100.0, 0.0],
        "std_amount": [5.0, 50.0, 0.0],
        "max_amount": [40.0, 100.0, 0.0],
        "count_card_id": [3, 7, 1],
    })


class VolatilityRecipeTest(unittest.TestCase):
    def test_builds_cv_from_avg_std_pair(self):
        df = _agg_df()
        r = VolatilityRecipe()
        self.assertTrue(r.applies(df))
        cols = r.build(df)
        self.assertIn("cv_amount", cols)
        # cv = std / (|avg| + eps): 5/10 = 0.5, 50/100 = 0.5, 0/eps = 0
        self.assertAlmostEqual(cols["cv_amount"].iloc[0], 0.5, places=4)
        self.assertAlmostEqual(cols["cv_amount"].iloc[1], 0.5, places=4)
        self.assertEqual(cols["cv_amount"].iloc[2], 0.0)  # zero-safe, no NaN/inf

    def test_skips_when_no_pair(self):
        df = pd.DataFrame({"avg_amount": [1.0], "count_x": [2]})  # no std_
        self.assertFalse(VolatilityRecipe().applies(df))


class SpikeRatioRecipeTest(unittest.TestCase):
    def test_builds_peak_ratio_from_avg_max_pair(self):
        df = _agg_df()
        cols = SpikeRatioRecipe().build(df)
        self.assertIn("peak_ratio_amount", cols)
        # max/avg: 40/10 = 4.0, 100/100 = 1.0
        self.assertAlmostEqual(cols["peak_ratio_amount"].iloc[0], 4.0, places=4)
        self.assertAlmostEqual(cols["peak_ratio_amount"].iloc[1], 1.0, places=4)


class RecipeGatingTest(unittest.TestCase):
    def test_fires_for_fraud(self):
        self.assertTrue(recipes_for(["fraud"]))

    def test_fires_for_revenue(self):
        self.assertTrue(recipes_for(["revenue"]))

    def test_silent_for_unrelated_expert(self):
        self.assertEqual(recipes_for(["growth"]), [])
        self.assertEqual(recipes_for([]), [])


class PackHookTest(unittest.TestCase):
    def test_financial_pack_returns_recipes_for_fraud(self):
        pack = FinancialPack()
        ctx = MerchantContext(pack_name="financial", primary_expert="fraud", council=["fraud"])
        recipes = pack.feature_recipes(ctx)
        names = {r.name for r in recipes}
        self.assertIn("financial.volatility_cv", names)
        self.assertIn("financial.spike_ratio", names)

    def test_financial_pack_silent_for_growth(self):
        pack = FinancialPack()
        ctx = MerchantContext(pack_name="financial", primary_expert="growth", council=["growth"])
        self.assertEqual(pack.feature_recipes(ctx), [])

    def test_null_pack_returns_no_recipes(self):
        ctx = MerchantContext(pack_name="none", primary_expert="")
        self.assertEqual(NullDomainPack().feature_recipes(ctx), [])


class IntegrationTest(unittest.TestCase):
    """The builder helper materializes recipe columns onto the dataframe in place."""

    def test_apply_domain_recipes_adds_columns(self):
        from src.services.feature_builder import _apply_domain_recipes

        df = _agg_df()
        state = {
            "domain_pack": "financial",
            "merchant_context": MerchantContext(
                pack_name="financial", primary_expert="fraud", council=["fraud"]
            ).to_dict(),
            "db_intelligence": {},
        }
        added = _apply_domain_recipes(df, state, meta=None)
        self.assertIn("cv_amount", added)
        self.assertIn("peak_ratio_amount", added)
        self.assertIn("cv_amount", df.columns)
        self.assertIn("peak_ratio_amount", df.columns)

    def test_apply_domain_recipes_noop_when_pack_none(self):
        from src.services.feature_builder import _apply_domain_recipes

        df = _agg_df()
        before = set(df.columns)
        added = _apply_domain_recipes(df, {"domain_pack": "none"}, meta=None)
        self.assertEqual(added, [])
        self.assertEqual(set(df.columns), before)


if __name__ == "__main__":
    unittest.main()
