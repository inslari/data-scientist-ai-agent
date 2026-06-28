"""Offline tests for the deterministic feature-recipe layer (no LLM/DB)."""

import unittest

import pandas as pd

from src.domains import NullDomainPack
from src.domains.base import MerchantContext
from src.domains.financial.pack import FinancialPack
import numpy as np

from src.domains.financial.feature_recipes import (
    VolatilityRecipe,
    SpikeRatioRecipe,
    LogAmountRecipe,
    TimePartsRecipe,
    InstallmentBurdenRecipe,
    AffordabilityRatioRecipe,
    ZeroBalanceFlagRecipe,
    HoldingCountBandRecipe,
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


class LogAmountRecipeTest(unittest.TestCase):
    def _skewed_df(self):
        rng = np.random.RandomState(0)
        return pd.DataFrame({
            "transaction_amount": rng.lognormal(3, 1.5, 300),   # money, skewed
            "account_balance": rng.lognormal(5, 1.0, 300),      # money, skewed
            "cv_amount": rng.rand(300),                          # derived → skip
            "customer_id": np.arange(300),                       # not money → skip
        })

    def test_builds_log_for_skewed_money_cols(self):
        df = self._skewed_df()
        r = LogAmountRecipe()
        self.assertTrue(r.applies(df))
        cols = r.build(df)
        self.assertIn("log_transaction_amount", cols)
        self.assertIn("log_account_balance", cols)
        # log compresses the right tail: post-transform skew is far smaller.
        self.assertLess(abs(cols["log_transaction_amount"].skew()), 1.0)

    def test_skips_derived_and_nonmoney_cols(self):
        cols = LogAmountRecipe().build(self._skewed_df())
        self.assertNotIn("log_cv_amount", cols)       # derived prefix excluded
        self.assertNotIn("log_customer_id", cols)     # name not monetary

    def test_skips_negative_columns(self):
        df = pd.DataFrame({"amount": [-1.0, 2.0, 3.0, 4.0]})  # has negatives
        self.assertFalse(LogAmountRecipe().applies(df))


class TimePartsRecipeTest(unittest.TestCase):
    def _ts_df(self):
        return pd.DataFrame({
            "event_timestamp": pd.date_range("2024-01-01", periods=200, freq="37min"),
            "amount": np.arange(200, dtype=float),
        })

    def test_builds_hour_cyclical_and_dow(self):
        df = self._ts_df()
        r = TimePartsRecipe()
        self.assertTrue(r.applies(df))
        cols = r.build(df)
        for c in ("hour_event_timestamp", "hour_sin_event_timestamp",
                  "hour_cos_event_timestamp", "dow_event_timestamp"):
            self.assertIn(c, cols)
        self.assertGreaterEqual(cols["hour_event_timestamp"].min(), 0)
        self.assertLessEqual(cols["hour_event_timestamp"].max(), 23)

    def test_skips_when_no_timestamp(self):
        df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
        self.assertFalse(TimePartsRecipe().applies(df))


class InstallmentBurdenRecipeTest(unittest.TestCase):
    def _credit_df(self):
        # Mirrors the German-credit schema: a monetary principal + a loan term.
        return pd.DataFrame({
            "Credit amount": [1200.0, 6000.0, 3000.0],
            "Duration": [6, 48, 12],          # months
            "Age": [30, 22, 45],              # term-like? no → not matched
        })

    def test_builds_burden_ratio(self):
        df = self._credit_df()
        r = InstallmentBurdenRecipe()
        self.assertTrue(r.applies(df))
        cols = r.build(df)
        self.assertIn("burden_Credit amount_per_Duration", cols)
        # amount / term: 1200/6 = 200, 6000/48 = 125, 3000/12 = 250
        burden = cols["burden_Credit amount_per_Duration"]
        self.assertAlmostEqual(burden.iloc[0], 200.0, places=2)
        self.assertAlmostEqual(burden.iloc[1], 125.0, places=2)
        self.assertAlmostEqual(burden.iloc[2], 250.0, places=2)

    def test_skips_without_term_column(self):
        df = pd.DataFrame({"Credit amount": [100.0, 200.0]})  # amount but no term
        self.assertFalse(InstallmentBurdenRecipe().applies(df))

    def test_skips_without_amount_column(self):
        df = pd.DataFrame({"Duration": [6, 12, 24]})  # term but no money col
        self.assertFalse(InstallmentBurdenRecipe().applies(df))

    def test_zero_term_is_safe(self):
        df = pd.DataFrame({"Credit amount": [100.0], "Duration": [0]})
        # max term is 0 → not a valid term column → recipe stays inert (no div/0)
        self.assertFalse(InstallmentBurdenRecipe().applies(df))


class AffordabilityRatioRecipeTest(unittest.TestCase):
    def _bnpl_df(self):
        # Mirrors the BNPL schema: an obligation (purchase) + recurring income.
        return pd.DataFrame({
            "purchase_amount": [5000.0, 1000.0, 2000.0],
            "monthly_income": [50000.0, 10000.0, 8000.0],
            "credit_score": [700, 300, 450],   # money? no → ignored
        })

    def test_builds_affordability_ratio(self):
        df = self._bnpl_df()
        r = AffordabilityRatioRecipe()
        self.assertTrue(r.applies(df))
        cols = r.build(df)
        self.assertIn("purchase_amount_to_income", cols)
        # obligation / income: 5000/50000 = 0.1, 1000/10000 = 0.1, 2000/8000 = 0.25
        ratio = cols["purchase_amount_to_income"]
        self.assertAlmostEqual(ratio.iloc[0], 0.1, places=4)
        self.assertAlmostEqual(ratio.iloc[2], 0.25, places=4)

    def test_skips_without_income(self):
        df = pd.DataFrame({"purchase_amount": [100.0, 200.0]})  # obligation, no income
        self.assertFalse(AffordabilityRatioRecipe().applies(df))

    def test_skips_without_obligation(self):
        df = pd.DataFrame({"monthly_income": [5000.0, 8000.0]})  # income, no obligation
        self.assertFalse(AffordabilityRatioRecipe().applies(df))

    def test_income_not_treated_as_obligation(self):
        # An income-only money column must never pair with itself.
        from src.domains.financial.feature_recipes import _obligation_cols, _income_cols
        df = pd.DataFrame({"salary": [5000.0, 8000.0]})
        self.assertEqual(_obligation_cols(df), [])
        self.assertEqual(_income_cols(df), ["salary"])


class ZeroBalanceFlagRecipeTest(unittest.TestCase):
    def _churn_df(self):
        # Mirrors the banking-churn schema: a balance with a chunk of zeros.
        return pd.DataFrame({
            "Balance": [0.0, 0.0, 0.0, 120000.0, 85000.0, 99000.0],
            "EstimatedSalary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 75000.0],
            "credit_score": [600, 700, 650, 720, 580, 690],  # score, not a magnitude
        })

    def test_builds_zero_flag_for_balance(self):
        df = self._churn_df()
        r = ZeroBalanceFlagRecipe()
        self.assertTrue(r.applies(df))
        cols = r.build(df)
        self.assertIn("is_zero_Balance", cols)
        # rows with Balance == 0 → 1.0, else 0.0
        self.assertEqual(list(cols["is_zero_Balance"]), [1, 1, 1, 0, 0, 0])

    def test_skips_score_and_no_zero_columns(self):
        cols = ZeroBalanceFlagRecipe().build(self._churn_df())
        self.assertNotIn("is_zero_credit_score", cols)       # score, not a magnitude
        self.assertNotIn("is_zero_EstimatedSalary", cols)    # no zero rows → no signal

    def test_skips_all_zero_and_no_zero(self):
        all_zero = pd.DataFrame({"balance": [0.0, 0.0, 0.0, 0.0]})
        no_zero = pd.DataFrame({"balance": [1.0, 2.0, 3.0, 4.0]})
        self.assertFalse(ZeroBalanceFlagRecipe().applies(all_zero))
        self.assertFalse(ZeroBalanceFlagRecipe().applies(no_zero))

    def test_skips_negative_columns(self):
        df = pd.DataFrame({"balance": [0.0, -5.0, 10.0, 0.0]})  # overdraft → negatives
        self.assertFalse(ZeroBalanceFlagRecipe().applies(df))


class HoldingCountBandRecipeTest(unittest.TestCase):
    def _churn_df(self):
        # Mirrors NumOfProducts (1 / 2 / 3 / 4) from the banking-churn notebook.
        return pd.DataFrame({
            "NumOfProducts": [1, 2, 3, 1, 4, 2, 1, 2],
            "Tenure": [3, 5, 1, 8, 2, 6, 10, 4],   # not a holding count by name
        })

    def test_builds_single_and_multi_bands(self):
        df = self._churn_df()
        r = HoldingCountBandRecipe()
        self.assertTrue(r.applies(df))
        cols = r.build(df)
        self.assertIn("numofproducts_is_single", cols)
        self.assertIn("numofproducts_is_multi", cols)
        self.assertEqual(list(cols["numofproducts_is_single"]), [1, 0, 0, 1, 0, 0, 1, 0])
        self.assertEqual(list(cols["numofproducts_is_multi"]),  [0, 1, 1, 0, 1, 1, 0, 1])

    def test_skips_when_all_single(self):
        df = pd.DataFrame({"products": [1, 1, 1, 1]})  # no multi-holders → no signal
        self.assertFalse(HoldingCountBandRecipe().applies(df))

    def test_skips_non_integer_and_oversized(self):
        frac = pd.DataFrame({"products": [1.0, 2.5, 3.0]})         # non-integer
        huge = pd.DataFrame({"product_count": [1, 999, 2]})        # > _HOLDING_MAX
        self.assertFalse(HoldingCountBandRecipe().applies(frac))
        self.assertFalse(HoldingCountBandRecipe().applies(huge))

    def test_skips_unrelated_count_column(self):
        df = pd.DataFrame({"Tenure": [1, 2, 3, 4]})  # name not a holding count
        self.assertFalse(HoldingCountBandRecipe().applies(df))


class RecipeGatingTest(unittest.TestCase):
    def test_fires_for_fraud(self):
        self.assertTrue(recipes_for(["fraud"]))

    def test_fires_for_revenue(self):
        self.assertTrue(recipes_for(["revenue"]))

    def test_fires_for_customer_with_churn_recipes(self):
        names = {r.name for r in recipes_for(["customer"])}
        self.assertIn("financial.zero_balance_flag", names)
        self.assertIn("financial.holding_count_band", names)
        self.assertIn("financial.log_amount", names)

    def test_council_union_dedups_log_amount(self):
        recipes = recipes_for(["fraud", "customer"])
        names = [r.name for r in recipes]
        self.assertEqual(names.count("financial.log_amount"), 1)  # shared, once only
        self.assertIn("financial.zero_balance_flag", names)       # from customer
        self.assertIn("financial.volatility_cv", names)           # from fraud

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

    def test_financial_pack_returns_churn_recipes_for_customer(self):
        pack = FinancialPack()
        ctx = MerchantContext(pack_name="financial", primary_expert="customer", council=["customer"])
        names = {r.name for r in pack.feature_recipes(ctx)}
        self.assertIn("financial.zero_balance_flag", names)
        self.assertIn("financial.holding_count_band", names)

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
