"""Offline unit tests for the Domain (Sector) Pack framework + financial pack.

These tests avoid any LLM/network calls: routing is verified through the
deterministic keyword fallback, and hooks are verified to be pure no-ops for the
null pack and non-empty for the financial pack.
"""

import unittest

from src.domains import get_active_pack, NullDomainPack, list_packs
from src.domains.base import MerchantContext
from src.domains.financial.pack import FinancialPack
from src.domains.financial import router


class RegistryResolutionTest(unittest.TestCase):
    def test_explicit_disable_resolves_to_null_pack(self):
        # Explicit "off"/unknown values always disable the layer, regardless of
        # the global DOMAIN_PACK default.
        for name in ("none", "off", "disabled", "unknown_pack_xyz"):
            pack = get_active_pack(name)
            self.assertIsInstance(pack, NullDomainPack)
            self.assertEqual(pack.name, "none")

    def test_empty_falls_back_to_global_default(self):
        # None / "" mean "use the configured global default" — not necessarily null.
        from src.core.config import settings
        for name in (None, ""):
            self.assertEqual(get_active_pack(name).name, get_active_pack(settings.DOMAIN_PACK).name)

    def test_financial_name_resolves_to_financial_pack(self):
        pack = get_active_pack("financial")
        self.assertIsInstance(pack, FinancialPack)
        self.assertEqual(pack.name, "financial")
        self.assertEqual(len(pack.experts), 5)

    def test_accepts_state_dict(self):
        self.assertIsInstance(get_active_pack({"domain_pack": "financial"}), FinancialPack)
        self.assertIsInstance(get_active_pack({"domain_pack": "none"}), NullDomainPack)

    def test_financial_pack_instance_cached(self):
        self.assertIs(get_active_pack("financial"), get_active_pack("financial"))

    def test_list_packs(self):
        packs = list_packs()
        names = [p["name"] for p in packs]
        self.assertEqual(names[0], "none")          # null pack always first
        self.assertIn("financial", names)
        for p in packs:                              # every entry has a label
            self.assertTrue(p["label"])


class NullPackHooksTest(unittest.TestCase):
    def setUp(self):
        self.pack = NullDomainPack()
        self.ctx = self.pack.select_experts("anything at all")

    def test_select_returns_empty_context(self):
        self.assertEqual(self.ctx.pack_name, "none")
        self.assertEqual(self.ctx.primary_expert, "")

    def test_all_hooks_are_noops(self):
        self.assertEqual(self.pack.enrich_intent_prompt(self.ctx), "")
        self.assertEqual(self.pack.query_hints(self.ctx), "")
        self.assertEqual(self.pack.feature_hints(self.ctx), "")
        self.assertFalse(self.pack.interpret_lens(self.ctx))  # falsy InterpretLens


class KeywordRoutingTest(unittest.TestCase):
    def test_single_expert_fraud(self):
        ctx = router.keyword_route("sahte işlemleri tespit et")
        self.assertEqual(ctx.primary_expert, "fraud")
        self.assertFalse(ctx.is_council)

    def test_single_expert_fraud_english(self):
        ctx = router.keyword_route("detect fraudulent transactions")
        self.assertEqual(ctx.primary_expert, "fraud")

    def test_council_revenue_and_customer(self):
        ctx = router.keyword_route("gelir ve müşteri segmentleri")
        self.assertTrue(ctx.is_council)
        self.assertIn("revenue", ctx.council)
        self.assertIn("customer", ctx.council)

    def test_no_keyword_defaults_single(self):
        ctx = router.keyword_route("xyzzy plugh foobar")
        self.assertFalse(ctx.is_council)
        self.assertEqual(ctx.primary_expert, "revenue")  # documented default

    def test_kart_does_not_false_match_revenue_kar(self):
        # "kart" must not match the revenue keyword "kar" (substring trap).
        scores = router.keyword_scores("kart işlemleri")
        self.assertEqual(scores["revenue"], 0)


class FinancialPackHooksTest(unittest.TestCase):
    def setUp(self):
        self.pack = FinancialPack()

    def _ctx(self, primary, council=None, is_council=False):
        return MerchantContext(
            pack_name="financial",
            primary_expert=primary,
            council=council or [primary],
            is_council=is_council,
        )

    def test_single_expert_hooks_nonempty(self):
        ctx = self._ctx("fraud")
        self.assertIn("Fraud Expert", self.pack.enrich_intent_prompt(ctx))
        self.assertIn("velocity", self.pack.query_hints(ctx).lower())
        self.assertIn("leakage", self.pack.feature_hints(ctx).lower())
        lens = self.pack.interpret_lens(ctx, "anomaly_detection")
        self.assertTrue(lens)
        self.assertIn("fraud", lens.system_suffix.lower())
        self.assertFalse(lens.is_council)

    def test_council_lens_has_synthesis(self):
        ctx = self._ctx("revenue", ["revenue", "customer"], is_council=True)
        lens = self.pack.interpret_lens(ctx, "report")
        self.assertTrue(lens.is_council)
        self.assertIn("SYNTHESIS", lens.system_suffix)
        self.assertIn("Revenue Expert", lens.system_suffix)
        self.assertIn("Customer Expert", lens.system_suffix)

    def test_unknown_expert_key_is_ignored(self):
        ctx = self._ctx("does_not_exist")
        # active_experts empty → hooks degrade to empty / generic, never raise
        self.assertEqual(self.pack.query_hints(ctx), "")
        self.assertEqual(self.pack.feature_hints(ctx), "")

    def test_hooks_accept_domain_doc(self):
        ctx = self._ctx("fraud")
        doc = "This dataset has transactions, cards, and a fraud_labels table."
        # enrich_intent_prompt embeds a snippet of the doc
        out = self.pack.enrich_intent_prompt(ctx, doc)
        self.assertIn("fraud_labels", out)
        # select_experts still returns a valid context with a doc supplied
        mc = self.pack.select_experts("detect fraud", None, doc)
        self.assertTrue(mc.primary_expert)

    def test_trim_doc(self):
        from src.domains.base import trim_doc
        self.assertEqual(trim_doc(None), "")
        self.assertEqual(trim_doc("short"), "short")
        big = "x" * 5000
        trimmed = trim_doc(big, max_chars=100)
        self.assertLess(len(trimmed), 200)
        self.assertIn("kısaltıldı", trimmed)

    def test_null_pack_ignores_domain_doc(self):
        np = NullDomainPack()
        ctx = np.select_experts("q", None, "some doc")
        self.assertEqual(ctx.primary_expert, "")
        self.assertEqual(np.enrich_intent_prompt(ctx, "some doc"), "")

    def test_context_roundtrip(self):
        ctx = self._ctx("revenue", ["revenue", "customer"], is_council=True)
        restored = MerchantContext.from_dict(ctx.to_dict())
        self.assertEqual(restored.primary_expert, "revenue")
        self.assertEqual(restored.council, ["revenue", "customer"])
        self.assertTrue(restored.is_council)


if __name__ == "__main__":
    unittest.main()
