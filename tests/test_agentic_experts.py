"""Offline tests for the agentic Merchant Intelligence layer (no LLM/network/DB)."""

import asyncio
import types
import unittest

from src.domains import NullDomainPack
from src.domains.schemas import ExpertFinding, MerchantAnalysis
from src.domains.financial.pack import FinancialPack
from src.domains.financial import agents
from src.domains.financial.tools import make_sql_tool


def _msg(tool_calls=None, content=""):
    return types.SimpleNamespace(tool_calls=tool_calls or [], content=content)


class SchemaTest(unittest.TestCase):
    def test_merchant_analysis_roundtrip(self):
        a = MerchantAnalysis(
            pack_name="financial", primary_expert="fraud", council=["fraud", "revenue"],
            is_council=True, synthesis="do X",
            findings=[ExpertFinding(expert="fraud", summary="s", key_signals=["12% fraud"])],
        )
        b = MerchantAnalysis.from_dict(a.to_dict())
        self.assertEqual(b.primary_expert, "fraud")
        self.assertTrue(b.is_council)
        self.assertEqual(b.findings[0].key_signals, ["12% fraud"])
        self.assertEqual(b.finding_for("fraud").summary, "s")
        self.assertIsNone(MerchantAnalysis.from_dict(None))


class SqlToolTest(unittest.TestCase):
    def test_rejects_non_select(self):
        tool = make_sql_tool(data_group_id=1, user_id=1)
        out = asyncio.run(tool.ainvoke({"sql": "DROP TABLE t"}))
        self.assertIn("ERROR", out)
        self.assertIn("DROP", out)

    def test_rejects_stacked_statement(self):
        # The prefix-only guard used to let a SELECT-prefixed stacked statement
        # through; the sqlparse guard rejects multi-statement SQL (C2).
        tool = make_sql_tool(data_group_id=1, user_id=1)
        out = asyncio.run(tool.ainvoke({"sql": "SELECT 1; DROP TABLE t"}))
        self.assertIn("ERROR", out)
        self.assertIn("Multiple statements", out)


class ExtractFindingTest(unittest.TestCase):
    def test_extracts_record_findings_and_queries(self):
        result = {"messages": [
            _msg(tool_calls=[{"name": "run_sql", "args": {"sql": "SELECT count(*) FROM t"}}]),
            _msg(tool_calls=[{"name": "record_findings", "args": {
                "summary": "high risk", "key_signals": ["5% fraud rate"],
                "recommended_target": "is_fraud", "recommended_features": ["velocity_1h"],
                "risks": ["label leakage"], "interpretation_notes": "watch FPs",
            }}]),
        ]}
        f = agents._extract_finding("fraud", result)
        self.assertEqual(f.recommended_target, "is_fraud")
        self.assertEqual(f.recommended_features, ["velocity_1h"])
        self.assertIn("SELECT count(*) FROM t", f.queries_used)

    def test_fallback_when_no_record(self):
        result = {"messages": [_msg(content="some free text conclusion")]}
        f = agents._extract_finding("revenue", result)
        self.assertIn("free text", f.summary)


class RunAnalysisOrchestrationTest(unittest.TestCase):
    def test_council_orchestration(self):
        async def fake_expert(key, *a, **k):
            return ExpertFinding(expert=key, summary=f"{key} ok", recommended_features=[f"{key}_feat"])

        async def fake_synth(findings, q):
            return "merged synthesis"

        orig_e, orig_s = agents._run_one_expert, agents._synthesize
        agents._run_one_expert, agents._synthesize = fake_expert, fake_synth
        try:
            analysis = asyncio.run(agents.run_analysis(
                expert_keys=["fraud", "revenue"], is_council=True, primary_expert="fraud",
                user_query="q", db_intelligence={}, domain_doc="", data_group_id=1,
                user_id=1, source_type="csv",
            ))
        finally:
            agents._run_one_expert, agents._synthesize = orig_e, orig_s

        self.assertIsNotNone(analysis)
        self.assertEqual(len(analysis.findings), 2)
        self.assertTrue(analysis.is_council)
        self.assertEqual(analysis.synthesis, "merged synthesis")

    def test_returns_none_when_all_experts_fail(self):
        async def fail_expert(key, *a, **k):
            return None
        orig = agents._run_one_expert
        agents._run_one_expert = fail_expert
        try:
            analysis = asyncio.run(agents.run_analysis(
                expert_keys=["fraud"], is_council=False, primary_expert="fraud",
                user_query="q", db_intelligence={}, domain_doc="", data_group_id=1,
                user_id=1, source_type="mcp",
            ))
        finally:
            agents._run_one_expert = orig
        self.assertIsNone(analysis)


class HooksRenderFromAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.pack = FinancialPack()
        self.analysis = MerchantAnalysis(
            pack_name="financial", primary_expert="fraud", council=["fraud"], is_council=False,
            synthesis="overall: monitor velocity",
            findings=[ExpertFinding(
                expert="fraud", summary="elevated risk", key_signals=["3% fraud"],
                recommended_target="is_fraud", recommended_features=["velocity_1h", "amount_z"],
                risks=["leakage via is_fraud"], interpretation_notes="balance FP vs catch",
            )],
        )

    def test_feature_hints_use_findings(self):
        out = self.pack.feature_hints(None, None, self.analysis)
        self.assertIn("is_fraud", out)
        self.assertIn("velocity_1h", out)
        self.assertIn("leakage", out)

    def test_query_hints_use_signals(self):
        out = self.pack.query_hints(None, None, self.analysis)
        self.assertIn("3% fraud", out)

    def test_interpret_lens_uses_synthesis(self):
        lens = self.pack.interpret_lens(None, "classification", self.analysis)
        self.assertTrue(lens)
        self.assertIn("monitor velocity", lens.system_suffix)


class NullPackAnalyzeTest(unittest.TestCase):
    def test_null_analyze_returns_none(self):
        out = asyncio.run(NullDomainPack().analyze(
            user_query="q", db_intelligence={}, domain_doc="", data_group_id=1,
            user_id=1, source_type="csv",
        ))
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
