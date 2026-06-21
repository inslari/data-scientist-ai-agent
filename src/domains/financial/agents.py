"""Agentic runtime for the financial experts — a thin binding over the generic engine.

The reusable ReAct-agent / council machinery lives in
:mod:`src.domains.generic.agents`; this module binds it to the financial experts and
framing labels. The module-level ``_run_one_expert`` / ``_synthesize`` /
``_extract_finding`` / ``run_analysis`` names are kept because the offline tests
patch and call them here. ``run_analysis`` resolves ``_run_one_expert`` /
``_synthesize`` from this module's namespace at call time, so test monkeypatching
of those seams works.
"""

from __future__ import annotations

from typing import List, Optional

from src.domains.generic import agents as _g
from src.domains.schemas import ExpertFinding, MerchantAnalysis
from .experts import ALL_EXPERTS

_PACK_NAME = "financial"
_ENGINE_LABEL = "Merchant Intelligence Engine"
_DOMAIN_DESCRIPTOR = "financial/payments data"

_EXPERTS_BY_KEY = {e.key: e for e in ALL_EXPERTS}

# Re-exported seams (kept stable for callers/tests that reference them here).
_parse_feature_spec = _g.parse_feature_spec
_VALID_FEATURE_OPS = _g._VALID_FEATURE_OPS
_FEATURE_OP_ARITY = _g._FEATURE_OP_ARITY


def _extract_finding(expert_key: str, result: dict) -> ExpertFinding:
    return _g.extract_finding(expert_key, result)


async def _run_one_expert(expert_key: str, user_query: str, db_context: str, domain_doc: str,
                          run_sql_tool) -> Optional[ExpertFinding]:
    expert = _EXPERTS_BY_KEY.get(expert_key)
    if expert is None:
        return None
    return await _g.run_one_expert(
        expert, user_query, db_context, domain_doc, run_sql_tool,
        engine_label=_ENGINE_LABEL, domain_descriptor=_DOMAIN_DESCRIPTOR,
    )


async def _synthesize(findings: List[ExpertFinding], user_query: str) -> str:
    return await _g.synthesize(findings, user_query, engine_label=_ENGINE_LABEL)


async def run_analysis(
    *,
    expert_keys: List[str],
    is_council: bool,
    primary_expert: str,
    user_query: str,
    db_intelligence: dict,
    domain_doc: str,
    data_group_id: int,
    user_id: int,
    source_type: str,
) -> Optional[MerchantAnalysis]:
    """Run the selected expert agents and synthesize a MerchantAnalysis. None on failure.

    Delegates orchestration to the generic engine but injects this module's
    ``_run_one_expert`` / ``_synthesize`` (looked up at call time) as the seams, so
    tests that monkeypatch them keep working.
    """
    return await _g.run_council(
        pack_name=_PACK_NAME,
        expert_keys=expert_keys,
        is_council=is_council,
        primary_expert=primary_expert,
        user_query=user_query,
        db_intelligence=db_intelligence,
        domain_doc=domain_doc,
        data_group_id=data_group_id,
        user_id=user_id,
        source_type=source_type,
        run_one_expert_fn=lambda *a, **k: _run_one_expert(*a, **k),
        synthesize_fn=lambda *a, **k: _synthesize(*a, **k),
    )
