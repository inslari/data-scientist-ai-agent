"""Agentic runtime for the financial experts.

Each selected expert becomes a tool-using ReAct agent (via ``langchain.agents.
create_agent``, the same API used in ``src/agents/intent_discovery_agent.py``) that
investigates the dataset with the source-agnostic ``run_sql`` tool, then records a
structured :class:`ExpertFinding`. Council members run concurrently; a synthesizer
LLM merges their findings into a :class:`MerchantAnalysis`.

Any failure (no LLM key, tool errors, timeout) yields ``None`` so callers fall back
to the static prompt hints — the pipeline never breaks.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.domains.base import trim_doc
from src.domains.schemas import ExpertFinding, MerchantAnalysis
from .experts import ALL_EXPERTS
from .tools import make_sql_tool

# Guardrails
_PER_EXPERT_TIMEOUT_S = 90
_RECURSION_LIMIT = 12          # ~4-5 tool rounds per expert
_MAX_DOC_CHARS = 1500


def _expert_by_key(key: str):
    for e in ALL_EXPERTS:
        if e.key == key:
            return e
    return None


def _db_summary(db_intelligence: dict, source_type: str) -> str:
    """Compact schema text for the agent prompt (works for MCP and CSV)."""
    di = db_intelligence or {}
    lines: list[str] = []
    for t in di.get("tables", []):
        lines.append(f"- table {t.get('table_name')} ({t.get('inferred_entity','?')})")
    cols = di.get("columns", [])
    if cols:
        lines.append("Columns (table.column | semantic_type | note):")
        for c in cols[:80]:
            note = (c.get("description") or "")[:60]
            lines.append(
                f"  {c.get('table_name')}.{c.get('column_name')} | "
                f"{c.get('semantic_type','?')}" + (f" | {note}" if note else "")
            )
    if source_type == "csv":
        lines.append(
            "\nDATA SOURCE: CSV files queried via DuckDB. Write DuckDB-compatible SQL. "
            "Table names are exactly the table names listed above."
        )
    else:
        lines.append("\nDATA SOURCE: SQL database (use standard SQL).")
    return "\n".join(lines) if lines else "No schema available."


class _RecordFindings(BaseModel):
    summary: str = Field(description="One-paragraph conclusion from your investigation")
    key_signals: List[str] = Field(default_factory=list, description="Observations WITH numbers from your queries")
    recommended_target: Optional[str] = Field(default=None, description="Column to predict, if a modeling target makes sense")
    recommended_features: List[str] = Field(default_factory=list, description="Concrete features to engineer for this dataset")
    risks: List[str] = Field(default_factory=list, description="Leakage / data-quality / interpretation risks")
    interpretation_notes: str = Field(default="", description="How downstream reporting should read the results")


def _build_expert_agent(expert, run_sql_tool):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool(args_schema=_RecordFindings)
    def record_findings(summary: str, key_signals: List[str] = [], recommended_target: Optional[str] = None,
                        recommended_features: List[str] = [], risks: List[str] = [], interpretation_notes: str = "") -> dict:
        """Record your final structured findings. Call this EXACTLY ONCE as your last step."""
        return {"status": "recorded"}

    system_prompt = (
        f"You are the {expert.name} of a Merchant Intelligence Engine for financial/payments data.\n"
        f"Persona: {expert.summary}\n"
        f"You care about these signals: {', '.join(expert.signals)}.\n\n"
        "WORKFLOW:\n"
        "1. Use the run_sql tool to investigate the ACTUAL data (counts, distributions, group-bys, "
        "null rates, class balance, ranges). Run at most 4-5 focused aggregate queries.\n"
        "2. Ground every claim in numbers you actually retrieved — do not guess.\n"
        "3. Then call record_findings EXACTLY ONCE with your structured conclusions "
        "(summary, key_signals with numbers, recommended_target, recommended_features, risks, "
        "interpretation_notes). Do not call run_sql after record_findings.\n"
        "Only reference columns/tables that exist in the schema below."
    )

    agent = create_agent(model=get_llm(temperature=0), tools=[run_sql_tool, record_findings], system_prompt=system_prompt)
    return agent


def _extract_finding(expert_key: str, result: dict) -> ExpertFinding:
    """Pull the record_findings args + run_sql audit trail out of the agent transcript."""
    findings_args: Optional[dict] = None
    queries: list[str] = []
    for msg in result.get("messages", []):
        for tc in (getattr(msg, "tool_calls", None) or []):
            name = tc.get("name")
            args = tc.get("args", {}) or {}
            if name == "record_findings" and findings_args is None:
                findings_args = args
            elif name == "run_sql" and args.get("sql"):
                queries.append(args["sql"])

    if findings_args is None:
        # Agent never recorded — synthesize a minimal finding from the last message text.
        last = result.get("messages", [])[-1] if result.get("messages") else None
        text = getattr(last, "content", "") if last else ""
        if isinstance(text, list):
            text = " ".join(str(p) for p in text)
        return ExpertFinding(expert=expert_key, summary=str(text)[:1000], queries_used=queries)

    return ExpertFinding(
        expert=expert_key,
        summary=findings_args.get("summary", ""),
        key_signals=findings_args.get("key_signals") or [],
        recommended_target=findings_args.get("recommended_target"),
        recommended_features=findings_args.get("recommended_features") or [],
        risks=findings_args.get("risks") or [],
        queries_used=queries,
        interpretation_notes=findings_args.get("interpretation_notes", ""),
    )


async def _run_one_expert(expert_key: str, user_query: str, db_context: str, domain_doc: str,
                          run_sql_tool) -> Optional[ExpertFinding]:
    expert = _expert_by_key(expert_key)
    if expert is None:
        return None
    try:
        agent = _build_expert_agent(expert, run_sql_tool)
        doc = trim_doc(domain_doc, max_chars=_MAX_DOC_CHARS)
        user_msg = (
            f"User goal: {user_query}\n\n"
            f"--- DATASET SCHEMA ---\n{db_context}\n"
            + (f"\n--- DATASET DESCRIPTION ---\n{doc}\n" if doc else "")
            + "\nInvestigate, then call record_findings."
        )
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                config={"recursion_limit": _RECURSION_LIMIT},
            ),
            timeout=_PER_EXPERT_TIMEOUT_S,
        )
        return _extract_finding(expert_key, result)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[merchant-agents] expert '{expert_key}' failed: {e}")
        return None


async def _synthesize(findings: List[ExpertFinding], user_query: str) -> str:
    if not findings:
        return ""
    if len(findings) == 1:
        return findings[0].summary
    try:
        bullets = "\n\n".join(
            f"### {f.expert}\nSummary: {f.summary}\nSignals: {'; '.join(f.key_signals)}\n"
            f"Target: {f.recommended_target}\nFeatures: {', '.join(f.recommended_features)}\n"
            f"Risks: {'; '.join(f.risks)}"
            for f in findings
        )
        llm = get_llm(temperature=0)
        resp = await llm.ainvoke([
            ("system", "You are the lead analyst of a Merchant Intelligence council. Merge the experts' "
                       "findings into one coherent, non-repetitive recommendation: the most important signals, "
                       "a single recommended modeling approach/target, the top features, and the key risks."),
            ("human", f"User goal: {user_query}\n\nExpert findings:\n{bullets}"),
        ])
        content = resp.content
        return content if isinstance(content, str) else " ".join(str(p) for p in content)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[merchant-agents] synthesis failed: {e}")
        return "\n\n".join(f"[{f.expert}] {f.summary}" for f in findings)


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
    """Run the selected expert agents and synthesize a MerchantAnalysis. None on failure."""
    if not expert_keys:
        return None
    run_sql_tool = make_sql_tool(data_group_id, user_id)
    db_context = _db_summary(db_intelligence, source_type)

    results = await asyncio.gather(
        *[_run_one_expert(k, user_query, db_context, domain_doc, run_sql_tool) for k in expert_keys]
    )
    findings = [f for f in results if f is not None]
    if not findings:
        return None

    synthesis = await _synthesize(findings, user_query)
    return MerchantAnalysis(
        pack_name="financial",
        primary_expert=primary_expert,
        council=expert_keys,
        is_council=is_council,
        findings=findings,
        synthesis=synthesis,
    )
