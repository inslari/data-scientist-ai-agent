"""Data-driven agentic runtime shared by every sector pack.

Each selected expert becomes a tool-using ReAct agent (via ``langchain.agents.
create_agent``) that investigates the dataset with the source-agnostic ``run_sql``
tool, then records a structured :class:`ExpertFinding`. Council members run
concurrently; a synthesizer LLM merges their findings into a
:class:`MerchantAnalysis`.

Everything sector-specific (the expert personas, the engine label, the domain
descriptor, the pack name) is passed in as parameters, so a concrete pack needs
no agents module of its own — :class:`src.domains.generic.pack.GenericDomainPack`
wires this up from the pack's config. A pack may still expose a thin module that
injects ``run_one_expert_fn`` / ``synthesize_fn`` seams (financial does, for its
unit tests).

Any failure (no LLM key, tool errors, timeout) yields ``None`` so callers fall
back to the static prompt hints — the pipeline never breaks.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.domains.base import Expert, trim_doc
from src.domains.schemas import (
    ExpertFinding, MerchantAnalysis, FEATURE_OP_SPEC, validate_feature_spec_shape,
)

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Guardrails
_PER_EXPERT_TIMEOUT_S = 90
_RECURSION_LIMIT = 12          # ~4-5 tool rounds per expert
_MAX_DOC_CHARS = 1500


def db_summary(db_intelligence: dict, source_type: str) -> str:
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


class _RecordQuery(BaseModel):
    title: str = Field(description="Short descriptive name, e.g. 'Revenue by Category'")
    description: str = Field(default="", description="What this query reveals and when to use it")
    sql: str = Field(description="The complete, ready-to-run SQL query")


class _ProposeFeature(BaseModel):
    name: str = Field(description="Output column name, e.g. 'amount_to_balance_ratio'")
    op: str = Field(description=(
        "One of: ratio, product (2-3 inputs); difference (2 inputs); zscore, log, normalize (1 input); "
        "power (1 input + params=[exponent]); weighted_sum (2-3 inputs + one weight per input in params)"
    ))
    inputs: List[str] = Field(description="Existing column name(s) the op consumes, in order — use the EXACT columns you saw in the data")
    params: List[float] = Field(default_factory=list, description="Numeric constants: power→[exponent], weighted_sum→one weight per input. Empty for the others.")
    rationale: str = Field(default="", description="Why this feature matters for this dataset")


def build_expert_agent(expert: Expert, run_sql_tool, *, engine_label: str, domain_descriptor: str):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool(args_schema=_RecordFindings)
    def record_findings(summary: str, key_signals: List[str] = [], recommended_target: Optional[str] = None,
                        recommended_features: List[str] = [], risks: List[str] = [], interpretation_notes: str = "") -> dict:
        """Record your final structured findings. Call this EXACTLY ONCE as your last step."""
        return {"status": "recorded"}

    @tool(args_schema=_RecordQuery)
    def record_query(title: str, description: str = "", sql: str = "") -> dict:
        """Save a useful analytical query you discovered. Call up to 3 times — once per genuinely useful query that reveals important patterns, risks, or business insights about this dataset. Only save queries you actually ran and that returned meaningful results."""
        return {"status": "recorded", "title": title}

    @tool(args_schema=_ProposeFeature)
    def propose_feature(name: str, op: str, inputs: List[str], params: List[float] = [], rationale: str = "") -> dict:
        """Propose ONE executable, deterministic feature to engineer. Call up to 3 times.

        Pick an `op` from the fixed palette and reference the EXACT existing columns you saw:
          - ratio(a, b[, c])        → a / |b|  (or a / |b*c|)   e.g. max_amount / avg_amount
          - product(a, b[, c])      → a * b (* c)
          - difference(a, b)        → a - b
          - power(a) params=[k]     → a ** k                    e.g. squared / sqrt features
          - weighted_sum(a, b[, c]) params=[w1, w2(, w3)] → w1*a + w2*b (+ w3*c)
          - zscore(a)               → standardized a (how far from the population mean)
          - log(a)                  → log1p of a (compress heavy-tailed amounts)
          - normalize(a)            → min-max scale a to [0, 1]
        `params` is empty except for power and weighted_sum. Only propose a feature when your data
        investigation showed it carries signal. Do NOT use the target column or anything derived
        from it as an input (leakage)."""
        return {"status": "recorded", "name": name}

    system_prompt = (
        f"You are the {expert.name} of a {engine_label} for {domain_descriptor}.\n"
        f"Persona: {expert.summary}\n"
        f"You care about these signals: {', '.join(expert.signals)}.\n\n"
        "WORKFLOW:\n"
        "1. Use the run_sql tool to investigate the ACTUAL data (counts, distributions, group-bys, "
        "null rates, class balance, ranges). Run at most 4-5 focused aggregate queries.\n"
        "2. Ground every claim in numbers you actually retrieved — do not guess.\n"
        "3. Call record_query (up to 3 times) for each query you ran that revealed a genuinely useful "
        "pattern, risk, or business insight. Include the exact SQL you used.\n"
        "4. Call propose_feature (up to 3 times) for executable features your investigation justifies — "
        "YOU decide which transforms matter for THIS dataset, referencing the real columns. The pipeline "
        "will build exactly these deterministically. Never use the target (or its derivatives) as an input.\n"
        "5. Finally call record_findings EXACTLY ONCE with your structured conclusions "
        "(summary, key_signals with numbers, recommended_target, recommended_features, risks, "
        "interpretation_notes). Do not call run_sql after record_findings.\n"
        "Only reference columns/tables that exist in the schema below."
    )

    agent = create_agent(
        model=get_llm(temperature=0),
        tools=[run_sql_tool, record_findings, record_query, propose_feature],
        system_prompt=system_prompt,
    )
    return agent


# Derived from the single source of truth (schemas.FEATURE_OP_SPEC). Kept as
# module-level names because the financial shim + offline tests import them.
_VALID_FEATURE_OPS = set(FEATURE_OP_SPEC.keys())
_FEATURE_OP_ARITY = {op: spec[0] for op, spec in FEATURE_OP_SPEC.items()}  # min inputs (compat)


def parse_feature_spec(args: dict):
    """Build a validated FeatureSpec from a propose_feature tool call, or None."""
    from src.domains.schemas import FeatureSpec

    name = (args.get("name") or "").strip()
    op = (args.get("op") or "").strip().lower()
    inputs = [str(c).strip() for c in (args.get("inputs") or []) if str(c).strip()]
    try:
        params = [float(v) for v in (args.get("params") or [])]
    except (TypeError, ValueError):
        return None
    if not name or op not in FEATURE_OP_SPEC:
        return None
    if not validate_feature_spec_shape(op, len(inputs), len(params)):
        return None
    try:
        return FeatureSpec(name=name, op=op, inputs=inputs, params=params, rationale=args.get("rationale", ""))
    except Exception:
        return None


def extract_finding(expert_key: str, result: dict) -> ExpertFinding:
    """Pull the record_findings args + run_sql audit trail + record_query/propose_feature calls."""
    from src.domains.schemas import DiscoveredQuery, FeatureSpec

    findings_args: Optional[dict] = None
    queries: list[str] = []
    discovered: list[DiscoveredQuery] = []
    feature_specs: list[FeatureSpec] = []

    for msg in result.get("messages", []):
        for tc in (getattr(msg, "tool_calls", None) or []):
            name = tc.get("name")
            args = tc.get("args", {}) or {}
            if name == "record_findings" and findings_args is None:
                findings_args = args
            elif name == "run_sql" and args.get("sql"):
                queries.append(args["sql"])
            elif name == "record_query" and args.get("sql") and args.get("title"):
                discovered.append(DiscoveredQuery(
                    title=args["title"],
                    description=args.get("description", ""),
                    sql=args["sql"],
                ))
            elif name == "propose_feature":
                spec = parse_feature_spec(args)
                if spec is not None:
                    feature_specs.append(spec)

    if findings_args is None:
        last = result.get("messages", [])[-1] if result.get("messages") else None
        text = getattr(last, "content", "") if last else ""
        if isinstance(text, list):
            text = " ".join(str(p) for p in text)
        return ExpertFinding(expert=expert_key, summary=str(text)[:1000], queries_used=queries,
                             discovered_queries=discovered, feature_specs=feature_specs)

    return ExpertFinding(
        expert=expert_key,
        summary=findings_args.get("summary", ""),
        key_signals=findings_args.get("key_signals") or [],
        recommended_target=findings_args.get("recommended_target"),
        recommended_features=findings_args.get("recommended_features") or [],
        risks=findings_args.get("risks") or [],
        queries_used=queries,
        interpretation_notes=findings_args.get("interpretation_notes", ""),
        discovered_queries=discovered,
        feature_specs=feature_specs,
    )


async def run_one_expert(expert: Expert, user_query: str, db_context: str, domain_doc: str,
                         run_sql_tool, *, engine_label: str, domain_descriptor: str) -> Optional[ExpertFinding]:
    try:
        agent = build_expert_agent(expert, run_sql_tool, engine_label=engine_label, domain_descriptor=domain_descriptor)
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
        return extract_finding(expert.key, result)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"[{engine_label}] expert '{expert.key}' failed: {e}")
        return None


async def synthesize(findings: List[ExpertFinding], user_query: str, *, engine_label: str) -> str:
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
            ("system", f"You are the lead analyst of a {engine_label} council. Merge the experts' "
                       "findings into one coherent, non-repetitive recommendation: the most important signals, "
                       "a single recommended modeling approach/target, the top features, and the key risks."),
            ("human", f"User goal: {user_query}\n\nExpert findings:\n{bullets}"),
        ])
        content = resp.content
        return content if isinstance(content, str) else " ".join(str(p) for p in content)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"[{engine_label}] synthesis failed: {e}")
        return "\n\n".join(f"[{f.expert}] {f.summary}" for f in findings)


async def run_council(
    *,
    pack_name: str,
    expert_keys: List[str],
    is_council: bool,
    primary_expert: str,
    user_query: str,
    db_intelligence: dict,
    domain_doc: str,
    data_group_id: int,
    user_id: int,
    source_type: str,
    run_one_expert_fn: Callable[..., Awaitable[Optional[ExpertFinding]]],
    synthesize_fn: Callable[..., Awaitable[str]],
) -> Optional[MerchantAnalysis]:
    """Run the selected expert agents and synthesize a MerchantAnalysis. None on failure.

    ``run_one_expert_fn(key, user_query, db_context, domain_doc, run_sql_tool)`` and
    ``synthesize_fn(findings, user_query)`` are injected so a pack can supply a
    patchable seam (financial) or the generic closures (other packs).
    """
    if not expert_keys:
        return None
    # Lazy import: the source-agnostic SQL tool lives under the financial package;
    # importing it at module top would create an import cycle (financial → generic).
    from src.domains.financial.tools import make_sql_tool
    run_sql_tool = make_sql_tool(data_group_id, user_id)
    db_context = db_summary(db_intelligence, source_type)

    results = await asyncio.gather(
        *[run_one_expert_fn(k, user_query, db_context, domain_doc, run_sql_tool) for k in expert_keys]
    )
    findings = [f for f in results if f is not None]
    if not findings:
        return None

    synthesis = await synthesize_fn(findings, user_query)
    return MerchantAnalysis(
        pack_name=pack_name,
        primary_expert=primary_expert,
        council=expert_keys,
        is_council=is_council,
        findings=findings,
        synthesis=synthesis,
    )
