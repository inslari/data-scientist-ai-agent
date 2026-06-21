"""Data-driven expert routing shared by every sector pack.

Hybrid strategy (identical in shape to the original financial router): an LLM
scores the query against the pack's experts and decides whether one expert
suffices or a *council* should collaborate. If the LLM is unavailable or errors,
a deterministic keyword scorer (TR + EN) provides the same decision shape — which
also makes routing unit-testable offline.

Every function takes the pack's experts/keywords/labels as parameters, so a
concrete pack is just data. See :mod:`src.domains.financial.router` for a thin
per-pack binding.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from src.domains.base import Expert, MerchantContext, trim_doc

from src.core.logging_config import get_logger
logger = get_logger(__name__)


# ── Deterministic keyword routing (fallback + offline testing) ──────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def keyword_scores(
    user_query: str,
    expert_keywords: dict[str, list[str]],
    expert_keys: Sequence[str],
) -> dict[str, int]:
    """Count matched keywords per expert. Handles Turkish suffixes via prefix
    matching for sufficiently long single-word keywords; multiword keywords use
    substring containment."""
    query_lower = user_query.lower()
    tokens = _tokenize(user_query)
    scores: dict[str, int] = {k: 0 for k in expert_keys}

    for key, keywords in expert_keywords.items():
        matched = 0
        for kw in keywords:
            kw = kw.lower()
            if " " in kw:
                if kw in query_lower:
                    matched += 1
            else:
                if kw in tokens:
                    matched += 1
                elif len(kw) >= 4 and any(t.startswith(kw) for t in tokens):
                    matched += 1
        scores[key] = matched
    return scores


def keyword_route(
    user_query: str,
    *,
    pack_name: str,
    experts: Sequence[Expert],
    expert_keywords: dict[str, list[str]],
    default_expert: str,
    domain_descriptor: str = "",
) -> MerchantContext:
    """Build a MerchantContext from keyword scores alone."""
    expert_keys = [e.key for e in experts]
    scores = keyword_scores(user_query, expert_keywords, expert_keys)
    ranked = sorted(expert_keys, key=lambda k: scores[k], reverse=True)

    relevant = [k for k in ranked if scores[k] > 0]
    if not relevant:
        # No domain keyword hit → default to the pack's broad persona, single
        # expert, low confidence.
        view = domain_descriptor or pack_name
        return MerchantContext(
            pack_name=pack_name,
            primary_expert=default_expert,
            council=[default_expert],
            sub_intent="",
            domain_framing=f"No strong domain signal detected; defaulting to a general {view} view.",
            is_council=False,
        )

    primary = relevant[0]
    is_council = len(relevant) >= 2
    council = relevant if is_council else [primary]
    return MerchantContext(
        pack_name=pack_name,
        primary_expert=primary,
        council=council,
        sub_intent="",
        domain_framing="",
        is_council=is_council,
    )


# ── LLM routing ─────────────────────────────────────────────────────────────
def _decision_model(expert_keys: Sequence[str]) -> type[BaseModel]:
    keys = list(expert_keys)

    class RouterDecision(BaseModel):
        primary_expert: str = Field(description=f"The single most relevant expert key. One of: {keys}")
        council: list[str] = Field(
            default_factory=list,
            description="All experts (keys) that should contribute, ordered by relevance, including the primary.",
        )
        is_council: bool = Field(
            default=False,
            description="True only if the query genuinely spans 2+ expert domains.",
        )
        sub_intent: str = Field(default="", description="Short phrase describing the specific analytical goal.")
        domain_framing: str = Field(default="", description="One sentence framing the query in this sector's terms.")

    return RouterDecision


def _expert_catalog(experts: Sequence[Expert]) -> str:
    return "\n".join(f"- {e.key}: {e.name} — {e.summary}" for e in experts)


def llm_route(
    user_query: str,
    *,
    pack_name: str,
    experts: Sequence[Expert],
    engine_label: str,
    domain_descriptor: str,
    db_intelligence: Optional[dict] = None,
    domain_doc: str = "",
) -> Optional[MerchantContext]:
    """Try LLM routing; return None on any failure so the caller can fall back."""
    valid = {e.key for e in experts}
    try:
        from src.core.llm import get_llm
        from src.core.config import settings

        system = (
            f"You are the router of a {engine_label} for {domain_descriptor}. "
            "Route the user's request to the most relevant expert(s).\n\n"
            "Experts:\n"
            f"{_expert_catalog(experts)}\n\n"
            "Rules:\n"
            "- Pick exactly one primary_expert (the dominant intent).\n"
            "- Set is_council=true and list multiple experts in `council` ONLY when the request "
            "clearly spans more than one domain.\n"
            "- `council` must always include the primary_expert first.\n"
            "- Use only the expert keys listed above."
        )
        doc = trim_doc(domain_doc, max_chars=2000)
        human = f"User request: {user_query}"
        if doc:
            human += f"\n\nDataset description (for context):\n{doc}"

        RouterDecision = _decision_model([e.key for e in experts])
        llm = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
        decision = llm.with_structured_output(RouterDecision).invoke(
            [("system", system), ("human", human)]
        )

        primary = decision.primary_expert.strip().lower()
        if primary not in valid:
            return None

        council = [c.strip().lower() for c in (decision.council or []) if c.strip().lower() in valid]
        if primary not in council:
            council = [primary] + council
        is_council = bool(decision.is_council and len(council) >= 2)

        return MerchantContext(
            pack_name=pack_name,
            primary_expert=primary,
            council=council if is_council else [primary],
            sub_intent=decision.sub_intent or "",
            domain_framing=decision.domain_framing or "",
            is_council=is_council,
        )
    except Exception as exc:
        logger.error(f"[{pack_name}.router] LLM routing failed ({exc}); using keyword fallback.")
        return None


def route(
    user_query: str,
    *,
    pack_name: str,
    experts: Sequence[Expert],
    expert_keywords: dict[str, list[str]],
    default_expert: str,
    engine_label: str,
    domain_descriptor: str,
    db_intelligence: Optional[dict] = None,
    domain_doc: str = "",
) -> MerchantContext:
    """LLM-first routing with deterministic keyword fallback."""
    ctx = llm_route(
        user_query,
        pack_name=pack_name,
        experts=experts,
        engine_label=engine_label,
        domain_descriptor=domain_descriptor,
        db_intelligence=db_intelligence,
        domain_doc=domain_doc,
    )
    if ctx is not None:
        return ctx
    return keyword_route(
        user_query,
        pack_name=pack_name,
        experts=experts,
        expert_keywords=expert_keywords,
        default_expert=default_expert,
        domain_descriptor=domain_descriptor,
    )
