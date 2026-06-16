"""Expert routing for the financial pack.

Hybrid strategy: an LLM scores the query against the five experts and decides
whether one expert suffices or a *council* of several should collaborate. If the
LLM is unavailable or errors, a deterministic keyword scorer (TR + EN) provides
the same decision shape — which also makes routing unit-testable offline.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from src.domains.base import MerchantContext, trim_doc
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS

_EXPERT_KEYS = [e.key for e in ALL_EXPERTS]
_VALID = set(_EXPERT_KEYS)


# ── Deterministic keyword routing (fallback + offline testing) ──────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def keyword_scores(user_query: str) -> dict[str, int]:
    """Count matched keywords per expert. Handles Turkish suffixes via prefix
    matching for sufficiently long single-word keywords; multiword keywords use
    substring containment."""
    query_lower = user_query.lower()
    tokens = _tokenize(user_query)
    scores: dict[str, int] = {k: 0 for k in _EXPERT_KEYS}

    for key, keywords in EXPERT_KEYWORDS.items():
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


def keyword_route(user_query: str) -> MerchantContext:
    """Build a MerchantContext from keyword scores alone."""
    scores = keyword_scores(user_query)
    ranked = sorted(_EXPERT_KEYS, key=lambda k: scores[k], reverse=True)

    relevant = [k for k in ranked if scores[k] > 0]
    if not relevant:
        # No domain keyword hit → default to a broad reporting persona (revenue),
        # single-expert, low confidence.
        return MerchantContext(
            pack_name="financial",
            primary_expert="revenue",
            council=["revenue"],
            sub_intent="",
            domain_framing="No strong domain signal detected; defaulting to a general financial view.",
            is_council=False,
        )

    primary = relevant[0]
    is_council = len(relevant) >= 2
    council = relevant if is_council else [primary]
    return MerchantContext(
        pack_name="financial",
        primary_expert=primary,
        council=council,
        sub_intent="",
        domain_framing="",
        is_council=is_council,
    )


# ── LLM routing ─────────────────────────────────────────────────────────────
class RouterDecision(BaseModel):
    primary_expert: str = Field(description=f"The single most relevant expert key. One of: {_EXPERT_KEYS}")
    council: list[str] = Field(
        default_factory=list,
        description="All experts (keys) that should contribute, ordered by relevance, including the primary.",
    )
    is_council: bool = Field(
        default=False,
        description="True only if the query genuinely spans 2+ expert domains.",
    )
    sub_intent: str = Field(default="", description="Short phrase describing the specific analytical goal.")
    domain_framing: str = Field(default="", description="One sentence framing the query in financial terms.")


def _expert_catalog() -> str:
    return "\n".join(f"- {e.key}: {e.name} — {e.summary}" for e in ALL_EXPERTS)


def llm_route(
    user_query: str,
    db_intelligence: Optional[dict] = None,
    domain_doc: str = "",
) -> Optional[MerchantContext]:
    """Try LLM routing; return None on any failure so the caller can fall back."""
    try:
        from src.core.llm import get_llm
        from src.core.config import settings

        system = (
            "You are the router of a Merchant Intelligence Engine for financial/payments data. "
            "Route the user's request to the most relevant expert(s).\n\n"
            "Experts:\n"
            f"{_expert_catalog()}\n\n"
            "Rules:\n"
            "- Pick exactly one primary_expert (the dominant intent).\n"
            "- Set is_council=true and list multiple experts in `council` ONLY when the request "
            "clearly spans more than one domain (e.g. revenue AND customer segments).\n"
            "- `council` must always include the primary_expert first.\n"
            "- Use only the expert keys listed above."
        )
        doc = trim_doc(domain_doc, max_chars=2000)
        human = f"User request: {user_query}"
        if doc:
            human += f"\n\nDataset description (for context):\n{doc}"

        llm = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
        decision: RouterDecision = llm.with_structured_output(RouterDecision).invoke(
            [("system", system), ("human", human)]
        )  # type: ignore[assignment]

        primary = decision.primary_expert.strip().lower()
        if primary not in _VALID:
            return None

        council = [c.strip().lower() for c in (decision.council or []) if c.strip().lower() in _VALID]
        if primary not in council:
            council = [primary] + council
        is_council = bool(decision.is_council and len(council) >= 2)

        return MerchantContext(
            pack_name="financial",
            primary_expert=primary,
            council=council if is_council else [primary],
            sub_intent=decision.sub_intent or "",
            domain_framing=decision.domain_framing or "",
            is_council=is_council,
        )
    except Exception as exc:
        print(f"[financial.router] LLM routing failed ({exc}); using keyword fallback.")
        return None


def route(
    user_query: str,
    db_intelligence: Optional[dict] = None,
    domain_doc: str = "",
) -> MerchantContext:
    """LLM-first routing with deterministic keyword fallback."""
    ctx = llm_route(user_query, db_intelligence, domain_doc)
    if ctx is not None:
        return ctx
    return keyword_route(user_query)
