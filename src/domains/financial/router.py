"""Expert routing for the financial pack — a thin binding over the generic router.

The actual hybrid routing logic (LLM-first + deterministic keyword fallback) lives
in :mod:`src.domains.generic.router`; here we just bind it to the financial pack's
experts, keywords and framing labels. The module-level function signatures are kept
stable because the offline tests import ``keyword_route`` / ``keyword_scores`` from
here.
"""

from __future__ import annotations

from typing import Optional

from src.domains.base import MerchantContext
from src.domains.generic import router as _g
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS

_PACK_NAME = "financial"
_DEFAULT_EXPERT = "revenue"
_ENGINE_LABEL = "Merchant Intelligence Engine"
_DOMAIN_DESCRIPTOR = "financial/payments data"

_EXPERT_KEYS = [e.key for e in ALL_EXPERTS]


def keyword_scores(user_query: str) -> dict[str, int]:
    return _g.keyword_scores(user_query, EXPERT_KEYWORDS, _EXPERT_KEYS)


def keyword_route(user_query: str) -> MerchantContext:
    return _g.keyword_route(
        user_query,
        pack_name=_PACK_NAME,
        experts=ALL_EXPERTS,
        expert_keywords=EXPERT_KEYWORDS,
        default_expert=_DEFAULT_EXPERT,
        domain_descriptor=_DOMAIN_DESCRIPTOR,
    )


def llm_route(
    user_query: str,
    db_intelligence: Optional[dict] = None,
    domain_doc: str = "",
) -> Optional[MerchantContext]:
    return _g.llm_route(
        user_query,
        pack_name=_PACK_NAME,
        experts=ALL_EXPERTS,
        engine_label=_ENGINE_LABEL,
        domain_descriptor=_DOMAIN_DESCRIPTOR,
        db_intelligence=db_intelligence,
        domain_doc=domain_doc,
    )


def route(
    user_query: str,
    db_intelligence: Optional[dict] = None,
    domain_doc: str = "",
) -> MerchantContext:
    return _g.route(
        user_query,
        pack_name=_PACK_NAME,
        experts=ALL_EXPERTS,
        expert_keywords=EXPERT_KEYWORDS,
        default_expert=_DEFAULT_EXPERT,
        engine_label=_ENGINE_LABEL,
        domain_descriptor=_DOMAIN_DESCRIPTOR,
        db_intelligence=db_intelligence,
        domain_doc=domain_doc,
    )
