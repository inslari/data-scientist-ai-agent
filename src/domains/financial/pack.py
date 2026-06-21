"""The Financial / Merchant Intelligence pack.

Now a thin, config-only subclass of :class:`GenericDomainPack`: it declares the
five financial experts and the framing labels, and overrides only
:meth:`feature_recipes` (the one piece that is genuinely financial-specific). All
four prompt hooks, expert routing and the agentic ``analyze`` are inherited from
the generic base and render identical text to the original hand-written pack.
"""

from __future__ import annotations

from src.domains.generic.pack import GenericDomainPack
from src.domains.base import FeatureRecipe
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS
from .feature_recipes import recipes_for


class FinancialPack(GenericDomainPack):
    name = "financial"
    label = "Finansal / Merchant Intelligence"
    experts = ALL_EXPERTS
    expert_keywords = EXPERT_KEYWORDS
    default_expert = "revenue"

    engine_label = "Merchant Intelligence Engine"
    lens_label = "MERCHANT INTELLIGENCE"
    domain_label = "FINANCIAL"
    domain_descriptor = "financial/payments data"
    domain_adjective = "financial"
    default_analyst_label = "Financial Analyst"
    analysis_type = "fintech"

    # ── Deterministic feature recipes (executable, not prompt text) ──────────
    def feature_recipes(self, ctx, db_intelligence=None, analysis=None) -> list[FeatureRecipe]:
        """Guaranteed risk/revenue features derived from the aggregated metrics.

        Gated on the active expert(s): only fires for fraud/revenue/payment, where
        volatility and amount-spike signals are meaningful.
        """
        keys: list[str] = []
        if ctx is not None:
            keys = ctx.council if (ctx.is_council and ctx.council) else [ctx.primary_expert]
        keys = [k for k in keys if k]
        return recipes_for(keys)
