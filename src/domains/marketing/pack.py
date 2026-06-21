"""The Marketing Intelligence pack.

A config-only subclass of :class:`GenericDomainPack`: it declares the marketing
experts and framing labels. All four prompt hooks, expert routing and the agentic
``analyze`` are inherited from the generic base. Sector-specific deep-analysis
modules are registered under the ``marketing`` key in
``src/services/analysis_engine.py`` and selected via :attr:`analysis_type`.
"""

from __future__ import annotations

from src.domains.generic.pack import GenericDomainPack
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS


class MarketingPack(GenericDomainPack):
    name = "marketing"
    label = "Pazarlama / Marketing Intelligence"
    experts = ALL_EXPERTS
    expert_keywords = EXPERT_KEYWORDS
    default_expert = "segmentation"      # segmentation is the broad default view

    engine_label = "Marketing Intelligence Engine"
    lens_label = "MARKETING INTELLIGENCE"
    domain_label = "MARKETING"
    domain_descriptor = "marketing / campaign & channel data"
    domain_adjective = "marketing"
    default_analyst_label = "Marketing Analyst"
    analysis_type = "marketing"
