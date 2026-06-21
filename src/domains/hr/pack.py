"""The HR / People Analytics Intelligence pack.

A config-only subclass of :class:`GenericDomainPack`: it declares the HR experts
and framing labels. All four prompt hooks, expert routing and the agentic
``analyze`` are inherited from the generic base. Sector-specific deep-analysis
modules are registered under the ``hr`` key in
``src/services/analysis_engine.py`` and selected via :attr:`analysis_type`.
"""

from __future__ import annotations

from src.domains.generic.pack import GenericDomainPack
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS


class HRPack(GenericDomainPack):
    name = "hr"
    label = "İK / HR People Analytics"
    experts = ALL_EXPERTS
    expert_keywords = EXPERT_KEYWORDS
    default_expert = "attrition"         # attrition is the broad default view

    engine_label = "HR People Analytics Engine"
    lens_label = "HR PEOPLE ANALYTICS"
    domain_label = "HR"
    domain_descriptor = "HR / workforce people-analytics data"
    domain_adjective = "people-analytics"
    default_analyst_label = "People Analytics Analyst"
    analysis_type = "hr"
