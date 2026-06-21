"""The Healthcare Intelligence pack.

A config-only subclass of :class:`GenericDomainPack`: it declares the healthcare
experts and framing labels. All four prompt hooks, expert routing and the agentic
``analyze`` are inherited from the generic base. Sector-specific deep-analysis
modules are registered under the ``healthcare`` key in
``src/services/analysis_engine.py`` and selected via :attr:`analysis_type`.
"""

from __future__ import annotations

from src.domains.generic.pack import GenericDomainPack
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS


class HealthcarePack(GenericDomainPack):
    name = "healthcare"
    label = "Sağlık / Healthcare Intelligence"
    experts = ALL_EXPERTS
    expert_keywords = EXPERT_KEYWORDS
    default_expert = "cohort"            # cohort/outcome is the broad default view

    engine_label = "Healthcare Intelligence Engine"
    lens_label = "HEALTHCARE INTELLIGENCE"
    domain_label = "HEALTHCARE"
    domain_descriptor = "healthcare / clinical patient data"
    domain_adjective = "clinical"
    default_analyst_label = "Healthcare Analyst"
    analysis_type = "healthcare"
