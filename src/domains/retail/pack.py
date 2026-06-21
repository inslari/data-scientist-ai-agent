"""The Retail Intelligence pack.

A config-only subclass of :class:`GenericDomainPack`: it declares the retail
experts and framing labels. All four prompt hooks, expert routing and the agentic
``analyze`` are inherited from the generic base. Sector-specific deep-analysis
modules are registered under the ``retail`` key in
``src/services/analysis_engine.py`` and selected via :attr:`analysis_type`.
"""

from __future__ import annotations

from src.domains.generic.pack import GenericDomainPack
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS


class RetailPack(GenericDomainPack):
    name = "retail"
    label = "Perakende / Retail Intelligence"
    experts = ALL_EXPERTS
    expert_keywords = EXPERT_KEYWORDS
    default_expert = "demand"            # demand/sales is the broad default view

    engine_label = "Retail Intelligence Engine"
    lens_label = "RETAIL INTELLIGENCE"
    domain_label = "RETAIL"
    domain_descriptor = "retail / brick-and-mortar sales data"
    domain_adjective = "retail"
    default_analyst_label = "Retail Analyst"
    analysis_type = "retail"
