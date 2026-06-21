"""The E-Commerce Intelligence pack.

A config-only subclass of :class:`GenericDomainPack`: it declares the e-commerce
experts and framing labels. All four prompt hooks, expert routing and the agentic
``analyze`` are inherited from the generic base. Sector-specific deep-analysis
modules are registered under the ``ecommerce`` key in
``src/services/analysis_engine.py`` and selected via :attr:`analysis_type`.
"""

from __future__ import annotations

from src.domains.generic.pack import GenericDomainPack
from .experts import ALL_EXPERTS, EXPERT_KEYWORDS


class EcommercePack(GenericDomainPack):
    name = "ecommerce"
    label = "E-Ticaret / E-Commerce Intelligence"
    experts = ALL_EXPERTS
    expert_keywords = EXPERT_KEYWORDS
    default_expert = "customer"          # RFM is the broad default view

    engine_label = "E-Commerce Intelligence Engine"
    lens_label = "E-COMMERCE INTELLIGENCE"
    domain_label = "E-COMMERCE"
    domain_descriptor = "e-commerce / online retail data"
    domain_adjective = "e-commerce"
    default_analyst_label = "E-Commerce Analyst"
    analysis_type = "ecommerce"
