"""
Domain (Sector) Pack framework.

A DomainPack is a pluggable layer that augments the (otherwise sector-agnostic)
data-science pipeline with domain expertise at four touch points:

    1. intent discovery
    2. query / feature-plan generation
    3. feature engineering
    4. output interpretation

Packs are selected purely by configuration (``settings.DOMAIN_PACK`` or a
per-run ``state["domain_pack"]`` override). When no pack is active the
``NullDomainPack`` is used and the pipeline behaves exactly as before — so the
whole layer is fully removable.

Add a new sector by creating ``src/domains/<sector>/pack.py`` and registering it
in ``registry.py``.
"""

from .base import DomainPack, Expert, MerchantContext, InterpretLens
from .null_pack import NullDomainPack
from .registry import get_active_pack, register_pack, list_packs

__all__ = [
    "DomainPack",
    "Expert",
    "MerchantContext",
    "InterpretLens",
    "NullDomainPack",
    "get_active_pack",
    "register_pack",
    "list_packs",
]
