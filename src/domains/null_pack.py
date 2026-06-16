"""The no-op pack used when no domain pack is active.

Returns an empty context and empty hints everywhere, so the pipeline behaves
identically to a build without the domain layer. This is what makes the whole
feature "pluggable / removable".
"""

from __future__ import annotations

from typing import Optional

from .base import DomainPack, MerchantContext


class NullDomainPack(DomainPack):
    name = "none"
    label = "Genel (sektör katmanı kapalı)"
    experts = []

    def select_experts(
        self,
        user_query: str,
        db_intelligence: Optional[dict] = None,
        domain_doc: str = "",
    ) -> MerchantContext:
        return MerchantContext(pack_name="none", primary_expert="")

    # All other hooks fall back to DomainPack's empty-string defaults.
