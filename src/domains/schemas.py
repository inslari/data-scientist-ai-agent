"""Structured output of the agentic Merchant Intelligence layer.

Expert agents investigate the dataset and return an :class:`ExpertFinding`.
A pack's ``analyze()`` aggregates these into a :class:`MerchantAnalysis`, which
is serialized into pipeline state (``merchant_analysis``) and consumed by the
four touch points (intent / query / feature engineering / interpretation).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DiscoveredQuery(BaseModel):
    """A reusable analytical query the expert agent surfaced while investigating."""

    title: str = Field(description="Short descriptive name for this query")
    description: str = Field(default="", description="What this query reveals and when to use it")
    sql: str = Field(description="The complete, ready-to-run SQL query")


class ExpertFinding(BaseModel):
    """What a single expert agent concluded after investigating the data."""

    expert: str = Field(description="Expert key, e.g. 'fraud'")
    summary: str = Field(default="", description="One-paragraph conclusion from this expert")
    key_signals: list[str] = Field(
        default_factory=list,
        description="Data-grounded observations, ideally with concrete numbers",
    )
    recommended_target: Optional[str] = Field(
        default=None, description="Column the expert recommends predicting, if any"
    )
    recommended_features: list[str] = Field(
        default_factory=list, description="Concrete domain features worth engineering"
    )
    risks: list[str] = Field(
        default_factory=list, description="Leakage / data-quality / interpretation risks"
    )
    queries_used: list[str] = Field(
        default_factory=list, description="SQL the agent actually ran (audit trail)"
    )
    interpretation_notes: str = Field(
        default="", description="How downstream reporting should read the results"
    )
    discovered_queries: list[DiscoveredQuery] = Field(
        default_factory=list,
        description="Useful analytical queries the expert saved for reuse",
    )


class MerchantAnalysis(BaseModel):
    """Aggregated agentic analysis for the active pack."""

    pack_name: str = ""
    primary_expert: str = ""
    council: list[str] = Field(default_factory=list)
    is_council: bool = False
    findings: list[ExpertFinding] = Field(default_factory=list)
    synthesis: str = ""

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["MerchantAnalysis"]:
        if not data:
            return None
        try:
            return cls.model_validate(data)
        except Exception:
            return None

    def finding_for(self, expert_key: str) -> Optional[ExpertFinding]:
        for f in self.findings:
            if f.expert == expert_key:
                return f
        return None
