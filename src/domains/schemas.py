"""Structured output of the agentic Merchant Intelligence layer.

Expert agents investigate the dataset and return an :class:`ExpertFinding`.
A pack's ``analyze()`` aggregates these into a :class:`MerchantAnalysis`, which
is serialized into pipeline state (``merchant_analysis``) and consumed by the
four touch points (intent / query / feature engineering / interpretation).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# The safe, parameterized op palette an expert agent (or the generic feature
# planner) may compose. Each op is a deterministic, leakage-checkable transform —
# the LLM never writes raw code, it only picks an op + the real columns/constants.
FeatureOp = Literal[
    "ratio", "product", "difference", "zscore", "log",
    "power", "weighted_sum", "normalize",
]

# Single source of truth for op shape: op -> (min_inputs, max_inputs, n_params).
# ``n_params == "match"`` means one numeric param per input (weighted_sum).
# Both the proposer (parse-time validation) and the builder (execution) consult
# this so the two can never drift.
FEATURE_OP_SPEC: dict[str, tuple[int, int, object]] = {
    "ratio":        (2, 3, 0),   # a / |b|  (or a / |b*c|) — velocity = distance/time
    "product":      (2, 3, 0),   # a * b (* c)             — momentum = mass*velocity
    "difference":   (2, 2, 0),   # a - b
    "zscore":       (1, 1, 0),   # standardized a
    "log":          (1, 1, 0),   # log1p(a)
    "power":        (1, 1, 1),   # a ** k  (k = params[0]) — height**2
    "weighted_sum": (2, 3, "match"),  # w1*a + w2*b (+ w3*c), one weight per input
    "normalize":    (1, 1, 0),   # min-max scale a to [0, 1]
}


def validate_feature_spec_shape(op: str, n_inputs: int, n_params: int) -> bool:
    """True iff ``op`` accepts this many inputs and params per FEATURE_OP_SPEC."""
    spec = FEATURE_OP_SPEC.get(op)
    if spec is None:
        return False
    min_in, max_in, n_param = spec
    if not (min_in <= n_inputs <= max_in):
        return False
    if n_param == "match":
        return n_params == n_inputs
    return n_params == n_param


class FeatureSpec(BaseModel):
    """An executable feature the LLM *decided* to build.

    The LLM (an expert agent, or the generic feature planner) picks an ``op``
    from the fixed palette and names the REAL input columns it saw while
    investigating the data — so there is no dependency on any internal naming
    convention. The feature builder validates the spec against the actual schema
    (columns exist, not the target/leaky) and applies the op deterministically.
    Invalid specs are skipped, never executed.
    """

    name: str = Field(description="Output column name, e.g. 'amount_to_balance_ratio'")
    op: FeatureOp = Field(description="Transform from the fixed palette (see FEATURE_OP_SPEC)")
    inputs: list[str] = Field(description="Existing column name(s) the op consumes, in order")
    params: list[float] = Field(
        default_factory=list,
        description="Numeric constants the op needs: power→[exponent], weighted_sum→one weight per input. Empty otherwise.",
    )
    rationale: str = Field(default="", description="Why this feature matters for this dataset")


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
    feature_specs: list[FeatureSpec] = Field(
        default_factory=list,
        description="Executable feature specs the expert decided to build (agent-driven)",
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
