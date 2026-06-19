"""Deterministic feature recipes for the financial pack.

These are the "beyond prompt enrichment" layer: instead of *suggesting* features
to the LLM (``feature_hints``), they **guarantee** them. They exploit a fact the
generic builder's LLM path cannot rely on — the feature builder's aggregation
emits a fixed naming convention (``sum_<m>``, ``avg_<m>``, ``max_<m>``,
``std_<m>``, ``count_<id>``). Because the pack *knows* this convention, it can
derive risk/revenue signals (volatility, amount spikes) that ``df.eval`` math
interactions could never name reliably.

Each recipe runs at model grain inside ``feature_builder.build_features`` and is
wrapped in try/except by the caller, so a misbehaving recipe is always inert.
"""

from __future__ import annotations

import re
from typing import Dict, List

from src.domains.base import FeatureRecipe

# Recipes below derive new columns from aggregated metric columns. We only fire
# them for experts where the signal is meaningful.
_RELEVANT_EXPERTS = {"fraud", "revenue", "payment"}

_EPS = 1e-9


def _paired_metric_bases(columns: List[str], prefix_a: str, prefix_b: str) -> List[str]:
    """Return the metric base names `m` for which both `<prefix_a>m` and
    `<prefix_b>m` exist among `columns` (e.g. avg_amount & std_amount)."""
    a = {c[len(prefix_a):] for c in columns if c.startswith(prefix_a)}
    b = {c[len(prefix_b):] for c in columns if c.startswith(prefix_b)}
    return sorted(a & b)


class VolatilityRecipe(FeatureRecipe):
    """Coefficient of variation per aggregated metric: ``cv_<m> = std_<m> / |avg_<m>|``.

    A high CV means the entity's transaction amounts are erratic — a classic
    fraud/risk and revenue-stability signal. Deterministic and leakage-safe
    (derived only from the entity's own aggregated metrics, never the label).
    """

    name = "financial.volatility_cv"
    description = "Coefficient of variation (std/avg) per aggregated metric"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return bool(_paired_metric_bases(list(df.columns), "avg_", "std_"))

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        out: Dict = {}
        for base in _paired_metric_bases(list(df.columns), "avg_", "std_"):
            denom = df[f"avg_{base}"].abs() + _EPS
            out[f"cv_{base}"] = (df[f"std_{base}"] / denom).fillna(0.0)
        return out


class SpikeRatioRecipe(FeatureRecipe):
    """Peak-to-average ratio per aggregated metric: ``peak_ratio_<m> = max_<m> / |avg_<m>|``.

    Captures single large outlier transactions relative to the entity's norm — a
    strong fraud signal (one anomalous high-value charge among many small ones).
    """

    name = "financial.spike_ratio"
    description = "Peak-to-average ratio (max/avg) per aggregated metric"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return bool(_paired_metric_bases(list(df.columns), "avg_", "max_"))

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        out: Dict = {}
        for base in _paired_metric_bases(list(df.columns), "avg_", "max_"):
            denom = df[f"avg_{base}"].abs() + _EPS
            out[f"peak_ratio_{base}"] = (df[f"max_{base}"] / denom).fillna(0.0)
        return out


_ALL_RECIPES: List[FeatureRecipe] = [VolatilityRecipe(), SpikeRatioRecipe()]


def recipes_for(expert_keys: List[str]) -> List[FeatureRecipe]:
    """Return the deterministic recipes relevant to the active expert(s)."""
    if not any(k in _RELEVANT_EXPERTS for k in (expert_keys or [])):
        return []
    return list(_ALL_RECIPES)
