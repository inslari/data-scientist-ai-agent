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

import numpy as np
import pandas as pd

from src.domains.base import FeatureRecipe

# Recipes below derive new columns from aggregated metric columns. We only fire
# them for experts where the signal is meaningful.
_RELEVANT_EXPERTS = {"fraud", "revenue", "payment"}

_EPS = 1e-9

# Column-name patterns that mark a numeric column as a *monetary* magnitude.
# Used by LogAmountRecipe: financial amounts are almost always right-skewed, so a
# log transform is the single most reliable variance-stabilising move (mirrors
# the credit-card notebook's ``Amount_transformed = log10(Amount + eps)``).
_MONEY_RE = re.compile(
    r"(amount|amt|value|price|balance|revenue|sales|income|salary|cost|charge|"
    r"payment|spend|txn_value|transaction|fee|premium|limit|debit|credit)",
    re.IGNORECASE,
)

# Column-name patterns that mark a column as a timestamp we can decompose. We do
# NOT match bare "time-since" integer counters here — only real datetimes are
# parsed, so an already-elapsed-seconds column is left to the generic builder.
_TIME_RE = re.compile(r"(timestamp|datetime|_date|^date|date_|_time$|^time$|_at$)", re.IGNORECASE)

# Derived-column prefixes we must never re-transform (avoid log(log) / ratios).
_SKIP_PREFIXES = ("log_", "cv_", "peak_ratio_", "count_", "hour_", "dow_", "is_")


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


def _is_money_col(name: str) -> bool:
    if any(name.startswith(p) for p in _SKIP_PREFIXES):
        return False
    return bool(_MONEY_RE.search(name))


def _money_cols(df) -> List[str]:  # noqa: ANN001
    """Numeric, non-negative, right-skewed columns that look monetary."""
    out: List[str] = []
    for c in df.columns:
        if not _is_money_col(str(c)):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        s = s.dropna()
        if s.empty or float(s.min()) < 0:
            continue                       # log needs non-negative support
        # Only transform genuinely skewed magnitudes; flat columns gain nothing.
        try:
            if len(s) >= 8 and float(s.skew()) < 1.0:
                continue
        except Exception:
            pass
        out.append(c)
    return out


class LogAmountRecipe(FeatureRecipe):
    """Log-transform skewed monetary columns: ``log_<m> = log10(<m> + eps)``.

    Financial amounts (transaction value, balance, revenue) are heavily
    right-skewed; a handful of large transactions dominate the raw scale and
    distort distance-, tree-split- and Gaussian-density-based models alike. The
    log compresses that tail into a near-symmetric distribution — the exact move
    the credit-card anomaly notebook makes on ``Amount`` before fitting its
    density model. Deterministic and leakage-safe (a pure per-row transform of an
    existing feature, never the label).
    """

    name = "financial.log_amount"
    description = "log10(x + eps) of skewed monetary columns"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return bool(_money_cols(df))

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        out: Dict = {}
        for c in _money_cols(df):
            name = f"log_{c}"
            if name in df.columns:
                continue
            out[name] = np.log10(df[c].astype(float).clip(lower=0.0) + 1e-3)
        return out


def _time_cols(df) -> List[str]:  # noqa: ANN001
    """Columns that are (or parse to) real datetimes, by dtype or name."""
    out: List[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
            continue
        if _TIME_RE.search(str(c)) and (s.dtype == object or pd.api.types.is_string_dtype(s)):
            out.append(c)
    return out


def _to_datetime(s):  # noqa: ANN001
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


class TimePartsRecipe(FeatureRecipe):
    """Decompose a transaction timestamp into fraud-relevant time-of-day signals.

    From a datetime column we derive the hour-of-day (the notebook's ``Hour``
    feature, a strong fraud signal — fraud clusters at unusual hours) plus its
    cyclical sine/cosine encoding (so 23:00 and 00:00 sit adjacent for
    distance/density models) and the day-of-week. Leakage-safe: every output is a
    deterministic function of the timestamp alone.
    """

    name = "financial.time_parts"
    description = "hour-of-day (+cyclical) and day-of-week from a timestamp"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        for c in _time_cols(df):
            dt = _to_datetime(df[c])
            if dt.notna().any():
                return True
        return False

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        out: Dict = {}
        for c in _time_cols(df):
            dt = _to_datetime(df[c])
            if not dt.notna().any():
                continue
            base = re.sub(r"\W+", "_", str(c)).strip("_").lower()
            hour = dt.dt.hour.astype("float")
            if f"hour_{base}" not in df.columns:
                out[f"hour_{base}"] = hour.fillna(-1.0)
                out[f"hour_sin_{base}"] = np.sin(2 * np.pi * hour / 24.0).fillna(0.0)
                out[f"hour_cos_{base}"] = np.cos(2 * np.pi * hour / 24.0).fillna(0.0)
                out[f"dow_{base}"] = dt.dt.dayofweek.astype("float").fillna(-1.0)
            break  # one timestamp's parts are enough; avoid column explosion
        return out


_ALL_RECIPES: List[FeatureRecipe] = [
    VolatilityRecipe(),
    SpikeRatioRecipe(),
    LogAmountRecipe(),
    TimePartsRecipe(),
]


def recipes_for(expert_keys: List[str]) -> List[FeatureRecipe]:
    """Return the deterministic recipes relevant to the active expert(s)."""
    if not any(k in _RELEVANT_EXPERTS for k in (expert_keys or [])):
        return []
    return list(_ALL_RECIPES)
