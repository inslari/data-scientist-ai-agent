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

# Recipes below derive new columns from existing ones. Each is gated to the
# expert(s) where its signal is meaningful via ``_EXPERT_RECIPES`` (see bottom).
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

# Column-name patterns that mark a numeric column as a *loan term / horizon*: the
# number of periods a credit runs for. Paired with a monetary amount by
# InstallmentBurdenRecipe to derive the per-period repayment burden — the single
# strongest default-risk signal the German-credit notebook never builds.
_TERM_RE = re.compile(
    r"(duration|tenor|term|maturity|installment|instalment|months?|^period$|"
    r"_period$|vade|sure|süre|taksit)",
    re.IGNORECASE,
)

# Column-name patterns that mark a monetary column as *recurring income / capacity*
# rather than an obligation. AffordabilityRatioRecipe divides an obligation
# (purchase/loan/debt) by income to get the affordability ratio — the legit,
# leakage-safe version of the BNPL notebook's ``income_to_credit`` hand-feature.
_INCOME_RE = re.compile(r"(income|salary|wage|earnings|paycheck|net_pay|gross_pay)", re.IGNORECASE)

# Column-name patterns that mark a numeric column as a *relationship-depth count*:
# how many products / accounts / services a customer holds. The banking-churn
# notebook bins ``NumOfProducts`` (1 / 2 / >2) because the churn relationship is
# non-monotonic — single-product customers churn far more than multi-product
# ones. HoldingCountBandRecipe generalises that into single/multi flags.
_HOLDING_RE = re.compile(
    r"(num_?of_?products|numofproducts|product_?count|products?|holdings?|"
    r"num_?accounts|accounts_?held|services?_?held|subscriptions?|"
    r"urun_?sayisi|ürün_?sayisi)",
    re.IGNORECASE,
)

# Bounds for the zero-value dormancy flag: only fire when zeros are a meaningful
# minority/majority (a column that is all-zero or has no zeros gains nothing).
_ZERO_FLAG_MIN_FRAC = 0.01
_ZERO_FLAG_MAX_FRAC = 0.99

# A holding count above this is treated as an id/measurement, not a product tally.
_HOLDING_MAX = 50

# Derived-column prefixes we must never re-transform (avoid log(log) / ratios).
_SKIP_PREFIXES = ("log_", "cv_", "peak_ratio_", "count_", "hour_", "dow_", "is_", "burden_")


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


# Words that make a column read as a *score/ratio/flag*, not a monetary magnitude,
# even when _MONEY_RE matches a substring (e.g. "credit_score" matches "credit").
# Used to keep ratio recipes from dividing by a credit score or an existing ratio.
_NOT_AMOUNT_RE = re.compile(
    r"(score|rating|grade|rank|ratio|rate|pct|percent|flag|count|index|segment|class)",
    re.IGNORECASE,
)


def _is_money_col(name: str) -> bool:
    if any(name.startswith(p) for p in _SKIP_PREFIXES):
        return False
    return bool(_MONEY_RE.search(name))


def _is_amount_col(name: str) -> bool:
    """A genuine monetary *magnitude* (numerator/denominator material): money-named
    but not a score/ratio/flag like ``credit_score`` or ``debt_to_income_ratio``."""
    return _is_money_col(name) and not _NOT_AMOUNT_RE.search(name)


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


def _ratio_amount_cols(df) -> List[str]:  # noqa: ANN001
    """Monetary, numeric, non-negative columns usable as a ratio numerator.

    Unlike :func:`_money_cols` this does NOT apply the skew gate — we want the raw
    principal/credit amount as a ratio input regardless of its distribution shape.
    Returned sorted by name for deterministic pairing.
    """
    out: List[str] = []
    for c in df.columns:
        name = str(c)
        if not _is_amount_col(name):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        s = s.dropna()
        if s.empty or float(s.min()) < 0:
            continue
        out.append(name)
    return sorted(out)


def _term_cols(df) -> List[str]:  # noqa: ANN001
    """Numeric, strictly-positive columns that look like a loan term/horizon.

    Excludes derived prefixes and any column that also reads as monetary (so an
    "amount" never doubles as a "term"). Sorted by name for deterministic pairing.
    """
    out: List[str] = []
    for c in df.columns:
        name = str(c)
        if any(name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if not _TERM_RE.search(name) or _is_money_col(name):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        s = s.dropna()
        if s.empty or float(s.max()) <= 0:
            continue
        out.append(name)
    return sorted(out)


class InstallmentBurdenRecipe(FeatureRecipe):
    """Per-period repayment burden: ``burden_<amount>_per_<term> = amount / term``.

    Credit risk is driven far less by the raw loan size than by how heavy the
    *recurring* payment is: a large loan over a long term can be safer than a
    small one crammed into a few months. The German-credit notebook models
    ``Credit amount`` and ``Duration`` only in isolation and never forms their
    ratio — yet the monthly installment (amount / duration) is one of the
    strongest default predictors a credit scorecard has.

    We pair the first monetary amount with the first term-like column (both name
    sorted for determinism) and emit a single ratio. Deterministic and
    leakage-safe: a pure per-row function of two existing features, never the
    label.
    """

    name = "financial.installment_burden"
    description = "Per-period repayment burden (amount / term)"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return bool(_ratio_amount_cols(df)) and bool(_term_cols(df))

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        amounts = _ratio_amount_cols(df)
        terms = _term_cols(df)
        if not amounts or not terms:
            return {}
        amt, term = amounts[0], terms[0]
        col = f"burden_{amt}_per_{term}"
        if col in df.columns:
            return {}
        denom = df[term].astype(float).abs() + _EPS
        return {col: (df[amt].astype(float) / denom).fillna(0.0)}


def _income_cols(df) -> List[str]:  # noqa: ANN001
    """Monetary, numeric, strictly-positive columns that read as recurring income."""
    out: List[str] = []
    for c in df.columns:
        name = str(c)
        if any(name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if not _INCOME_RE.search(name):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        s = s.dropna()
        if s.empty or float(s.max()) <= 0:
            continue
        out.append(name)
    return sorted(out)


def _obligation_cols(df) -> List[str]:  # noqa: ANN001
    """Monetary magnitudes that are an *obligation* (purchase/loan/debt), not income."""
    out: List[str] = []
    for c in df.columns:
        name = str(c)
        if not _is_amount_col(name) or _INCOME_RE.search(name):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        s = s.dropna()
        if s.empty or float(s.min()) < 0:
            continue
        out.append(name)
    return sorted(out)


class AffordabilityRatioRecipe(FeatureRecipe):
    """Obligation-to-income ratio: ``<obligation>_to_income = obligation / income``.

    Default risk is fundamentally about *capacity to repay*: a $5k purchase is
    trivial on a $200k income and crushing on a $15k one. The raw amount can't
    express that — only the ratio to income can. This is the leakage-safe,
    generalised form of the BNPL notebook's ``income_to_credit`` hand-feature
    (and stronger, since purchase/income beats income/credit-score as a capacity
    proxy). Both inputs are known at application time → no temporal leakage, and
    it never touches the label.

    Pairs the first obligation with the first income column (both name sorted for
    determinism) and emits a single ratio.
    """

    name = "financial.affordability_ratio"
    description = "Obligation-to-income ratio (amount / income)"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return bool(_obligation_cols(df)) and bool(_income_cols(df))

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        obligations = _obligation_cols(df)
        incomes = _income_cols(df)
        if not obligations or not incomes:
            return {}
        amt, inc = obligations[0], incomes[0]
        col = f"{amt}_to_income"
        if col in df.columns:
            return {}
        denom = df[inc].astype(float).abs() + _EPS
        return {col: (df[amt].astype(float) / denom).fillna(0.0)}


def _zero_balance_cols(df) -> List[str]:  # noqa: ANN001
    """Monetary magnitude columns with a meaningful share of exactly-zero rows.

    A zero balance / zero spend is a dormancy signal: in the banking-churn
    notebook ~36% of customers carry a zero balance and churn at a markedly
    higher rate, yet the raw amount can't separate "exactly zero" (dormant) from
    "small but active". We only fire when zeros are neither absent nor universal
    so the flag carries signal. Sorted by name for determinism.
    """
    out: List[str] = []
    for c in df.columns:
        name = str(c)
        if not _is_amount_col(name):          # money magnitude, not a score/ratio/flag
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        s = s.dropna()
        if s.empty or float(s.min()) < 0:
            continue
        zero_frac = float((s == 0).mean())
        if _ZERO_FLAG_MIN_FRAC <= zero_frac <= _ZERO_FLAG_MAX_FRAC:
            out.append(name)
    return sorted(out)


class ZeroBalanceFlagRecipe(FeatureRecipe):
    """Dormancy flag for a zero monetary balance: ``is_zero_<m> = (m == 0)``.

    Mirrors the churn notebook's ``Account_Balance`` engineered feature (Balance
    == 0 vs. > 0). Tree splits and linear models both read the raw balance as a
    magnitude and miss the sharp behavioural cliff at exactly zero (a dormant
    account is qualitatively, not just quantitatively, different from a small
    active one). Deterministic and leakage-safe: a pure per-row test of an
    existing feature, never the label.
    """

    name = "financial.zero_balance_flag"
    description = "Binary dormancy flag for an exactly-zero monetary balance"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return bool(_zero_balance_cols(df))

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        out: Dict = {}
        for c in _zero_balance_cols(df):
            name = f"is_zero_{c}"
            if name in df.columns:
                continue
            out[name] = (df[c].astype(float) == 0).astype(float)
        return out


def _holding_count_cols(df) -> List[str]:  # noqa: ANN001
    """Small non-negative integer columns that count products/accounts held.

    Excludes monetary columns and derived prefixes, and requires the column to
    actually span single (==1) and multi (>=2) holders — otherwise the bands
    carry no signal. Capped at ``_HOLDING_MAX`` so an id or measurement column
    that merely matches the name pattern is rejected. Sorted for determinism.
    """
    out: List[str] = []
    for c in df.columns:
        name = str(c)
        if any(name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if not _HOLDING_RE.search(name) or _is_money_col(name):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        s = s.dropna()
        if s.empty or float(s.min()) < 0 or float(s.max()) > _HOLDING_MAX:
            continue
        if not np.allclose(s.to_numpy(), s.round().to_numpy()):   # integer-valued only
            continue
        if not (bool((s == 1).any()) and bool((s >= 2).any())):   # must span single vs multi
            continue
        out.append(name)
    return sorted(out)


class HoldingCountBandRecipe(FeatureRecipe):
    """Single- vs multi-holding flags: ``<m>_is_single`` (==1), ``<m>_is_multi`` (>=2).

    Generalises the churn notebook's ``Total_Products`` binning. The churn
    relationship with the product count is non-monotonic — single-product
    customers are the clearest flight risk while multi-product customers are the
    stickiest — so the raw count is a weak feature but the bands isolate the
    cliff at 1→2 holdings. Deterministic and leakage-safe (pure per-row tests of
    an existing feature, never the label).
    """

    name = "financial.holding_count_band"
    description = "Single- vs multi-holding flags from a product/account count"
    leakage_safe = True

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return bool(_holding_count_cols(df))

    def build(self, df, meta=None) -> Dict:  # noqa: ANN001
        out: Dict = {}
        for c in _holding_count_cols(df):
            base = re.sub(r"\W+", "_", str(c)).strip("_").lower()
            single, multi = f"{base}_is_single", f"{base}_is_multi"
            if single not in df.columns:
                out[single] = (df[c].astype(float) == 1).astype(float)
            if multi not in df.columns:
                out[multi] = (df[c].astype(float) >= 2).astype(float)
        return out


# Recipes for the risk/revenue experts (volatility, amount spikes, credit-risk
# ratios) — derived from aggregated metric magnitudes and timestamps.
_RISK_REVENUE_RECIPES: List[FeatureRecipe] = [
    VolatilityRecipe(),
    SpikeRatioRecipe(),
    LogAmountRecipe(),
    TimePartsRecipe(),
    InstallmentBurdenRecipe(),
    AffordabilityRatioRecipe(),
]

# Churn / retention recipes for the customer expert — a port of the
# banking-churn notebook's engineered features (dormancy + holding bands), plus
# the log transform that tames the skewed monetary side of RFM.
_CHURN_RECIPES: List[FeatureRecipe] = [
    LogAmountRecipe(),
    ZeroBalanceFlagRecipe(),
    HoldingCountBandRecipe(),
]

# Which deterministic recipes attach to which expert. recipes_for() returns the
# de-duplicated union across the active experts (a council of e.g. fraud +
# customer gets both sets, LogAmount only once).
_EXPERT_RECIPES: Dict[str, List[FeatureRecipe]] = {
    "fraud": _RISK_REVENUE_RECIPES,
    "revenue": _RISK_REVENUE_RECIPES,
    "payment": _RISK_REVENUE_RECIPES,
    "customer": _CHURN_RECIPES,
}

# Back-compat alias (was the single flat list consumed by recipes_for).
_ALL_RECIPES: List[FeatureRecipe] = _RISK_REVENUE_RECIPES


def recipes_for(expert_keys: List[str]) -> List[FeatureRecipe]:
    """Return the deterministic recipes relevant to the active expert(s).

    The union across experts, de-duplicated by recipe name and preserving first
    occurrence order, so a multi-expert council never materializes a recipe twice.
    """
    seen: Dict[str, FeatureRecipe] = {}
    for key in (expert_keys or []):
        for recipe in _EXPERT_RECIPES.get(key, []):
            seen.setdefault(recipe.name, recipe)
    return list(seen.values())
