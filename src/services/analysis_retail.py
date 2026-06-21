"""Retail-specific deep-analysis modules (registered under the ``retail`` key in
``analysis_engine.ANALYSIS_MODULES``).

Every function keeps the shared registry signature::

    compute_x(df, profile, target_col) -> dict | None

and stays **dataset-agnostic**: column roles (datetime / quantity / price) are
detected heuristically from names + dtypes, and the function returns ``None``
(skip) whenever the data can't support the analysis. No column name is ever
hardcoded.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ── Column-role detection (name + dtype heuristics) ─────────────────────────
_QTY_PAT = r"(quantity|qty|units|sold|sales|demand|volume|adet|miktar|sat[ıi][şs])"
_PRICE_PAT = r"(price|unit.?price|cost|fiyat|birim)"
_DATE_PAT = r"(date|time|tarih|timestamp|datetime)"


def _numeric(df: pd.DataFrame, profile: dict) -> set:
    cols = set(profile.get("summary", {}).get("numeric_columns", []) or [])
    cols |= {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return {c for c in cols if c in df.columns}


def _match(df: pd.DataFrame, pattern: str) -> list[str]:
    pat = re.compile(pattern, re.IGNORECASE)
    return [c for c in df.columns if pat.search(str(c))]


def _detect_datetime(df: pd.DataFrame, profile: dict) -> Optional[str]:
    for c in profile.get("summary", {}).get("datetime_columns", []) or []:
        if c in df.columns:
            return c
    for c in _match(df, _DATE_PAT):
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.7:
                return c
        except Exception:
            continue
    return None


def _first_numeric(df: pd.DataFrame, numeric: set, pattern: str) -> Optional[str]:
    for c in _match(df, pattern):
        if c in numeric:
            return c
    return None


# ── Demand trend (time series) ──────────────────────────────────────────────
def compute_demand_trend(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Aggregate demand over time: total, trend direction and peak weekday.

    Needs a datetime column and a numeric quantity/sales column. Returns
    ``{"demand_trend": {...}}`` or ``None``.
    """
    numeric = _numeric(df, profile)
    dt_col = _detect_datetime(df, profile)
    qty_col = _first_numeric(df, numeric, _QTY_PAT)
    if not dt_col or not qty_col or dt_col == qty_col:
        return None

    work = df[[dt_col, qty_col]].copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
    work = work.dropna()
    if len(work) < 30 or work[dt_col].dt.normalize().nunique() < 10:
        return None

    daily = work.groupby(work[dt_col].dt.normalize())[qty_col].sum().sort_index()
    x = np.arange(len(daily))
    slope = float(np.polyfit(x, daily.values, 1)[0]) if len(daily) > 1 else 0.0
    span = float(daily.mean()) or 1.0
    rel = slope / span
    trend = "up" if rel > 0.01 else "down" if rel < -0.01 else "flat"

    weekday = work.groupby(work[dt_col].dt.day_name())[qty_col].sum()
    peak_weekday = str(weekday.idxmax()) if len(weekday) else None

    return {
        "demand_trend": {
            "date_col": dt_col,
            "quantity_col": qty_col,
            "n_days": int(len(daily)),
            "total": round(float(daily.sum()), 2),
            "mean_daily": round(float(daily.mean()), 2),
            "trend": trend,
            "slope_per_day": round(slope, 4),
            "peak_weekday": peak_weekday,
        }
    }


# ── Price elasticity (log-log demand vs. price) ─────────────────────────────
def compute_price_elasticity(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Estimate price elasticity of demand from a log-log slope.

    Needs a numeric price column and a numeric quantity column with positive
    values and enough price variation. Returns ``{"price_elasticity": {...}}`` or ``None``.
    """
    numeric = _numeric(df, profile)
    price_col = _first_numeric(df, numeric, _PRICE_PAT)
    qty_col = _first_numeric(df, numeric, _QTY_PAT)
    if not price_col or not qty_col or price_col == qty_col:
        return None

    work = df[[price_col, qty_col]].copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
    work = work.dropna()
    work = work[(work[price_col] > 0) & (work[qty_col] > 0)]
    if len(work) < 30 or work[price_col].nunique() < 6:
        return None

    logp = np.log(work[price_col].values)
    logq = np.log(work[qty_col].values)
    if np.std(logp) == 0:
        return None
    elasticity = float(np.polyfit(logp, logq, 1)[0])

    if elasticity <= -1:
        interp = "elastic"
    elif elasticity < 0:
        interp = "inelastic"
    else:
        interp = "atypical (non-negative)"

    return {
        "price_elasticity": {
            "price_col": price_col,
            "quantity_col": qty_col,
            "n": int(len(work)),
            "elasticity": round(elasticity, 3),
            "interpretation": interp,
            "mean_price": round(float(work[price_col].mean()), 2),
        }
    }
