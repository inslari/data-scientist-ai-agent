"""Marketing-specific deep-analysis modules (registered under the ``marketing``
key in ``analysis_engine.ANALYSIS_MODULES``).

Every function keeps the shared registry signature::

    compute_x(df, profile, target_col) -> dict | None

and stays **dataset-agnostic**: column roles (channel / cost / revenue) are
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
_CHANNEL_PAT = r"(channel|source|medium|platform|campaign|kanal|kaynak|mecra|kampanya)"
_COST_PAT = r"(cost|spend|budget|cpc|cpm|invest|maliyet|b[uü]t[çc]e|harcama|yat[ıi]r[ıi]m)"
_REVENUE_PAT = r"(revenue|sales|income|value|conversion|gelir|ciro|d[oö]n[uü][şs]|gmv)"


def _numeric(df: pd.DataFrame, profile: dict) -> set:
    cols = set(profile.get("summary", {}).get("numeric_columns", []) or [])
    cols |= {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return {c for c in cols if c in df.columns}


def _match(df: pd.DataFrame, pattern: str) -> list[str]:
    pat = re.compile(pattern, re.IGNORECASE)
    return [c for c in df.columns if pat.search(str(c))]


def _first_numeric(df: pd.DataFrame, numeric: set, pattern: str, exclude=()) -> Optional[str]:
    for c in _match(df, pattern):
        if c in numeric and c not in exclude:
            return c
    return None


def _pick_channel(df: pd.DataFrame, profile: dict) -> Optional[str]:
    cats = profile.get("summary", {}).get("categorical_columns", []) or []
    named = [c for c in _match(df, _CHANNEL_PAT)
             if c in df.columns and df[c].dtype == object]
    cand = named or [c for c in cats if c in df.columns]
    for c in cand:
        if 2 <= df[c].nunique(dropna=True) <= 50:
            return c
    return None


# ── Channel performance ─────────────────────────────────────────────────────
def compute_channel_performance(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Per-channel volume and average metric (revenue/conversion if present).

    Needs a channel/source categorical column. Returns
    ``{"channel_performance": {...}}`` or ``None``.
    """
    channel = _pick_channel(df, profile)
    if not channel:
        return None
    numeric = _numeric(df, profile)
    metric = _first_numeric(df, numeric, _REVENUE_PAT)
    if metric is None:
        # fall back to any numeric column for a volume-weighted view
        metric = next(iter(sorted(numeric)), None)

    rows = []
    if metric:
        work = df[[channel, metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[channel])
        for g, sub in work.groupby(channel):
            if len(sub) < 3:
                continue
            rows.append({
                "channel": str(g),
                "n": int(len(sub)),
                "total_metric": round(float(sub[metric].sum()), 2),
                "mean_metric": round(float(sub[metric].mean()), 2),
            })
        rows.sort(key=lambda r: r["total_metric"], reverse=True)
    else:
        for g, sub in df.groupby(channel):
            rows.append({"channel": str(g), "n": int(len(sub))})
        rows.sort(key=lambda r: r["n"], reverse=True)

    if len(rows) < 2:
        return None

    return {
        "channel_performance": {
            "channel_col": channel,
            "metric_col": metric,
            "n_channels": int(len(rows)),
            "channels": rows[:20],
        }
    }


# ── Campaign ROI ────────────────────────────────────────────────────────────
def compute_campaign_roi(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Overall ROI = (revenue - cost) / cost, plus per-channel ROI when available.

    Needs a numeric cost/spend column and a numeric revenue column. Returns
    ``{"campaign_roi": {...}}`` or ``None``.
    """
    numeric = _numeric(df, profile)
    cost = _first_numeric(df, numeric, _COST_PAT)
    revenue = _first_numeric(df, numeric, _REVENUE_PAT, exclude=(cost,) if cost else ())
    if not cost or not revenue or cost == revenue:
        return None

    work = df[[cost, revenue]].copy()
    work[cost] = pd.to_numeric(work[cost], errors="coerce")
    work[revenue] = pd.to_numeric(work[revenue], errors="coerce")
    work = work.dropna()
    work = work[work[cost] > 0]
    if len(work) < 20:
        return None

    total_cost = float(work[cost].sum())
    total_rev = float(df[revenue].pipe(pd.to_numeric, errors="coerce").dropna().sum())
    overall_roi = round((total_rev - total_cost) / total_cost, 4) if total_cost else None

    by_channel = None
    channel = _pick_channel(df, profile)
    if channel:
        cw = df[[channel, cost, revenue]].copy()
        cw[cost] = pd.to_numeric(cw[cost], errors="coerce")
        cw[revenue] = pd.to_numeric(cw[revenue], errors="coerce")
        cw = cw.dropna(subset=[channel])
        rows = []
        for g, sub in cw.groupby(channel):
            c = float(sub[cost].sum())
            r = float(sub[revenue].sum())
            if c <= 0:
                continue
            rows.append({
                "channel": str(g),
                "n": int(len(sub)),
                "cost": round(c, 2),
                "revenue": round(r, 2),
                "roi": round((r - c) / c, 4),
            })
        rows.sort(key=lambda x: x["roi"], reverse=True)
        if rows:
            by_channel = {"column": channel, "channels": rows[:15]}

    return {
        "campaign_roi": {
            "cost_col": cost,
            "revenue_col": revenue,
            "n": int(len(work)),
            "total_cost": round(total_cost, 2),
            "total_revenue": round(total_rev, 2),
            "overall_roi": overall_roi,
            "by_channel": by_channel,
        }
    }
