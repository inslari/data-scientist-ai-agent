"""Healthcare-specific deep-analysis modules (registered under the ``healthcare``
key in ``analysis_engine.ANALYSIS_MODULES``).

Every function keeps the shared registry signature::

    compute_x(df, profile, target_col) -> dict | None

and stays **dataset-agnostic**: column roles (time / event / readmission / group)
are detected heuristically from names + dtypes, and the function returns ``None``
(skip) whenever the data can't support the analysis. No column name is ever
hardcoded.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ── Column-role detection (name + dtype heuristics) ─────────────────────────
_TIME_PAT = r"(time|duration|tenure|days|length.?of.?stay|\blos\b|survival|follow.?up|stay|s[uü]re|g[uü]n)"
_EVENT_PAT = r"(event|status|death|died|mortality|deceased|expired|relapse|[oö]l[uü]m|vefat)"
_READMIT_PAT = r"(readmit|readmission|recurren|relapse|return|tekrar.?yat|yeniden.?yat)"
_GROUP_PAT = r"(diagnosis|icd|condition|disease|department|ward|service|cohort|group|tan[ıi]|hastal[ıi]k|b[oö]l[uü]m)"


def _numeric(df: pd.DataFrame, profile: dict) -> set:
    cols = set(profile.get("summary", {}).get("numeric_columns", []) or [])
    cols |= {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return {c for c in cols if c in df.columns}


def _match(df: pd.DataFrame, pattern: str) -> list[str]:
    pat = re.compile(pattern, re.IGNORECASE)
    return [c for c in df.columns if pat.search(str(c))]


def _detect_binary01(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """A column whose name matches `pattern` and whose values are exactly {0, 1}."""
    for c in _match(df, pattern):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        u = set(np.unique(s.values).tolist()) if len(s) else set()
        if u and u.issubset({0, 1}) and len(u) == 2:
            return c
    return None


def _pick_group(df: pd.DataFrame, profile: dict, exclude=()) -> Optional[str]:
    cats = profile.get("summary", {}).get("categorical_columns", []) or []
    cands = [c for c in cats if c in df.columns and c not in exclude]
    if not cands:
        cands = [c for c in df.columns if c not in exclude and df[c].dtype == object]
    # prefer a clinically meaningful grouping column by name, then any low-card one
    named = [c for c in cands if re.search(_GROUP_PAT, str(c), re.IGNORECASE)]
    for c in named + cands:
        if 2 <= df[c].nunique(dropna=True) <= 30:
            return c
    return None


# ── Survival summary (time-to-event) ────────────────────────────────────────
def compute_survival_summary(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Event rate, median time-to-event and a rough survival curve.

    Needs a non-negative duration column and a binary event/censoring flag.
    Returns ``{"survival_summary": {...}}`` or ``None``.
    """
    numeric = _numeric(df, profile)
    time_col = None
    for c in _match(df, _TIME_PAT):
        if c in numeric:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) and (s >= 0).mean() > 0.9 and s.nunique() > 5:
                time_col = c
                break
    event_col = _detect_binary01(df, _EVENT_PAT)
    if not time_col or not event_col or time_col == event_col:
        return None

    work = df[[time_col, event_col]].copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")
    work = work.dropna()
    work = work[work[time_col] >= 0]
    if len(work) < 30:
        return None

    event = work[event_col].astype(int)
    times = work[time_col]
    event_rate = float(event.mean())

    # Rough survival proxy: fraction still under observation at each time quantile.
    qs = [0.25, 0.5, 0.75]
    curve = []
    for q in qs:
        t = float(times.quantile(q))
        surviving = float((times >= t).mean())
        curve.append({"time": round(t, 2), "at_or_beyond": round(surviving, 4)})

    return {
        "survival_summary": {
            "time_col": time_col,
            "event_col": event_col,
            "n": int(len(work)),
            "event_rate": round(event_rate, 4),
            "median_time": round(float(times.median()), 2),
            "median_time_event": round(float(times[event == 1].median()), 2) if (event == 1).any() else None,
            "median_time_censored": round(float(times[event == 0].median()), 2) if (event == 0).any() else None,
            "survival_curve": curve,
        }
    }


# ── Readmission / outcome rate ──────────────────────────────────────────────
def compute_readmission_rate(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Overall readmission rate plus the rate broken down by a clinical group.

    Needs a binary readmission flag. Returns ``{"readmission_rate": {...}}`` or ``None``.
    """
    col = _detect_binary01(df, _READMIT_PAT)
    if not col:
        return None

    y = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(y) < 30:
        return None
    overall = float(y.mean())

    by_group = None
    group = _pick_group(df, profile, exclude=(col,))
    if group:
        work = df[[group, col]].copy()
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna()
        rows = []
        for g, sub in work.groupby(group):
            if len(sub) < 5:
                continue
            rows.append({
                "group": str(g),
                "n": int(len(sub)),
                "rate": round(float(sub[col].mean()), 4),
            })
        rows.sort(key=lambda r: r["rate"], reverse=True)
        if rows:
            by_group = {"column": group, "groups": rows[:15]}

    return {
        "readmission_rate": {
            "flag_col": col,
            "n": int(len(y)),
            "overall_rate": round(overall, 4),
            "by_group": by_group,
        }
    }
