"""HR / people-analytics deep-analysis modules (registered under the ``hr`` key in
``analysis_engine.ANALYSIS_MODULES``).

Every function keeps the shared registry signature::

    compute_x(df, profile, target_col) -> dict | None

and stays **dataset-agnostic**: column roles (attrition flag / salary / group)
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
_ATTR_PAT = r"(attrition|left|quit|turnover|terminat|resign|churn|exit|ayr[ıi]l|isten.?ayr|left_company)"
_SALARY_PAT = r"(salary|compensation|wage|pay|income|earning|maa[şs]|[uü]cret|gelir)"
_GROUP_PAT = r"(department|dept|division|team|role|title|job|grade|level|gender|b[oö]l[uü]m|departman|cinsiyet|unvan|kademe)"


def _numeric(df: pd.DataFrame, profile: dict) -> set:
    cols = set(profile.get("summary", {}).get("numeric_columns", []) or [])
    cols |= {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return {c for c in cols if c in df.columns}


def _match(df: pd.DataFrame, pattern: str) -> list[str]:
    pat = re.compile(pattern, re.IGNORECASE)
    return [c for c in df.columns if pat.search(str(c))]


def _detect_binary01(df: pd.DataFrame, pattern: str) -> Optional[str]:
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
    named = [c for c in cands if re.search(_GROUP_PAT, str(c), re.IGNORECASE)]
    for c in named + cands:
        if 2 <= df[c].nunique(dropna=True) <= 30:
            return c
    return None


def _first_numeric(df: pd.DataFrame, numeric: set, pattern: str) -> Optional[str]:
    for c in _match(df, pattern):
        if c in numeric:
            return c
    return None


# ── Attrition overview ──────────────────────────────────────────────────────
def compute_attrition_overview(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Overall attrition rate plus the rate by an organizational group.

    Needs a binary attrition flag. Returns ``{"attrition_overview": {...}}`` or ``None``.
    """
    col = _detect_binary01(df, _ATTR_PAT)
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
        "attrition_overview": {
            "flag_col": col,
            "n": int(len(y)),
            "overall_rate": round(overall, 4),
            "by_group": by_group,
        }
    }


# ── Compensation gaps ───────────────────────────────────────────────────────
def compute_compensation_gaps(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Pay distribution by group plus the spread between highest- and lowest-paid groups.

    Needs a numeric salary/compensation column and a categorical group. Returns
    ``{"compensation_gaps": {...}}`` or ``None``.
    """
    numeric = _numeric(df, profile)
    salary = _first_numeric(df, numeric, _SALARY_PAT)
    if not salary:
        return None
    group = _pick_group(df, profile, exclude=(salary,))
    if not group:
        return None

    work = df[[group, salary]].copy()
    work[salary] = pd.to_numeric(work[salary], errors="coerce")
    work = work.dropna()
    if len(work) < 30 or work[salary].nunique() < 5:
        return None

    rows = []
    for g, sub in work.groupby(group):
        if len(sub) < 5:
            continue
        rows.append({
            "group": str(g),
            "n": int(len(sub)),
            "mean_pay": round(float(sub[salary].mean()), 2),
            "median_pay": round(float(sub[salary].median()), 2),
        })
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r["mean_pay"], reverse=True)

    overall_mean = float(work[salary].mean())
    hi, lo = rows[0]["mean_pay"], rows[-1]["mean_pay"]
    gap_pct = round(100.0 * (hi - lo) / overall_mean, 2) if overall_mean else None

    return {
        "compensation_gaps": {
            "salary_col": salary,
            "group_col": group,
            "n": int(len(work)),
            "overall_mean_pay": round(overall_mean, 2),
            "gap_pct_of_mean": gap_pct,
            "highest_group": rows[0]["group"],
            "lowest_group": rows[-1]["group"],
            "groups": rows[:15],
        }
    }
