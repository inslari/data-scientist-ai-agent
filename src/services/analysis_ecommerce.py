"""E-commerce specific deep-analysis modules (registered under the ``ecommerce``
key in ``analysis_engine.ANALYSIS_MODULES``).

Every function keeps the shared registry signature::

    compute_x(df, profile, target_col) -> dict | None

and stays **dataset-agnostic**: column roles (customer / order / product /
monetary / datetime) are detected heuristically from names + dtypes, and the
function returns ``None`` (skip) whenever the data can't support the analysis.
No column name is ever hardcoded.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ── Column-role detection (name + dtype heuristics) ─────────────────────────
_ROLE_PATTERNS: dict[str, str] = {
    "customer": r"(customer|client|user|account|buyer|cust|m[uü]ş?teri|musteri|kullan[ıi]c[ıi])",
    "order":    r"(order|invoice|transaction|receipt|basket|cart|sipari[şs]|fatura)",
    "category": r"(category|categ|department|segment|kategori)",
    "product":  r"(product|item|sku|article|stock|[uü]r[uü]n)",
    "monetary": r"(amount|price|total|revenue|spend|sales|monetary|tutar|fiyat|net|value|gmv|ödeme|odeme)",
    "datetime": r"(date|time|tarih|timestamp|datetime)",
}


def _match_cols(df: pd.DataFrame, role: str) -> list[str]:
    pat = re.compile(_ROLE_PATTERNS[role], re.IGNORECASE)
    return [c for c in df.columns if pat.search(str(c))]


def _detect_datetime(df: pd.DataFrame, profile: dict) -> Optional[str]:
    summary = profile.get("summary", {})
    for c in summary.get("datetime_columns", []) or []:
        if c in df.columns:
            return c
    for c in _match_cols(df, "datetime"):
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.7:
                return c
        except Exception:
            continue
    return None


def _detect_id(df: pd.DataFrame, role: str) -> Optional[str]:
    """Pick the best identifier-like column for a role (highest cardinality, but
    not unique-per-row for 'order'/'customer' grouping to be meaningful)."""
    candidates = _match_cols(df, role)
    best, best_card = None, -1
    n = len(df)
    for c in candidates:
        nun = df[c].nunique(dropna=True)
        if nun < 2 or nun >= n:           # constant or unique-per-row → not a group key
            continue
        if nun > best_card:
            best, best_card = c, nun
    return best


def _detect_monetary(df: pd.DataFrame, profile: dict) -> Optional[str]:
    numeric = set(profile.get("summary", {}).get("numeric_columns", []) or [])
    for c in _match_cols(df, "monetary"):
        if c in numeric or pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _detect_item(df: pd.DataFrame) -> Optional[str]:
    """Prefer a category column (fewer, interpretable groups) over raw product."""
    for role in ("category", "product"):
        candidates = _match_cols(df, role)
        # favor moderate cardinality so co-occurrence is meaningful
        ranked = sorted(
            (c for c in candidates if 1 < df[c].nunique(dropna=True) <= 2000),
            key=lambda c: df[c].nunique(dropna=True),
        )
        if ranked:
            return ranked[0]
    return None


# ── RFM segmentation ────────────────────────────────────────────────────────
def _rfm_segment(r: int, f: int, m: int) -> str:
    """Map R/F (1-5) scores to a coarse, well-known RFM segment label."""
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2 and f >= 4:
        return "At Risk (high value)"
    if r <= 2 and f <= 2:
        return "Hibernating / Lost"
    return "Needs Attention"


def compute_rfm(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Per-customer Recency/Frequency/Monetary scores + segment distribution.

    Needs a customer id, a datetime column and a monetary column. Returns
    ``{"rfm": {...}}`` with per-segment sizes and average R/F/M, or ``None``.
    """
    cust = _detect_id(df, "customer")
    dt_col = _detect_datetime(df, profile)
    money = _detect_monetary(df, profile)
    if not cust or not dt_col or not money:
        return None

    work = df[[cust, dt_col, money]].copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work[money] = pd.to_numeric(work[money], errors="coerce")
    work = work.dropna(subset=[cust, dt_col, money])
    if work[cust].nunique() < 10:
        return None

    snapshot = work[dt_col].max() + pd.Timedelta(days=1)
    agg = work.groupby(cust).agg(
        recency=(dt_col, lambda s: (snapshot - s.max()).days),
        frequency=(dt_col, "count"),
        monetary=(money, "sum"),
    ).reset_index()

    def _score(series: pd.Series, reverse: bool = False) -> pd.Series:
        # quintile rank 1-5; recency is reversed (smaller recency = better = 5)
        try:
            q = pd.qcut(series.rank(method="first"), 5, labels=False) + 1
        except Exception:
            q = pd.Series(np.full(len(series), 3), index=series.index)
        return (6 - q) if reverse else q

    agg["R"] = _score(agg["recency"], reverse=True).astype(int)
    agg["F"] = _score(agg["frequency"]).astype(int)
    agg["M"] = _score(agg["monetary"]).astype(int)
    agg["segment"] = [
        _rfm_segment(r, f, m) for r, f, m in zip(agg["R"], agg["F"], agg["M"])
    ]

    seg_rows = []
    for seg, g in agg.groupby("segment"):
        seg_rows.append({
            "segment": seg,
            "count": int(len(g)),
            "pct": round(100.0 * len(g) / len(agg), 2),
            "avg_recency_days": round(float(g["recency"].mean()), 1),
            "avg_frequency": round(float(g["frequency"].mean()), 2),
            "avg_monetary": round(float(g["monetary"].mean()), 2),
        })
    seg_rows.sort(key=lambda x: x["avg_monetary"], reverse=True)

    return {
        "rfm": {
            "customer_col": cust,
            "date_col": dt_col,
            "monetary_col": money,
            "n_customers": int(len(agg)),
            "segments": seg_rows,
            "overall": {
                "avg_recency_days": round(float(agg["recency"].mean()), 1),
                "avg_frequency": round(float(agg["frequency"].mean()), 2),
                "avg_monetary": round(float(agg["monetary"].mean()), 2),
            },
        }
    }


# ── Market-basket affinity (association rules) ──────────────────────────────
def compute_basket_affinity(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """Top co-occurring product/category pairs within an order (support/confidence/lift).

    Needs an order id and a product/category column. Returns
    ``{"basket_affinity": {...}}`` with the strongest pairs, or ``None``.
    """
    order = _detect_id(df, "order")
    item = _detect_item(df)
    if not order or not item or order == item:
        return None

    work = df[[order, item]].dropna()
    # keep multi-line orders only; co-occurrence is undefined for single-item baskets
    n_orders = work[order].nunique()
    if n_orders < 20:
        return None

    # Restrict to the most common items to keep the pair scan cheap & interpretable
    top_items = work[item].value_counts().head(40).index
    work = work[work[item].isin(top_items)]

    # order → set of items
    baskets = work.groupby(order)[item].apply(lambda s: set(s))
    baskets = baskets[baskets.map(len) >= 2]
    total = len(baskets)
    if total < 20:
        return None

    from collections import Counter
    item_count: Counter = Counter()
    pair_count: Counter = Counter()
    for items in baskets:
        items = sorted(items)
        for it in items:
            item_count[it] += 1
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair_count[(items[i], items[j])] += 1

    pairs = []
    for (a, b), c in pair_count.items():
        support = c / total
        conf_a_b = c / item_count[a] if item_count[a] else 0.0
        sup_b = item_count[b] / total
        lift = (conf_a_b / sup_b) if sup_b else 0.0
        pairs.append({
            "a": str(a), "b": str(b), "count": int(c),
            "support": round(support, 4),
            "confidence": round(conf_a_b, 4),
            "lift": round(lift, 3),
        })
    if not pairs:
        return None
    pairs.sort(key=lambda x: (x["lift"], x["support"]), reverse=True)

    return {
        "basket_affinity": {
            "order_col": order,
            "item_col": item,
            "n_orders": int(total),
            "n_items_considered": int(len(top_items)),
            "pairs": pairs[:20],
        }
    }
