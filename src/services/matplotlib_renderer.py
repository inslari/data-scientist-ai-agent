"""
Matplotlib görselleştirme render servisi (Data Visualization graph'ının motoru).

Node/service ayrımı (CLAUDE.md): tüm çizim mantığı burada; viz_render_node yalnızca
ince adaptördür. Bir _VizSpec (chart_type, x, y, group, agg, title) + ham satırlar +
kolonlar alır ve döner:

    {"image_b64": <png base64>, "chart": <recharts payload | None>, "error": <str | None>}

- `image_b64`: matplotlib ile sunucuda çizilen statik PNG (tam özelleştirme).
- `chart`: basit tipler (bar/line/pie) için frontend'in interaktif recharts moduna
  besleyebileceği hafif payload; diğer tiplerde None (hibrit render).

Headless: modül import edilirken Agg backend seçilir → worker/thread'de GUI olmadan çalışır.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # GUI yok — worker/thread güvenli. import'lardan ÖNCE olmalı.
import matplotlib.pyplot as plt
import pandas as pd

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Sabit, deterministik renk paleti (RANDOM_SEED gerektirmez)
_PALETTE = [
    "#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa",
    "#fb7185", "#4ade80", "#facc15", "#60a5fa", "#f97316",
]

_SIMPLE_TYPES = {"bar", "line", "pie"}
_KNOWN_TYPES = {"bar", "line", "pie", "scatter", "hist", "box", "heatmap", "grouped_bar"}


def _to_dataframe(rows: list, columns: list) -> pd.DataFrame:
    """Ham tuple satırlarını DataFrame'e çevirir; sayısala çevrilebilen kolonları çevirir."""
    df = pd.DataFrame([list(r) for r in rows], columns=columns)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Kolonun çoğunluğu sayıya çevrilebiliyorsa sayısal kabul et
        if converted.notna().mean() >= 0.8:
            df[col] = converted
    return df


def _aggregate(df: pd.DataFrame, x: str, y: str, agg: str) -> pd.DataFrame:
    """x'e göre grupla, y'yi aggregate et. agg='none' → dokunma."""
    if not agg or agg == "none" or not x or not y or x not in df.columns:
        return df
    try:
        if agg == "count":
            out = df.groupby(x, dropna=False).size().reset_index(name=y)
        elif y in df.columns:
            out = getattr(df.groupby(x, dropna=False)[y], agg)().reset_index()
        else:
            return df
        return out
    except Exception as e:
        logger.warning(f"[viz-render] aggregation başarısız ({agg}): {e}")
        return df


def _new_fig():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    return fig, ax


def _finish(fig, title: str) -> str:
    """Figürü başlıkla kapatıp base64 PNG döner."""
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _build_recharts_payload(chart_type: str, title: str, labels: list, values: list, x: str, y: str) -> dict:
    """Basit tipler için frontend interaktif (recharts) moduna hafif payload."""
    return {
        "type":   chart_type,
        "title":  title,
        "x":      x,
        "y":      y,
        "labels": [str(l) for l in labels],
        "values": [float(v) if v is not None and not pd.isna(v) else 0.0 for v in values],
    }


def render_matplotlib(spec: dict, rows: list, columns: list) -> dict:
    """
    _VizSpec + ham satırlar → matplotlib PNG (+ basit tipler için recharts payload).

    Hata durumunda {"image_b64": None, "chart": None, "error": <msg>} döner (asla raise etmez).
    """
    chart_type = (spec.get("chart_type") or "bar").lower()
    if chart_type not in _KNOWN_TYPES:
        chart_type = "bar"
    x     = spec.get("x")
    y     = spec.get("y")
    group = spec.get("group")
    agg   = (spec.get("agg") or "none").lower()
    title = spec.get("title") or "Visualization"

    if not rows or not columns:
        return {"image_b64": None, "chart": None, "error": "No data to render"}

    try:
        df = _to_dataframe(rows, columns)
    except Exception as e:
        return {"image_b64": None, "chart": None, "error": f"DataFrame build failed: {e}"}

    # x/y verilmediyse makul varsayılanlara düş (ilk kategori + ilk sayısal)
    if x is None or x not in df.columns:
        x = columns[0] if columns else None
    if y is None or (y not in df.columns and chart_type not in ("hist", "box")):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x]
        y = numeric_cols[0] if numeric_cols else (columns[1] if len(columns) > 1 else x)

    recharts: dict | None = None

    try:
        if chart_type in ("bar", "line", "pie", "grouped_bar"):
            plot_df = _aggregate(df, x, y, agg)
        else:
            plot_df = df

        fig, ax = _new_fig()

        if chart_type == "bar":
            labels = plot_df[x].astype(str).tolist()
            values = plot_df[y].tolist()
            ax.bar(labels, values, color=_PALETTE[0])
            ax.set_xlabel(x); ax.set_ylabel(y)
            _rotate_if_crowded(ax, labels)
            recharts = _build_recharts_payload("bar", title, labels, values, x, y)

        elif chart_type == "line":
            labels = plot_df[x].astype(str).tolist()
            values = plot_df[y].tolist()
            ax.plot(labels, values, marker="o", color=_PALETTE[0])
            ax.set_xlabel(x); ax.set_ylabel(y)
            _rotate_if_crowded(ax, labels)
            recharts = _build_recharts_payload("line", title, labels, values, x, y)

        elif chart_type == "pie":
            labels = plot_df[x].astype(str).tolist()
            values = plot_df[y].tolist()
            ax.pie(values, labels=labels, autopct="%1.1f%%", colors=_PALETTE, startangle=90)
            ax.axis("equal")
            recharts = _build_recharts_payload("pie", title, labels, values, x, y)

        elif chart_type == "scatter":
            ax.scatter(df[x], df[y], color=_PALETTE[0], alpha=0.6, edgecolors="none")
            ax.set_xlabel(x); ax.set_ylabel(y)

        elif chart_type == "hist":
            series = pd.to_numeric(df[x], errors="coerce").dropna()
            ax.hist(series, bins=min(30, max(5, series.nunique())), color=_PALETTE[0], edgecolor="white")
            ax.set_xlabel(x); ax.set_ylabel("frequency")

        elif chart_type == "box":
            _render_box(ax, df, x, y)

        elif chart_type == "heatmap":
            _render_heatmap(fig, ax, df, x, y, spec)

        elif chart_type == "grouped_bar":
            _render_grouped_bar(ax, df, x, y, group, agg)

        image_b64 = _finish(fig, title)
        return {"image_b64": image_b64, "chart": recharts, "error": None}

    except Exception as e:
        logger.error(f"[viz-render] {chart_type} render başarısız: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return {"image_b64": None, "chart": None, "error": f"Render failed: {e}"}


def _rotate_if_crowded(ax, labels: list) -> None:
    if len(labels) > 6:
        ax.tick_params(axis="x", rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")


def _render_box(ax, df: pd.DataFrame, x: str, y: str) -> None:
    """y sayısal; x kategorikse kategoriye göre kutu, değilse tek kutu."""
    num_col = y if (y in df.columns and pd.api.types.is_numeric_dtype(df[y])) else x
    cat_col = x if x != num_col else None
    if cat_col and cat_col in df.columns and df[cat_col].nunique() <= 15:
        groups, data = [], []
        for key, grp in df.groupby(cat_col, dropna=False):
            vals = pd.to_numeric(grp[num_col], errors="coerce").dropna()
            if len(vals):
                groups.append(str(key)); data.append(vals)
        ax.boxplot(data, tick_labels=groups)
        ax.set_xlabel(cat_col); ax.set_ylabel(num_col)
        _rotate_if_crowded(ax, groups)
    else:
        vals = pd.to_numeric(df[num_col], errors="coerce").dropna()
        ax.boxplot(vals, tick_labels=[num_col])
        ax.set_ylabel(num_col)


def _render_heatmap(fig, ax, df: pd.DataFrame, x: str, y: str, spec: dict) -> None:
    """x, y kategorik; üçüncü sayısal kolon yoğunluk. Yoksa korelasyon matrisi."""
    value_cols = [c for c in df.columns if c not in (x, y) and pd.api.types.is_numeric_dtype(df[c])]
    if x in df.columns and y in df.columns and value_cols:
        pivot = df.pivot_table(index=y, columns=x, values=value_cols[0], aggfunc="mean")
    else:
        pivot = df.select_dtypes("number").corr()
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(i) for i in pivot.index])
    fig.colorbar(im, ax=ax)


def _render_grouped_bar(ax, df: pd.DataFrame, x: str, y: str, group: str, agg: str) -> None:
    """x kategori, group ikinci kategori, y sayısal → yan yana barlar."""
    if not group or group not in df.columns:
        # group yoksa normal bar'a düş
        agg_df = _aggregate(df, x, y, agg if agg != "none" else "sum")
        ax.bar(agg_df[x].astype(str), agg_df[y], color=_PALETTE[0])
        ax.set_xlabel(x); ax.set_ylabel(y)
        return
    func = agg if agg and agg != "none" else "sum"
    pivot = df.pivot_table(index=x, columns=group, values=y, aggfunc=func)
    import numpy as np
    xs = np.arange(len(pivot.index))
    n = len(pivot.columns)
    width = 0.8 / max(n, 1)
    for i, col in enumerate(pivot.columns):
        ax.bar(xs + i * width, pivot[col].values, width, label=str(col), color=_PALETTE[i % len(_PALETTE)])
    ax.set_xticks(xs + width * (n - 1) / 2)
    ax.set_xticklabels([str(idx) for idx in pivot.index], rotation=45, ha="right")
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.legend(title=group, fontsize=8)
