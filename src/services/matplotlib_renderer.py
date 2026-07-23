"""
Matplotlib görselleştirme render servisi (Data Visualization graph'ının motoru).

Node/service ayrımı (CLAUDE.md): tüm çizim mantığı burada; viz_render_node yalnızca
ince adaptördür. Bir _VizSpec (chart_type, x, y, group, agg, title, biçimlendirme
alanları) + ham satırlar + kolonlar + tema alır ve döner:

    {"image_b64": <png base64>, "chart": <frontend payload | None>, "error": <str | None>}

- `image_b64`: matplotlib ile sunucuda çizilen statik PNG — indirme / rapora gömme için.
- `chart`: frontend'in recharts ile İSTEMCİDE çizdiği payload. Tema değişimi sunucuya
  gitmeden uygulanabilsin diye artık `box`/`heatmap` dışındaki TÜM tipler için üretilir
  (bu iki tipin recharts karşılığı yok, onlarda PNG'ye düşülür).

Renkler asla burada sabitlenmez: `src/services/viz_themes.resolve_theme()` çıktısı
kullanılır (frontend de aynı kataloğu çeker → tek kaynak).

Headless: modül import edilirken Agg backend seçilir → worker/thread'de GUI olmadan çalışır.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # GUI yok — worker/thread güvenli. import'lardan ÖNCE olmalı.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.services.viz_themes import resolve_theme

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Frontend'in kendi çizdiği tipler + PNG'ye düşenler
_NO_PAYLOAD_TYPES = {"box", "heatmap"}
_KNOWN_TYPES = {
    # rapor (iş) grafikleri
    "kpi", "bar", "hbar", "line", "trend", "pie", "donut",
    "stacked100", "funnel", "waterfall", "grouped_bar",
    # analist (EDA) grafikleri
    "scatter", "hist", "box", "heatmap",
}

_OTHER_LABEL = "Diğer"
_MAX_CATEGORIES = 25          # bu sayının üstünde otomatik top_n uygulanır
MAX_SCATTER_POINTS = 1500     # üstü seyreltilir (okunabilirlik + payload boyutu)


# ─── Veri hazırlığı ───────────────────────────────────────────────────────────

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


def _shape_categories(df: pd.DataFrame, x: str, y: str, spec: dict) -> pd.DataFrame:
    """
    Kategorik grafikleri okunur hale getirir: sıralama + top_n ('Diğer'e toplama).

    Rapor modunda planlayıcı top_n verir; vermediyse kategori sayısı _MAX_CATEGORIES'i
    aşınca yine de kısıtlanır — 60 çubuklu bir grafik kimseye bir şey anlatmaz.
    """
    if not x or x not in df.columns or not y or y not in df.columns:
        return df
    if not pd.api.types.is_numeric_dtype(df[y]):
        return df

    out = df
    sort = (spec.get("sort") or "none").lower()
    if sort in ("desc", "asc"):
        out = out.sort_values(y, ascending=(sort == "asc"))

    top_n = spec.get("top_n")
    try:
        top_n = int(top_n) if top_n else None
    except (TypeError, ValueError):
        top_n = None
    if top_n is None and len(out) > _MAX_CATEGORIES:
        top_n = _MAX_CATEGORIES

    if top_n and 0 < top_n < len(out):
        # 'Diğer'i doğru toplayabilmek için büyükten küçüğe bakılır
        ranked = out.sort_values(y, ascending=False)
        head, tail = ranked.iloc[:top_n], ranked.iloc[top_n:]
        other = pd.DataFrame({x: [_OTHER_LABEL], y: [tail[y].sum()]})
        out = pd.concat([head, other], ignore_index=True)
        if sort == "asc":
            out = out.sort_values(y, ascending=True)

    return out.reset_index(drop=True)


def _resolve_axes(df: pd.DataFrame, columns: list, spec: dict, chart_type: str) -> tuple:
    """spec'teki x/y yoksa ya da şemada bulunmuyorsa makul varsayılanlara düşer."""
    x, y = spec.get("x"), spec.get("y")
    if x is None or x not in df.columns:
        x = columns[0] if columns else None
    if y is None or (y not in df.columns and chart_type not in ("hist", "box")):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x]
        y = numeric_cols[0] if numeric_cols else (columns[1] if len(columns) > 1 else x)
    return x, y


# ─── Sayı biçimlendirme (PNG tarafı; frontend aynı kuralları formatValue.ts'te uygular) ──

# Kompakt kısaltmalar dile bağlıdır: tr'de "B" bindir, en'de milyardır — karıştırılamaz.
_COMPACT_SUFFIXES = {
    "tr": ((1e9, "Mr"), (1e6, "Mn"), (1e3, "B")),
    "en": ((1e9, "B"),  (1e6, "M"),  (1e3, "K")),
}

DEFAULT_LOCALE = "tr"


def _separators(text: str, locale: str) -> str:
    """Python'un '1,234.5' çıktısını yerelin ayraçlarına çevirir (tr → '1.234,5')."""
    if locale != "tr":
        return text
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _format_value(value, fmt: str = "number", currency: str | None = None,
                  locale: str = DEFAULT_LOCALE) -> str:
    """Bir sayıyı okunur metne çevirir: binlik ayraç, %, para birimi, 1,2 Mn kompakt."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    fmt = (fmt or "number").lower()
    locale = locale if locale in _COMPACT_SUFFIXES else DEFAULT_LOCALE

    if fmt == "percent":
        text = _separators(f"{v:,.1f}", locale)
        return f"%{text}" if locale == "tr" else f"{text}%"

    if fmt == "compact":
        text = None
        for limit, suffix in _COMPACT_SUFFIXES[locale]:
            if abs(v) >= limit:
                text = f"{_separators(f'{v / limit:,.1f}', locale)} {suffix}"
                break
        if text is None:
            text = _separators(f"{v:,.0f}", locale)
    else:
        raw = f"{v:,.2f}" if abs(v) < 100 and v != int(v) else f"{v:,.0f}"
        text = _separators(raw, locale)

    if fmt == "currency" and currency:
        return f"{text} {currency}"
    return text


def _formatter(spec: dict):
    """
    spec'e bağlı tek argümanlı biçimlendirici (bar_label / tick formatter için).

    Yerel, render_matplotlib girişinde spec'e `_locale` olarak yazılır — böylece her
    çizim fonksiyonuna ayrı bir locale parametresi taşımak gerekmez.
    """
    return lambda v: _format_value(
        v, spec.get("value_format") or "number", spec.get("currency"),
        spec.get("_locale") or DEFAULT_LOCALE,
    )


def _format_value_axis(ax, spec: dict, axis: str = "y") -> None:
    """Değer eksenindeki ham sayıları (120000) okunur hale getirir (120.000 ₺)."""
    fmt = _formatter(spec)
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(lambda v, _pos: fmt(v)))


# ─── Figür / eksen stili ──────────────────────────────────────────────────────

def _new_fig(theme: dict, figsize=(8, 4.8)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    return fig, ax


def _clean_axes(ax, theme: dict, grid_axis: str = "y") -> None:
    """Chartjunk'ı azaltır: üst/sağ çerçeve kapalı, tek yönde ince ızgara, soluk eksenler."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])
    if grid_axis in ("x", "y"):
        ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def _set_labels(ax, spec: dict, x: str, y: str, swap: bool = False) -> None:
    """Teknik kolon adı yerine planlayıcının verdiği insan-okur etiketi kullanır."""
    x_label = spec.get("x_label") or x or ""
    y_label = spec.get("y_label") or y or ""
    if swap:
        x_label, y_label = y_label, x_label
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)


def _rotate_if_crowded(ax, labels: list) -> None:
    if len(labels) > 6:
        ax.tick_params(axis="x", rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")


def _data_labels(ax, containers, spec: dict, theme: dict) -> None:
    """İstenmişse her çubuğun üstüne/sağına değerini yazar."""
    if not spec.get("data_labels"):
        return
    fmt = _formatter(spec)
    try:
        ax.bar_label(containers, labels=[fmt(v) for v in containers.datavalues],
                     padding=3, fontsize=8, color=theme["text"])
    except Exception as e:  # pragma: no cover - matplotlib sürüm farkları
        logger.warning(f"[viz-render] data label yazılamadı: {e}")


def _finish(fig, ax, spec: dict, theme: dict) -> str:
    """
    Başlık + alt başlığı yerleştirip figürü base64 PNG'ye çevirir.

    `title` AÇIKÇA boş verilirse ("") başlık çizilmez — PDF raporu grafiği başlıksız
    ister, çünkü başlığı sayfanın kendi tipografisi taşır (viz_report._chart_image).
    """
    raw_title = spec.get("title")
    title = "Visualization" if raw_title is None else str(raw_title).strip()
    subtitle = spec.get("subtitle")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=theme["text"], y=0.99)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", fontsize=9.5, color=theme["muted"])
    top = 1.0 if not title and not subtitle else (0.94 if subtitle else 0.96)
    fig.tight_layout(rect=(0, 0, 1, top))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ─── Frontend payload ─────────────────────────────────────────────────────────

def _num(v) -> float:
    try:
        f = float(v)
        return 0.0 if pd.isna(f) else f
    except (TypeError, ValueError):
        return 0.0


def _base_payload(chart_type: str, spec: dict, x: str, y: str) -> dict:
    """Her tipin paylaştığı sunum meta'sı (frontend bunu doğrudan tüketir)."""
    return {
        "type":         chart_type,
        "title":        spec.get("title") or "Visualization",
        "subtitle":     spec.get("subtitle"),
        "x":            x,
        "y":            y,
        "x_label":      spec.get("x_label") or x,
        "y_label":      spec.get("y_label") or y,
        "value_format": spec.get("value_format") or "number",
        "currency":     spec.get("currency"),
        "data_labels":  bool(spec.get("data_labels")),
        "target":       spec.get("target"),
        "labels":       [],
        "values":       [],
    }


# ─── Tip bazlı çizimler ───────────────────────────────────────────────────────

def _render_bar(ax, plot_df, x, y, spec, theme, horizontal: bool = False):
    labels = plot_df[x].astype(str).tolist()
    values = [_num(v) for v in plot_df[y].tolist()]
    color = theme["colors"][0]
    if horizontal:
        # Yatay barda en büyük en üstte olsun (okuma yönü)
        bars = ax.barh(labels[::-1], values[::-1], color=color, height=0.65, zorder=3)
        _clean_axes(ax, theme, grid_axis="x")
        _set_labels(ax, spec, x, y, swap=True)
        _format_value_axis(ax, spec, axis="x")
    else:
        bars = ax.bar(labels, values, color=color, width=0.65, zorder=3)
        _clean_axes(ax, theme, grid_axis="y")
        _set_labels(ax, spec, x, y)
        _format_value_axis(ax, spec)
        _rotate_if_crowded(ax, labels)
    _data_labels(ax, bars, spec, theme)
    return labels, values


def _render_line(ax, plot_df, x, y, spec, theme, is_trend: bool = False):
    labels = plot_df[x].astype(str).tolist()
    values = [_num(v) for v in plot_df[y].tolist()]
    color = theme["colors"][0]
    ax.plot(labels, values, marker="o", markersize=4, linewidth=2, color=color, zorder=3)
    if is_trend and values:
        # Son noktayı vurgula — "şu an nerede duruyoruz" sorusunun cevabı
        ax.scatter([labels[-1]], [values[-1]], s=70, color=color, zorder=4,
                   edgecolors=theme["surface"], linewidths=2)
        ax.annotate(_formatter(spec)(values[-1]), (labels[-1], values[-1]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, fontweight="bold", color=theme["text"])
    target = spec.get("target")
    if target is not None:
        ax.axhline(_num(target), color=theme["muted"], linestyle="--", linewidth=1.2, zorder=2)
        ax.annotate(f"Hedef: {_formatter(spec)(target)}", (0, _num(target)),
                    textcoords="offset points", xytext=(4, 4), fontsize=8, color=theme["muted"])
    _clean_axes(ax, theme, grid_axis="y")
    _set_labels(ax, spec, x, y)
    _format_value_axis(ax, spec)
    _rotate_if_crowded(ax, labels)
    return labels, values


def _render_pie(ax, plot_df, x, y, spec, theme, donut: bool = False):
    labels = plot_df[x].astype(str).tolist()
    values = [abs(_num(v)) for v in plot_df[y].tolist()]
    colors = [theme["colors"][i % len(theme["colors"])] for i in range(len(labels))]
    wedge_kw = {"width": 0.42, "edgecolor": theme["surface"], "linewidth": 2} if donut \
        else {"edgecolor": theme["surface"], "linewidth": 2}
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90,
           wedgeprops=wedge_kw, textprops={"fontsize": 9, "color": theme["text"]})
    ax.axis("equal")
    if donut:
        total = sum(values)
        ax.text(0, 0, _formatter(spec)(total), ha="center", va="center",
                fontsize=15, fontweight="bold", color=theme["text"])
    return labels, values


def _render_kpi(ax, plot_df, x, y, spec, theme):
    """Tek büyük sayı — teknik olmayan okuyucu için en okunur 'grafik'."""
    num_col = y if (y in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[y])) else None
    if num_col is None:
        numeric = [c for c in plot_df.columns if pd.api.types.is_numeric_dtype(plot_df[c])]
        num_col = numeric[0] if numeric else plot_df.columns[0]
    value = _num(plot_df[num_col].iloc[0]) if len(plot_df) else 0.0

    fmt = _formatter(spec)
    ax.axis("off")
    ax.text(0.5, 0.60, fmt(value), ha="center", va="center", fontsize=44,
            fontweight="bold", color=theme["text"], transform=ax.transAxes)
    # Başlık zaten aynı şeyi söylüyorsa alt yazıyı tekrar etme
    caption = spec.get("y_label") or spec.get("x_label") or num_col
    if str(caption).strip().lower() != str(spec.get("title") or "").strip().lower():
        ax.text(0.5, 0.34, str(caption), ha="center", va="center", fontsize=11,
                color=theme["muted"], transform=ax.transAxes)

    delta = None
    compare_to = spec.get("compare_to")
    if compare_to and compare_to in plot_df.columns and len(plot_df):
        prev = _num(plot_df[compare_to].iloc[0])
        if prev:
            delta = (value - prev) / abs(prev) * 100.0
    elif spec.get("target") is not None:
        target = _num(spec["target"])
        if target:
            delta = (value - target) / abs(target) * 100.0

    if delta is not None:
        color = theme["positive"] if delta >= 0 else theme["negative"]
        arrow = "▲" if delta >= 0 else "▼"
        pct = _format_value(abs(delta), "percent", None, spec.get("_locale") or DEFAULT_LOCALE)
        ax.text(0.5, 0.19, f"{arrow} {pct}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color, transform=ax.transAxes)
    return value, delta


def _ink_on(fill: str, theme: dict) -> str:
    """Dolgunun üstünde okunur kalan metin rengi (açık dolguda koyu, koyu dolguda beyaz)."""
    try:
        r, g, b = (int(fill.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except (ValueError, TypeError):
        return theme["text"]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if luminance > 0.55 else "#ffffff"


def _render_funnel(ax, plot_df, x, y, spec, theme):
    """Aşama aşama daralan huni — dönüşüm anlatısı."""
    ranked = plot_df.sort_values(y, ascending=False)
    labels = ranked[x].astype(str).tolist()
    values = [abs(_num(v)) for v in ranked[y].tolist()]
    top = values[0] if values else 1.0
    fmt = _formatter(spec)

    # Aşama adları solda eksen etiketi olarak durur; çubuğun içinde yalnızca değer yazar
    # (uzun aşama adları dar dilimlerden taşmasın diye).
    for i, (label, value) in enumerate(zip(labels, values)):
        width = (value / top) if top else 0
        color = theme["colors"][i % len(theme["colors"])]
        ax.barh(i, width, left=(1 - width) / 2, height=0.62, color=color, zorder=3)
        share = f" · %{value / top * 100:,.0f}" if top else ""
        text = f"{fmt(value)}{share}"
        if width >= 0.3:
            ax.text(0.5, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=_ink_on(color, theme), zorder=4)
        else:
            ax.text((1 + width) / 2 + 0.01, i, text, ha="left", va="center", fontsize=9,
                    fontweight="bold", color=theme["text"], zorder=4)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=theme["muted"], labelsize=9.5, length=0)
    return labels, values


def _render_waterfall(ax, plot_df, x, y, spec, theme):
    """Artı/eksi katkıların toplam nasıl oluştuğunu gösterir."""
    labels = plot_df[x].astype(str).tolist()
    values = [_num(v) for v in plot_df[y].tolist()]
    fmt = _formatter(spec)

    cumulative = 0.0
    for i, value in enumerate(values):
        color = theme["positive"] if value >= 0 else theme["negative"]
        ax.bar(i, value, bottom=cumulative, color=color, width=0.6, zorder=3)
        ax.text(i, cumulative + value + (max(map(abs, values)) * 0.02 if values else 0),
                fmt(value), ha="center", fontsize=8, color=theme["text"], zorder=4)
        cumulative += value

    ax.bar(len(values), cumulative, color=theme["colors"][0], width=0.6, zorder=3)
    ax.text(len(values), cumulative, fmt(cumulative), ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=theme["text"], zorder=4)

    ticks = labels + ["Toplam"]
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks)
    _clean_axes(ax, theme, grid_axis="y")
    _set_labels(ax, spec, x, y)
    _format_value_axis(ax, spec)
    _rotate_if_crowded(ax, ticks)
    return labels, values


def _pivot_series(df: pd.DataFrame, x: str, y: str, group: str, agg: str):
    """x × group kırılımını (index, kolon adları, matris) olarak döner."""
    func = agg if agg and agg != "none" else "sum"
    pivot = df.pivot_table(index=x, columns=group, values=y, aggfunc=func).fillna(0)
    return pivot


def _render_grouped_bar(ax, df, x, y, group, spec, theme):
    """x kategori, group ikinci kategori, y sayısal → yan yana barlar."""
    if not group or group not in df.columns:
        agg_df = _aggregate(df, x, y, (spec.get("agg") or "sum"))
        agg_df = _shape_categories(agg_df, x, y, spec)
        labels, values = _render_bar(ax, agg_df, x, y, spec, theme)
        return labels, [{"name": spec.get("y_label") or y, "values": values}]

    pivot = _pivot_series(df, x, y, group, spec.get("agg"))
    xs = np.arange(len(pivot.index))
    n = max(len(pivot.columns), 1)
    width = 0.8 / n
    series = []
    for i, col in enumerate(pivot.columns):
        values = [_num(v) for v in pivot[col].values]
        ax.bar(xs + i * width, values, width * 0.9, label=str(col),
               color=theme["colors"][i % len(theme["colors"])], zorder=3)
        series.append({"name": str(col), "values": values})

    ax.set_xticks(xs + width * (n - 1) / 2)
    ax.set_xticklabels([str(idx) for idx in pivot.index])
    _clean_axes(ax, theme, grid_axis="y")
    _set_labels(ax, spec, x, y)
    _format_value_axis(ax, spec)
    _rotate_if_crowded(ax, list(pivot.index))
    legend = ax.legend(title=spec.get("group_label") or group, fontsize=8, frameon=False)
    legend.get_title().set_fontsize(8)
    return [str(i) for i in pivot.index], series


def _render_stacked100(ax, df, x, y, group, spec, theme):
    """Her kategoriyi %100'e normalize eder → pay karşılaştırması."""
    if not group or group not in df.columns:
        raise ValueError("stacked100 requires a `group` column")

    pivot = _pivot_series(df, x, y, group, spec.get("agg"))
    totals = pivot.sum(axis=1).replace(0, np.nan)
    shares = pivot.div(totals, axis=0).fillna(0) * 100.0

    labels = [str(i) for i in shares.index]
    bottom = np.zeros(len(shares.index))
    series = []
    for i, col in enumerate(shares.columns):
        values = [_num(v) for v in shares[col].values]
        ax.bar(labels, values, bottom=bottom, width=0.6, label=str(col),
               color=theme["colors"][i % len(theme["colors"])],
               edgecolor=theme["surface"], linewidth=1.5, zorder=3)
        bottom += np.array(values)
        series.append({"name": str(col), "values": values})

    ax.set_ylim(0, 100)
    _clean_axes(ax, theme, grid_axis="y")
    ax.set_xlabel(spec.get("x_label") or x, fontsize=10)
    ax.set_ylabel("Pay (%)", fontsize=10)
    _rotate_if_crowded(ax, labels)
    legend = ax.legend(title=group, fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    legend.get_title().set_fontsize(8)
    return labels, series


def _render_scatter(ax, df, x, y, spec, theme):
    # Binlerce nokta ne PNG'de ne tarayıcıda okunur bir şey katmaz; sadece kaydı ve
    # payload'ı şişirir. Deterministik olarak (sabit adımla) seyreltilir.
    plot_df = df
    if len(df) > MAX_SCATTER_POINTS:
        step = len(df) // MAX_SCATTER_POINTS + 1
        plot_df = df.iloc[::step]
        logger.info(f"[viz-render] scatter {len(df)} → {len(plot_df)} noktaya seyreltildi")

    ax.scatter(plot_df[x], plot_df[y], color=theme["colors"][0], alpha=0.5,
               edgecolors="none", s=18, zorder=3)
    _clean_axes(ax, theme, grid_axis="y")
    _set_labels(ax, spec, x, y)
    _format_value_axis(ax, spec)
    points = [[_num(a), _num(b)] for a, b in zip(plot_df[x].tolist(), plot_df[y].tolist())]
    return points, len(df)


def _render_hist(ax, df, x, spec, theme):
    series = pd.to_numeric(df[x], errors="coerce").dropna()
    bins = min(30, max(5, series.nunique()))
    counts, edges, _ = ax.hist(series, bins=bins, color=theme["colors"][0],
                               edgecolor=theme["surface"], zorder=3)
    _clean_axes(ax, theme, grid_axis="y")
    ax.set_xlabel(spec.get("x_label") or x, fontsize=10)
    ax.set_ylabel(spec.get("y_label") or "Frekans", fontsize=10)
    labels = [f"{edges[i]:,.1f}–{edges[i + 1]:,.1f}" for i in range(len(counts))]
    return labels, [float(c) for c in counts]


def _render_box(ax, df: pd.DataFrame, x: str, y: str, spec: dict, theme: dict) -> None:
    """y sayısal; x kategorikse kategoriye göre kutu, değilse tek kutu."""
    num_col = y if (y in df.columns and pd.api.types.is_numeric_dtype(df[y])) else x
    cat_col = x if x != num_col else None
    box_kw = {
        "patch_artist": True,
        "boxprops": {"facecolor": theme["colors"][0], "alpha": 0.65, "edgecolor": theme["axis"]},
        "medianprops": {"color": theme["text"], "linewidth": 1.5},
        "whiskerprops": {"color": theme["axis"]},
        "capprops": {"color": theme["axis"]},
        "flierprops": {"markeredgecolor": theme["muted"], "markersize": 4},
    }
    if cat_col and cat_col in df.columns and df[cat_col].nunique() <= 15:
        groups, data = [], []
        for key, grp in df.groupby(cat_col, dropna=False):
            vals = pd.to_numeric(grp[num_col], errors="coerce").dropna()
            if len(vals):
                groups.append(str(key)); data.append(vals)
        ax.boxplot(data, tick_labels=groups, **box_kw)
        _set_labels(ax, spec, cat_col, num_col)
        _rotate_if_crowded(ax, groups)
    else:
        vals = pd.to_numeric(df[num_col], errors="coerce").dropna()
        ax.boxplot(vals, tick_labels=[num_col], **box_kw)
        ax.set_ylabel(spec.get("y_label") or num_col, fontsize=10)
    _clean_axes(ax, theme, grid_axis="y")


def _render_heatmap(fig, ax, df: pd.DataFrame, x: str, y: str, theme: dict) -> None:
    """x, y kategorik; üçüncü sayısal kolon yoğunluk. Yoksa korelasyon matrisi."""
    value_cols = [c for c in df.columns if c not in (x, y) and pd.api.types.is_numeric_dtype(df[c])]
    if x in df.columns and y in df.columns and value_cols:
        pivot = df.pivot_table(index=y, columns=x, values=value_cols[0], aggfunc="mean")
    else:
        pivot = df.select_dtypes("number").corr()
    # Sıralı büyüklük → tek hue, açıktan koyuya (gökkuşağı değil)
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=theme["muted"], labelsize=8)


# ─── Giriş noktası ────────────────────────────────────────────────────────────

def render_matplotlib(spec: dict, rows: list, columns: list, theme=None,
                      locale: str = DEFAULT_LOCALE) -> dict:
    """
    _VizSpec + ham satırlar + tema → matplotlib PNG (+ frontend render payload'ı).

    `locale` yalnızca sayı biçimlendirmesini etkiler (binlik ayraç, kompakt kısaltma);
    spec'e `_locale` olarak yazılır, bkz. `_formatter`.

    Hata durumunda {"image_b64": None, "chart": None, "error": <msg>} döner (asla raise etmez).
    """
    spec = {**spec, "_locale": locale}
    chart_type = (spec.get("chart_type") or "bar").lower()
    if chart_type not in _KNOWN_TYPES:
        chart_type = "bar"
    group = spec.get("group")
    agg   = (spec.get("agg") or "none").lower()
    resolved_theme = resolve_theme(theme)

    if not rows or not columns:
        return {"image_b64": None, "chart": None, "error": "No data to render"}

    try:
        df = _to_dataframe(rows, columns)
    except Exception as e:
        return {"image_b64": None, "chart": None, "error": f"DataFrame build failed: {e}"}

    x, y = _resolve_axes(df, columns, spec, chart_type)
    payload: dict | None = None

    try:
        # Tek serili kategorik tipler: önce aggregate, sonra sırala/kısalt
        if chart_type in ("bar", "hbar", "line", "trend", "pie", "donut", "funnel", "waterfall", "kpi"):
            plot_df = _aggregate(df, x, y, agg)
            if chart_type not in ("kpi", "waterfall", "line", "trend"):
                plot_df = _shape_categories(plot_df, x, y, spec)
        else:
            plot_df = df

        figsize = (7, 3.6) if chart_type == "kpi" else (8, 4.8)
        fig, ax = _new_fig(resolved_theme, figsize)

        if chart_type in ("bar", "hbar"):
            labels, values = _render_bar(ax, plot_df, x, y, spec, resolved_theme,
                                         horizontal=(chart_type == "hbar"))
            payload = {**_base_payload(chart_type, spec, x, y), "labels": [str(l) for l in labels], "values": values}

        elif chart_type in ("line", "trend"):
            labels, values = _render_line(ax, plot_df, x, y, spec, resolved_theme,
                                          is_trend=(chart_type == "trend"))
            payload = {**_base_payload(chart_type, spec, x, y), "labels": [str(l) for l in labels], "values": values}

        elif chart_type in ("pie", "donut"):
            labels, values = _render_pie(ax, plot_df, x, y, spec, resolved_theme,
                                         donut=(chart_type == "donut"))
            payload = {**_base_payload(chart_type, spec, x, y), "labels": [str(l) for l in labels], "values": values}

        elif chart_type == "kpi":
            value, delta = _render_kpi(ax, plot_df, x, y, spec, resolved_theme)
            payload = {**_base_payload(chart_type, spec, x, y),
                       "labels": [spec.get("y_label") or y or ""], "values": [value],
                       "kpi": {"value": value, "delta_pct": delta,
                               "caption": spec.get("y_label") or spec.get("x_label") or y}}

        elif chart_type == "funnel":
            labels, values = _render_funnel(ax, plot_df, x, y, spec, resolved_theme)
            payload = {**_base_payload(chart_type, spec, x, y), "labels": [str(l) for l in labels], "values": values}

        elif chart_type == "waterfall":
            labels, values = _render_waterfall(ax, plot_df, x, y, spec, resolved_theme)
            payload = {**_base_payload(chart_type, spec, x, y), "labels": [str(l) for l in labels], "values": values,
                       "total": float(sum(values))}

        elif chart_type == "grouped_bar":
            labels, series = _render_grouped_bar(ax, df, x, y, group, spec, resolved_theme)
            payload = {**_base_payload(chart_type, spec, x, y), "labels": labels, "series": series,
                       "group": group}

        elif chart_type == "stacked100":
            labels, series = _render_stacked100(ax, df, x, y, group, spec, resolved_theme)
            payload = {**_base_payload(chart_type, spec, x, y), "labels": labels, "series": series,
                       "group": group, "value_format": "percent"}

        elif chart_type == "scatter":
            points, total = _render_scatter(ax, df, x, y, spec, resolved_theme)
            payload = {**_base_payload(chart_type, spec, x, y), "points": points,
                       "sampled_from": total if total > len(points) else None}

        elif chart_type == "hist":
            labels, values = _render_hist(ax, df, x, spec, resolved_theme)
            payload = {**_base_payload(chart_type, spec, x, y), "labels": labels, "values": values,
                       "y_label": spec.get("y_label") or "Frekans"}

        elif chart_type == "box":
            _render_box(ax, df, x, y, spec, resolved_theme)

        elif chart_type == "heatmap":
            _render_heatmap(fig, ax, df, x, y, resolved_theme)

        image_b64 = _finish(fig, ax, spec, resolved_theme)
        if chart_type in _NO_PAYLOAD_TYPES:
            payload = None
        return {"image_b64": image_b64, "chart": payload, "error": None}

    except Exception as e:
        logger.error(f"[viz-render] {chart_type} render başarısız: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return {"image_b64": None, "chart": None, "error": f"Render failed: {e}"}
