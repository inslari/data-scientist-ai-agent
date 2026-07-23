"""
Galeri grafiklerinden tek parça PDF rapor üretir.

Neden matplotlib: PNG'leri zaten matplotlib üretiyor ve `PdfPages` standart kurulumda
mevcut — reportlab/weasyprint gibi yeni bir bağımlılık (ve Docker imajında sistem
kütüphanesi) gerektirmez. Sayfa düzeni saf figür yerleşimidir.

Çıktı: kapak sayfası + her grafik için bir sayfa (başlık, görsel, AI yorumu, ops. SQL).
"""
from __future__ import annotations

import base64
import io
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.services.viz_themes import resolve_theme

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# A4 dikey (inç)
_PAGE = (8.27, 11.69)

_STRINGS = {
    "tr": {
        "generated": "Oluşturulma",
        "charts":    "grafik",
        "sql":       "Sorgu",
        "page":      "Sayfa",
        "untitled":  "Rapor",
    },
    "en": {
        "generated": "Generated",
        "charts":    "charts",
        "sql":       "Query",
        "page":      "Page",
        "untitled":  "Report",
    },
}


def _t(locale: str) -> dict:
    return _STRINGS.get(locale, _STRINGS["tr"])


def _wrap(text: str, width: int, max_lines: int) -> str:
    """Metni sayfaya sığacak şekilde sarar ve gerekirse kırpar."""
    lines: list[str] = []
    for paragraph in (text or "").splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 1)] + "…"
    return "\n".join(lines)


def _new_page(pdf_theme: dict):
    fig = plt.figure(figsize=_PAGE)
    fig.patch.set_facecolor(pdf_theme["surface"])
    return fig


def _cover_page(pdf: PdfPages, theme: dict, title: str, subtitle: str | None,
                chart_count: int, locale: str) -> None:
    s = _t(locale)
    fig = _new_page(theme)

    # Marka şeridi: temanın ilk renkleri
    for i, color in enumerate(theme["colors"][:6]):
        fig.add_artist(plt.Rectangle((0.08 + i * 0.06, 0.74), 0.055, 0.012,
                                     color=color, transform=fig.transFigure))

    fig.text(0.08, 0.66, title, fontsize=28, fontweight="bold",
             color=theme["text"], va="top", wrap=True)
    if subtitle:
        fig.text(0.08, 0.58, _wrap(subtitle, 60, 3), fontsize=13,
                 color=theme["muted"], va="top")

    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    fig.text(0.08, 0.12, f"{s['generated']}: {stamp}", fontsize=10, color=theme["muted"])
    fig.text(0.08, 0.09, f"{chart_count} {s['charts']}", fontsize=10, color=theme["muted"])

    pdf.savefig(fig, facecolor=theme["surface"])
    plt.close(fig)


def _chart_image(item: dict, theme) -> tuple[str | None, bool]:
    """
    Sayfaya konacak görseli döner: (base64, sayfa_başlığı_çizilsin_mi).

    Ham veri saklanmışsa grafik BAŞLIKSIZ yeniden çizilir — başlığı sayfanın kendi
    tipografisi taşır, böylece aynı başlık iki kere görünmez. Veri yoksa (eski kayıt)
    saklanan PNG kullanılır ve başlık zaten görselin içinde olduğu için tekrarlanmaz.
    """
    from src.services.matplotlib_renderer import render_matplotlib
    from src.services.viz_data import from_storable

    columns, rows = from_storable(item.get("data"))
    if columns and rows and item.get("spec"):
        spec = {**item["spec"], "title": "", "subtitle": None}
        result = render_matplotlib(spec, rows, columns, theme=theme,
                                   locale=item.get("locale") or "tr")
        if result.get("image_b64"):
            return result["image_b64"], True
    return item.get("image_b64"), False


def _chart_page(pdf: PdfPages, theme: dict, item: dict, index: int, total: int,
                include_summary: bool, include_sql: bool, locale: str) -> None:
    s = _t(locale)
    fig = _new_page(theme)
    top = 0.94

    image_b64, draw_title = _chart_image(item, theme)

    if draw_title:
        title_lines = _wrap((item.get("title") or "").strip(), 58, 2)
        fig.text(0.08, top, title_lines, fontsize=16, fontweight="bold",
                 color=theme["text"], va="top")
        top -= 0.035 * (title_lines.count("\n") + 1) + 0.02

    # Grafik görseli — en-boy oranı korunarak sayfaya oturtulur
    bottom = top - 0.40
    if image_b64:
        try:
            img = plt.imread(io.BytesIO(base64.b64decode(image_b64)), format="png")
            h, w = img.shape[0], img.shape[1]
            box_w = 0.84
            box_h = min(box_w * (h / w) * (_PAGE[0] / _PAGE[1]), 0.42)
            bottom = top - box_h
            ax = fig.add_axes((0.08, bottom, box_w, box_h))
            ax.imshow(img)
            ax.axis("off")
        except Exception as e:  # pragma: no cover - bozuk base64
            logger.warning(f"[viz-report] görsel gömülemedi: {e}")

    # Metinler görselin hemen altından akar (sabit ofset değil — boşluk oluşmasın)
    line_h = 0.017
    cursor = bottom - 0.045
    if include_summary and item.get("summary"):
        wrapped = _wrap(item["summary"], 92, 8)
        fig.text(0.08, cursor, wrapped, fontsize=10.5, color=theme["text"],
                 va="top", linespacing=1.5)
        cursor -= line_h * 1.5 * (wrapped.count("\n") + 1) + 0.035

    if include_sql and item.get("sql"):
        fig.text(0.08, cursor, s["sql"], fontsize=9, fontweight="bold",
                 color=theme["muted"], va="top")
        fig.text(0.08, cursor - 0.022, _wrap(item["sql"], 100, 10), fontsize=7.5,
                 color=theme["muted"], va="top", family="monospace", linespacing=1.4)

    fig.text(0.92, 0.04, f"{s['page']} {index}/{total}", fontsize=8,
             color=theme["muted"], ha="right")

    pdf.savefig(fig, facecolor=theme["surface"])
    plt.close(fig)


def build_pdf_report(
    items: list[dict],
    *,
    title: str | None = None,
    subtitle: str | None = None,
    theme=None,
    include_summaries: bool = True,
    include_sql: bool = False,
    locale: str = "tr",
) -> bytes:
    """
    Grafik kayıtlarından PDF üretir.

    `items`: sırayla {"title", "image_b64", "summary", "sql", "spec", "data"} sözlükleri —
    çağıran (route) sıralamayı kullanıcının seçtiği düzende verir. `spec` + `data` varsa
    grafik rapor için başlıksız yeniden çizilir (bkz. _chart_image).
    """
    locale = locale if locale in _STRINGS else "tr"
    resolved = resolve_theme(theme)
    report_title = (title or "").strip() or _t(locale)["untitled"]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _cover_page(pdf, resolved, report_title, subtitle, len(items), locale)
        for i, item in enumerate(items, start=1):
            _chart_page(pdf, resolved, item, i, len(items),
                        include_summaries, include_sql, locale)

    buf.seek(0)
    return buf.read()
