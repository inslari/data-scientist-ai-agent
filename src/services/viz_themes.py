"""
Görselleştirme tema kataloğu (renk paletlerinin TEK kaynağı).

matplotlib_renderer (sunucu PNG'si) ve frontend (recharts) aynı temayı kullanır:
frontend kataloğu `GET /viz-themes` ile buradan çeker, ikinci bir tanım yoktur.

Her tema:
    {
      "id", "label", "kind",          # kind: categorical | ordinal
      "colors":   [hex, ...],         # seri renkleri (sıra ANLAMLIDIR, bkz. aşağıda)
      "surface", "grid", "text",      # grafik kromu
      "muted", "axis",
      "positive", "negative", "warning",   # KPI delta / waterfall işaret renkleri
      "dark": bool,                   # koyu yüzey teması mı
    }

Renk sırası kozmetik değil, ERİŞİLEBİLİRLİK mekanizmasıdır: paletler renk körlüğü
(protan/deutan/tritan) ayrımı, açıklık bandı, kroma tabanı ve yüzey kontrastı
kontrollerinden komşu-çift sırasıyla geçirilerek seçildi. Yeni tema eklerken ya da
mevcut bir temanın sırasını değiştirirken paleti yeniden doğrulayın; rastgele hex
eklemeyin.
"""
from __future__ import annotations

import re

from src.core.logging_config import get_logger
logger = get_logger(__name__)

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# Açık yüzey kroması (tüm açık temalarda ortak)
_LIGHT_CHROME = {
    "surface":  "#fcfcfb",
    "grid":     "#e1e0d9",
    "text":     "#0b0b0b",
    "muted":    "#898781",
    "axis":     "#c3c2b7",
    "positive": "#0ca30c",
    "negative": "#d03b3b",
    "warning":  "#fab219",
    "dark":     False,
}

_DARK_CHROME = {
    "surface":  "#1a1a19",
    "grid":     "#2c2c2a",
    "text":     "#ffffff",
    "muted":    "#898781",
    "axis":     "#383835",
    "positive": "#0ca30c",
    "negative": "#d03b3b",
    "warning":  "#fab219",
    "dark":     True,
}


def _light(id_: str, label: str, colors: list[str], kind: str = "categorical") -> dict:
    return {"id": id_, "label": label, "kind": kind, "colors": colors, **_LIGHT_CHROME}


VIZ_THEMES: dict[str, dict] = {
    # Varsayılan: 8 slot, komşu çiftlerin tamamı CVD ve normal görü tabanlarını geçer.
    "corporate": _light("corporate", "Kurumsal", [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ]),
    # Sıcak ağırlıklı; açıklık farkı CVD ayrımını taşıyor.
    "sunset": _light("sunset", "Gün Batımı", [
        "#eb6834", "#6b3fa0", "#eda100", "#b52a2a", "#e87ba4", "#8a5b00",
    ]),
    # Yumuşak/az doygun ama hâlâ ayırt edilebilir (gerçek pastel tabanları geçemez).
    "soft": _light("soft", "Yumuşak", [
        "#5b8fd0", "#d97f4a", "#3fa88a", "#7f78c9", "#d9a83c", "#c97ba0",
    ]),
    # Tek hue, açıktan koyuya — sıralı/tek serili grafikler (funnel, ordinal bar) için.
    "mono_blue": _light("mono_blue", "Tek Renk (Mavi)", [
        "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b",
    ], kind="ordinal"),
    # Hepsi açık yüzeyde >= 3:1 kontrast — baskı ve erişilebilirlik için.
    "high_contrast": _light("high_contrast", "Yüksek Kontrast", [
        "#1c5cab", "#b3431a", "#00795a", "#8a5b00",
        "#b53a6b", "#006300", "#5949c4", "#b52a2a",
    ]),
    # Koyu yüzey için ayrıca basamaklanmış aynı sekiz hue (otomatik ters çevirme DEĞİL).
    "dark": {
        "id": "dark", "label": "Koyu", "kind": "categorical",
        "colors": [
            "#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767",
        ],
        **_DARK_CHROME,
    },
}

DEFAULT_THEME_ID = "corporate"


def _valid_colors(colors) -> list[str] | None:
    """Hex listesi doğrular; en az 2 geçerli renk yoksa None."""
    if not isinstance(colors, (list, tuple)):
        return None
    out = [c.strip() for c in colors if isinstance(c, str) and _HEX_RE.match(c.strip())]
    return out[:12] if len(out) >= 2 else None


def resolve_theme(theme=None) -> dict:
    """
    Serbest tema girdisini tam bir tema sözlüğüne çevirir. Asla raise etmez.

    Kabul edilen girdiler:
      None                                  → varsayılan tema
      "corporate" / {"id": "corporate"}      → katalogdan
      {"id": "custom", "colors": [hex, ...]} → kullanıcının kendi paleti
                                               (eksik krom alanları varsayılandan)
      {"id": "custom", "colors": [...], "base": "dark"} → koyu kroma üzerine özel palet
    """
    default = VIZ_THEMES[DEFAULT_THEME_ID]

    if theme is None:
        return dict(default)

    if isinstance(theme, str):
        return dict(VIZ_THEMES.get(theme.strip(), default))

    if not isinstance(theme, dict):
        logger.warning(f"[viz-theme] Beklenmeyen tema tipi: {type(theme).__name__}; varsayılana düşülüyor")
        return dict(default)

    theme_id = str(theme.get("id") or "").strip()
    if theme_id and theme_id in VIZ_THEMES:
        return dict(VIZ_THEMES[theme_id])

    colors = _valid_colors(theme.get("colors"))
    if not colors:
        return dict(default)

    base = VIZ_THEMES.get(str(theme.get("base") or ""), default)
    resolved = dict(base)
    resolved.update({
        "id":     "custom",
        "label":  str(theme.get("label") or "Özel")[:40],
        "kind":   "categorical",
        "colors": colors,
    })
    # Kullanıcı krom alanlarını da ezebilir (arka plan/ızgara/metin)
    for key in ("surface", "grid", "text", "muted", "axis"):
        val = theme.get(key)
        if isinstance(val, str) and _HEX_RE.match(val.strip()):
            resolved[key] = val.strip()
    return resolved


def theme_catalog() -> list[dict]:
    """API'nin döndüğü sıralı tema listesi."""
    return list(VIZ_THEMES.values())
