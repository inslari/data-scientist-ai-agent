from __future__ import annotations

"""
Feature Naming (human-readable)
-------------------------------
Makine üretimi feature isimlerini (feature_builder konvansiyonları) teknik
olmayan okuyucu için sade dile çevirir. DETERMINISTIK ve PREFIX/PATTERN-TABANLI:
hiçbir dataset-spesifik isim hard-code edilmez — prefix tanınır, kalan kolon
token'ı otomatik ayrıştırılır. Böylece yarın farklı isimli bir veri gelse
(``ratio_income_age``) kural yine çalışır.

Tek istisna: motorun ürettiği sabit RFM isimleri (spend_recent/old/trend) —
bunlar veriye değil reçeteye bağlıdır.

Kaynak konvansiyonlar (src/services/feature_builder.py):
    lag_{k}_{col}, roll{w}_mean_{col}, roll{w}_std_{col},
    count_/sum_/avg_/max_/std_/fav_{col}, {col}_isna, {col}_diversity,
    dev_{num}_by_{cat}, zdev_{num}_by_{cat}, grpstd_{num}_by_{cat},
    is_zero_{col}, ratio_{a}_{b}, was_missing_{col}
"""

import re

SUPPORTED = ("tr", "en")

# ── Sade karşılık şablonları (TR/EN). {..} alanları .format ile doldurulur. ──
_T = {
    "lag":        {"tr": "{col} — {k} dönem önceki değeri", "en": "{col}, {k} period(s) earlier"},
    "roll_mean":  {"tr": "{col} — son {w} dönem ortalaması", "en": "{col} — {w}-period moving average"},
    "roll_std":   {"tr": "{col} — son {w} dönem oynaklığı", "en": "{col} — {w}-period volatility"},
    "count":      {"tr": "{col} sıklığı", "en": "frequency of {col}"},
    "sum":        {"tr": "toplam {col}", "en": "total {col}"},
    "avg":        {"tr": "ortalama {col}", "en": "average {col}"},
    "max":        {"tr": "en yüksek {col}", "en": "max {col}"},
    "std":        {"tr": "{col} değişkenliği", "en": "{col} variability"},
    "fav":        {"tr": "en sık {col}", "en": "most frequent {col}"},
    "is_zero":    {"tr": "{col} sıfır mı", "en": "whether {col} is zero"},
    "isna":       {"tr": "{col} eksik mi", "en": "whether {col} is missing"},
    "was_missing":{"tr": "{col} eksikti mi", "en": "whether {col} was missing"},
    "diversity":  {"tr": "{col} çeşitliliği", "en": "{col} diversity"},
    "dev":        {"tr": "{num} — {cat} grubu ortalamasından sapma", "en": "{num} vs. its {cat}-group average"},
    "zdev":       {"tr": "{num} — {cat} grubuna göre standart sapma puanı", "en": "{num} standardized vs. its {cat} group"},
    "grpstd":     {"tr": "{cat} grubunda {num} değişkenliği", "en": "{num} variability within {cat} group"},
    "ratio":      {"tr": "{a} / {b} oranı", "en": "{a} to {b} ratio"},
    "spend_recent": {"tr": "son dönem harcaması", "en": "recent-period spend"},
    "spend_old":    {"tr": "önceki dönem harcaması", "en": "prior-period spend"},
    "spend_trend":  {"tr": "harcama eğilimi (son/önceki)", "en": "spend trend (recent/prior)"},
}


def _norm_lang(lang: str | None) -> str:
    l = (lang or "").strip().lower()
    return "en" if l.startswith("en") else "tr"


def _clean(col: str, lang: str) -> str:
    """İç kolon token'ını sadeleştirir: türev bir isimse kuralı yine uygular
    (özyineleme prefix soyarak sonlanır), değilse alt çizgi → boşluk."""
    label = humanize_feature_label(col, lang)
    return label if label != col else col.replace("_", " ")


def humanize_feature_label(name: str, lang: str = "tr") -> str:
    """Feature ismini sade karşılığına çevirir (ham isim EKLENMEZ).
    Kural eşleşmezse alt çizgileri boşluğa çevirir (ham casing korunur)."""
    lang = _norm_lang(lang)
    if not name or not isinstance(name, str):
        return name

    def tpl(key: str, **kw) -> str:
        return _T[key][lang].format(**kw)

    # Sabit RFM isimleri (motora bağlı, veriye değil)
    if name in ("spend_recent", "spend_old", "spend_trend"):
        return tpl(name)

    # lag_{k}_{col}
    m = re.match(r"^lag_(\d+)_(.+)$", name)
    if m:
        return tpl("lag", k=m.group(1), col=_clean(m.group(2), lang))

    # roll{w}_mean_{col} / roll{w}_std_{col}
    m = re.match(r"^roll(\d+)_mean_(.+)$", name)
    if m:
        return tpl("roll_mean", w=m.group(1), col=_clean(m.group(2), lang))
    m = re.match(r"^roll(\d+)_std_(.+)$", name)
    if m:
        return tpl("roll_std", w=m.group(1), col=_clean(m.group(2), lang))

    # dev/zdev/grpstd_{num}_by_{cat} — '_by_' ile net bölme (prefix = şablon anahtarı)
    m = re.match(r"^(dev|zdev|grpstd)_(.+)_by_(.+)$", name)
    if m:
        return tpl(m.group(1), num=_clean(m.group(2), lang), cat=_clean(m.group(3), lang))

    # is_zero_{col} / was_missing_{col}
    m = re.match(r"^is_zero_(.+)$", name)
    if m:
        return tpl("is_zero", col=_clean(m.group(1), lang))
    m = re.match(r"^was_missing_(.+)$", name)
    if m:
        return tpl("was_missing", col=_clean(m.group(1), lang))

    # count_/sum_/avg_/max_/std_/fav_{col}
    m = re.match(r"^(count|sum|avg|max|std|fav)_(.+)$", name)
    if m:
        return tpl(m.group(1), col=_clean(m.group(2), lang))

    # ratio_{a}_{b} — son alt çizgiden böl (payda = son token)
    m = re.match(r"^ratio_(.+)_([^_]+)$", name)
    if m:
        return tpl("ratio", a=_clean(m.group(1), lang), b=_clean(m.group(2), lang))

    # {col}_isna / {col}_diversity (suffix)
    m = re.match(r"^(.+)_isna$", name)
    if m:
        return tpl("isna", col=_clean(m.group(1), lang))
    m = re.match(r"^(.+)_diversity$", name)
    if m:
        return tpl("diversity", col=_clean(m.group(1), lang))

    # fallback: alt çizgi → boşluk (ham casing korunur)
    return name.replace("_", " ")


def humanize_feature(name: str, lang: str = "tr") -> str:
    """Rapor gösterimi: 'Sade etiket (ham_isim)'. Ham isim uzman için parantezde
    kalır. Kural eşleşmediyse (etiket ham ile aynıysa) parantez eklenmez."""
    label = humanize_feature_label(name, lang)
    if not name or label == name:
        return name
    return f"{label} ({name})"
