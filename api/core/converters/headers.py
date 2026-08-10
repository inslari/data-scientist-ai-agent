"""
api/core/converters/headers.py
------------------------------
Çok satırlı başlıkları tek satıra indirger.

Anket/rapor export'larında başlık sık sık iki satırdır: üstte soru metni, altta
alt etiketler (matris sorularının seçenekleri). Tek satır başlık varsayılırsa
alt etiket satırı VERİ olarak kalır — sayı sütunları metne döner ve kimse fark
etmez. `header_rows` ayarı bu satırları başlığın parçası sayar; buradaki
`flatten_columns` de pandas'ın ürettiği MultiIndex'i okunabilir tek isme çevirir.

pandas'ın bıraktığı iki artığı temizliyoruz:
  - `Unnamed: 3_level_0` gibi yer tutucular → boş kabul edilir
  - birleştirilmiş (merged) hücrelerde soru yalnızca ilk sütunda durur →
    üst seviyeler sağa doğru ileri doldurulur
"""

import re
from typing import Any, List, Sequence

import pandas as pd

DEFAULT_SEPARATOR = " - "

_UNNAMED = re.compile(r"^Unnamed:\s*\d+(_level_\d+)?$")


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.lower() == "nan" or _UNNAMED.match(text):
        return ""
    return text


def _dedupe(names: Sequence[str]) -> List[str]:
    """Aynı ada sahip kolonlara sayaç ekler (pandas'ın `.1` eki yerine `_2`)."""
    seen: dict = {}
    result: List[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        result.append(name if count == 1 else f"{name}_{count}")
    return result


def flatten_columns(columns: pd.Index, separator: str = DEFAULT_SEPARATOR) -> List[str]:
    """
    MultiIndex kolonları tek satırlık adlara indirger.

    `[("Senaryo puanı", "ihtimal1"), ("Unnamed: 3_level_0", "ihtimal2")]`
    →  `["Senaryo puanı - ihtimal1", "Senaryo puanı - ihtimal2"]`
    """
    if not isinstance(columns, pd.MultiIndex):
        return _dedupe([_clean(c) or f"column_{i + 1}" for i, c in enumerate(columns)])

    depth = columns.nlevels
    levels = [[_clean(col[i]) for col in columns] for i in range(depth)]

    # Birleştirilmiş hücreler: ÜST seviyeleri sağa doğru ileri doldur. Son seviye
    # doldurulmaz — orada boşluk gerçekten "bu sütunun alt etiketi yok" demektir.
    for level in levels[:-1]:
        last = ""
        for i, value in enumerate(level):
            if value:
                last = value
            else:
                level[i] = last

    names = []
    for col_index in range(len(columns)):
        parts: List[str] = []
        for level in levels:
            part = level[col_index]
            if part and part not in parts:
                parts.append(part)
        names.append(separator.join(parts) if parts else f"column_{col_index + 1}")

    return _dedupe(names)


def build_preview(df: pd.DataFrame, rows: int = 8) -> dict:
    """
    Arayüzün gösterdiği önizleme tablosunu üretir.

    Boş hücreler boş string olmalı: `str(nan)` "nan" verir ve kullanıcı bunu
    veri sanır. `where(notnull, None)` sayısal sütunlarda NaN'ı None'a çevirmez,
    o yüzden hücre bazında `pd.isna` ile bakıyoruz.
    """
    head = df.head(rows)
    return {
        "columns": [str(c) for c in df.columns],
        "rows": [
            ["" if pd.isna(value) else str(value) for value in row]
            for row in head.itertuples(index=False, name=None)
        ],
    }


def looks_like_stray_label_row(df: pd.DataFrame) -> int:
    """
    İlk veri satırı aslında başlığın ikinci satırı mı?

    İmza şu: bir sütunun ilk değeri METİN, ama altındaki değerlerin neredeyse
    hepsi SAYI. Tek sütunda tesadüf olabilir; iki ya da daha fazla sütunda aynı
    anda oluyorsa o satır veri değil, etikettir.

    Returns:
        Bu imzayı gösteren sütun sayısı (0 = şüphe yok).
    """
    if len(df) < 3:
        return 0

    suspicious = 0
    for col in df.columns:
        series = df[col]
        first = series.iloc[0]
        if first is None or (isinstance(first, float) and pd.isna(first)):
            continue
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            continue  # ilk değer zaten sayı → etiket değil

        rest = series.iloc[1:]
        numeric = pd.to_numeric(rest, errors="coerce")
        known = rest.notna().sum()
        if known >= 2 and numeric.notna().sum() >= known * 0.8:
            suspicious += 1

    return suspicious if suspicious >= 2 else 0


def header_argument(has_header: bool, header_rows: Any) -> Any:
    """
    pandas'ın `header` parametresini üretir.

    None  → başlık yok
    0     → tek satır başlık
    [0,1] → çok satırlı başlık (MultiIndex, sonra flatten_columns ile indirgenir)
    """
    if not has_header:
        return None
    try:
        rows = int(header_rows or 1)
    except (TypeError, ValueError):
        rows = 1
    rows = max(1, min(rows, 5))
    return 0 if rows == 1 else list(range(rows))
