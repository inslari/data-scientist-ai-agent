"""
Dayanıklı CSV okuma — kodlama + ayraç tespiti
=============================================
Excel/Türkçe export'ları sık sık UTF-16 (BOM 0xFF 0xFE), UTF-8-BOM veya
Windows-1254 (cp1254) kodlamasıyla ve ',' yerine ';' / TAB ayracıyla gelir.
Düz `pd.read_csv(...)` bunları "'utf-8' codec can't decode byte 0xff" gibi
hatalarla reddeder. Bu modül kodlamayı (BOM → charset_normalizer → cp1254 yedek)
ve ayracı (başlık satırına göre) tespit edip güvenle okur.

Tüm CSV giriş noktaları (upload, örnek veri, kolon istatistiği, text-analysis
kolon çıkarımı) bu yardımcıları kullanır.
"""
from __future__ import annotations

import io

import pandas as pd

# BOM imzaları → kodlama (sıra önemli: UTF-32 BOM, UTF-16 BOM'u kapsar).
_BOM_ENCODINGS = [
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
]

_CANDIDATE_SEPS = [",", ";", "\t", "|"]


def detect_encoding(sample: bytes) -> str:
    """Bir bayt örneğinden kodlamayı tespit eder. BOM → utf-8 → charset_normalizer → cp1254."""
    for bom, enc in _BOM_ENCODINGS:
        if sample.startswith(bom):
            return enc
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(sample).best()
        if best and best.encoding:
            enc = best.encoding.lower().replace("_", "-")
            # cp1252/latin-1 ile cp1254 yalnızca birkaç Türkçe karakterde (ş/ğ/İ/ı)
            # ayrışır; küçük örnekte sık karışır. Türkçe odaklı ürün → Latin-1
            # ailesini cp1254'e yönlendir (ş→þ gibi bozulmayı önler).
            if enc in ("cp1252", "windows-1252", "latin-1", "latin1", "iso-8859-1"):
                return "cp1254"
            return best.encoding
    except Exception:
        pass
    return "cp1254"  # Türkçe Windows yedeği (iso-8859-9 ile uyumlu)


def _sniff_sep(text: str) -> str:
    """İlk anlamlı satıra göre ayracı tahmin eder; belirsizse virgül."""
    for line in text.splitlines():
        if line.strip():
            counts = {d: line.count(d) for d in _CANDIDATE_SEPS}
            best = max(counts, key=lambda k: counts[k])
            return best if counts[best] > 0 else ","
    return ","


def read_csv_bytes(data: bytes, **kwargs) -> pd.DataFrame:
    """
    Bayt içeriğinden DataFrame okur; kodlama ve (verilmemişse) ayraç otomatik
    tespit edilir. Kalan kwargs (nrows, usecols, ...) pandas'a geçer.
    """
    enc = detect_encoding(data[:8192])
    text = data.decode(enc, errors="replace")
    if "sep" not in kwargs and "delimiter" not in kwargs:
        kwargs["sep"] = _sniff_sep(text[:8192])
    return pd.read_csv(io.StringIO(text), **kwargs)


def read_csv_path(path: str, **kwargs) -> pd.DataFrame:
    """Yerel bir dosyadan dayanıklı CSV okur (kodlama + ayraç tespitli)."""
    with open(path, "rb") as f:
        data = f.read()
    return read_csv_bytes(data, **kwargs)
