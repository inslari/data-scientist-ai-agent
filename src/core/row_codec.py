"""
Row Codec — MCP/veri-grubu sorgu sonuçları için sağlam, simetrik satır serializer.
==================================================================================
Sorgu sonuçları süreçler arasında metin olarak taşınır:

    Columns: c1, c2, c3
    Returned N rows.
    Results:
    (v1, v2, v3)
    ...

Eski üretim `str(tuple(row))` idi. Bu, PRİMİTİFLER (int/float/str/bool/None) için
Python repr'i sayesinde virgül/tırnak/newline/unicode'u DOĞRU kaçışlar. Sorun,
repr'i bir *constructor çağrısı* olan non-literal nesnelerdi:

    Timestamp('2024-01-06 07:09:39+0300', tz='Europe/Istanbul')   # içteki virgül!
    datetime.datetime(2024, 1, 6, 7, 9, 39, tzinfo=...)           # bir sürü virgül
    Decimal('10.5')

Bunlar `ast.literal_eval` ile parse EDİLEMEZ (çağrı ifadesi, literal değil) ve
istemcideki regex-fallback içteki virgülde satırı bölerek TÜM sağ kolonları kaydırır
(kolon kayması). Çözüm biçim değiştirmek değil: her hücreyi serialize'dan ÖNCE
literal-güvenli bir primitife indirgemek. Böylece `str(tuple(row))` daima geçerli bir
Python-literal tuple üretir ve `ast.literal_eval` (tüm tüketicilerde zaten var) onu
KAYIPSIZ, KAYMASIZ parse eder.

Dinamik/genel: coercion TİP-TABANLIDIR — kolon adı/tipe özel dal yok, timestamp'e
özel patch yok. İçinde virgül/özel karakter/tz taşıyan HERHANGİ bir değer türü
güvenli serialize edilir.

Simetri: `encode_result` (üretici) ve `decode_result` (istemci) burada, tek
doğruluk kaynağında. Deserialize ast-only'dir (kırılgan regex YOK); parse edilemeyen
satır KARANTİNAYA alınır (Veri Kalitesi Katmanı 1a — sessiz tahmin yok).
"""

from __future__ import annotations

import ast
import math
import numbers
from typing import Any, Iterable, Sequence

try:  # pandas opsiyonel: NaT / pd.NA tespiti için (yoksa graceful skip)
    import pandas as _pd
except Exception:  # pragma: no cover
    _pd = None


def _isna_safe(v: Any) -> bool:
    """pandas NaT / NA gibi skaler eksik-değerleri güvenle tespit eder.

    pd.isna dizide array döndürür → yalnız skalerlerde kullan; hata olursa False.
    """
    if _pd is None:
        return False
    try:
        res = _pd.isna(v)
        return bool(res) if isinstance(res, (bool,)) or getattr(res, "ndim", 0) == 0 else False
    except (ValueError, TypeError):
        return False


def _coerce_cell(v: Any) -> Any:
    """Bir hücreyi literal-güvenli primitife indirger (int/float/str/bool/None).

    - None                         → None
    - bool                         → bool         (int'ten ÖNCE; bool, Integral'dır)
    - Integral (numpy int64 dahil) → int
    - Real (numpy float64 dahil)   → float; NaN/Inf → None (ast 'nan' literal'ı yok!)
    - str                          → str          (repr kaçışlamayı halleder)
    - NaT / pd.NA                  → None
    - diğer (Timestamp/datetime/Decimal/UUID/bytes/…) → str  (güvenli, virgülsüz repr)
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, str):
        return v
    if _isna_safe(v):
        return None
    return str(v)


def encode_row(row: Sequence[Any]) -> str:
    """Tek satırı literal-güvenli tuple repr'ine çevirir (tek satırlık, ast-parse'lı)."""
    return repr(tuple(_coerce_cell(c) for c in row))


def encode_result(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    max_rows: int = 10_000,
) -> str:
    """Sorgu sonucunu kanonik envelope + literal-güvenli satırlarla serialize eder.

    Her iki üretici (MCP DB yolu ve CSV/DuckDB yolu) bunu kullanır → tek biçim,
    drift yok. Çıktı, tüm mevcut `ast.literal_eval` tabanlı tüketicilerle uyumludur.
    """
    rows = list(rows)
    out: list[str] = [
        f"Columns: {', '.join(str(c) for c in columns)}",
        f"Returned {len(rows)} rows.",
        "Results:",
    ]
    for row in rows[:max_rows]:
        out.append(encode_row(row))
    if len(rows) > max_rows:
        out.append(f"... (Showing only first {max_rows} rows)")
    return "\n".join(out)


def decode_row(line: str) -> tuple:
    """Tek satırı ast ile tuple'a çevirir. Parse edilemezse hata yükseltir.

    Kırılgan regex YOK: literal değilse çağıran karantinaya alır (Layer 1a).
    """
    val = ast.literal_eval(line)
    if not isinstance(val, tuple):
        val = (val,)
    return val


def decode_result(text: str) -> tuple[list[str], list[tuple], dict]:
    """Kanonik envelope'u kolonlara + satırlara ayrıştırır (ast-only, quarantine-aware).

    Dönüş: (columns, rows, structural)
      structural: Veri Kalitesi Katmanı 1a yapısal raporu
        {expected_cols, total_rows, quarantined_rows, mismatch_rate, sample_bad_rows}

    Alan-sayısı invaryantı: her satır TAM olarak len(columns) alan üretmeli. Aksi
    (veya ast hatası) → satır KARANTİNAYA alınır (sessiz pad/trim yok).
    """
    lines = text.strip().splitlines() if text else []

    columns: list[str] = []
    for line in lines:
        if line.startswith("Columns:"):
            columns = [c.strip() for c in line.replace("Columns:", "").split(",")]
            break
    expected = len(columns)

    rows: list[tuple] = []
    quarantined: list[str] = []
    for line in lines:
        s = line.strip()
        if not (s.startswith("(") and s.endswith(")")):
            continue
        try:
            parsed = decode_row(s)
        except Exception:
            quarantined.append(s)
            continue
        if expected and len(parsed) != expected:
            quarantined.append(s)          # alan sayısı ≠ header → kayma; tahmin etme
        else:
            rows.append(parsed)

    total = len(rows) + len(quarantined)
    structural = {
        "expected_cols":    expected,
        "total_rows":       total,
        "quarantined_rows": len(quarantined),
        "mismatch_rate":    round(len(quarantined) / max(total, 1), 4),
        "sample_bad_rows":  quarantined[:5],
    }
    return columns, rows, structural
