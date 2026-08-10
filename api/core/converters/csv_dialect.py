"""
api/core/converters/csv_dialect.py
----------------------------------
Ham CSV byte'larından kodlama / ayırıcı / sayı biçimi tespiti.

Neden gerekli: Türkiye'de Excel'den çıkan tipik CSV `;` ile ayrılır,
Windows-1254 (≈ISO-8859-9) kodlanır ve sayılar `1.234,56` biçimindedir.
`pd.read_csv` varsayılanları (`sep=","`, `encoding="utf-8"`) bu dosyayı ya tek
sütuna çökertir ya da UnicodeDecodeError ile reddeder.

Hem CSV normalize dönüştürücüsü (`csv_normalize.py`) hem de yükleme hattının
son-çare fallback'i (`api/core/csv_ingest.py`) bu modülü kullanır.
"""

import csv
import io
import re
from typing import Any, Dict, List, Optional

# Tek-byte kodlamalar aynı byte'ı farklı harfe eşler (0xFE: cp1254'te 'ş',
# cp1250'de 'ţ', cp1252'de 'þ') — hiçbiri hata vermez, dolayısıyla "çözülüyor
# mu" testi bunları ayırt EDEMEZ. Bu yüzden adayları Türkçe harf sayısına göre
# puanlıyoruz; sıra eşitlikte tie-break'tir (cp1254 önce).
_ENCODING_CANDIDATES = ("cp1254", "cp1252", "cp1250", "iso-8859-9", "latin-1")

# Türkçeye özgü harfler — puanlama bunlara bakar.
_TURKISH_CHARS = frozenset("çğıöşüÇĞİÖŞÜ")

# BOM imzaları
_BOM_ENCODINGS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)

_DELIMITER_CANDIDATES = (",", ";", "\t", "|")

# Örnekleme sınırları — büyük dosyada tüm içeriği taramaya gerek yok.
_SAMPLE_BYTES = 256 * 1024
_SAMPLE_LINES = 50

# 1.234,56 / 1234,56 → ondalık virgül; 1,234.56 / 1234.56 → ondalık nokta
_TR_NUMBER = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$|^-?\d+,\d+$")
_EN_NUMBER = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$|^-?\d+\.\d+$")


def detect_encoding(content: bytes) -> str:
    """Byte'ların kodlamasını tespit eder; her zaman kullanılabilir bir ad döner."""
    for bom, encoding in _BOM_ENCODINGS:
        if content.startswith(bom):
            return encoding

    sample = content[:_SAMPLE_BYTES]

    # UTF-8 sorunsuz çözülüyorsa tartışma yok.
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # Tek-byte adaylar arasında Türkçe harf sayısı en yüksek olanı seç.
    # (Genel amaçlı tahmin ediciler burada cp1250'yi cp1254'e tercih edip
    # "Müşteri"yi "Müţteri" yapıyor — ürün Türkçe veriyle çalışıyor.)
    best_encoding: Optional[str] = None
    best_score = 0
    for encoding in _ENCODING_CANDIDATES:
        try:
            text = sample.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        score = sum(1 for ch in text if ch in _TURKISH_CHARS)
        if score > best_score:
            best_score = score
            best_encoding = encoding

    if best_encoding:
        return best_encoding

    # Türkçe sinyali yok → dosya başka bir dile ait olabilir; genel tahmin ediciye sor.
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(sample).best()
        if best and best.encoding:
            guess = best.encoding.lower().replace("_", "-")
            return "cp1254" if guess == "windows-1254" else guess
    except Exception:
        pass

    for encoding in _ENCODING_CANDIDATES:
        try:
            sample.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue

    return "latin-1"


def _sample_lines(text: str) -> List[str]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[:_SAMPLE_LINES]


def detect_delimiter(text: str) -> str:
    """
    Ayırıcıyı satırlar arası tutarlılığa bakarak seçer.

    csv.Sniffer tek satıra fazla güvendiği için önce kendi sayımımızı yaparız:
    "her satırda aynı sayıda ve en az bir kez geçen" aday kazanır.
    """
    lines = _sample_lines(text)
    if not lines:
        return ","

    best_delim = None
    best_score = (0, 0.0)  # (alan sayısı, tutarlılık oranı)

    for delim in _DELIMITER_CANDIDATES:
        try:
            rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
        except csv.Error:
            continue
        counts = [len(r) for r in rows if r]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal < 2:
            continue
        consistency = counts.count(modal) / len(counts)
        score = (modal, consistency)
        # Daha çok sütun + daha tutarlı olan kazanır; eşitlikte aday sırası korunur.
        if consistency >= 0.7 and score > best_score:
            best_score = score
            best_delim = delim

    if best_delim:
        return best_delim

    try:
        return csv.Sniffer().sniff(
            "\n".join(lines[:20]), delimiters="".join(_DELIMITER_CANDIDATES)
        ).delimiter
    except csv.Error:
        return ","


def looks_like_tr_number(value: str) -> bool:
    """'1.234,56' / '987,40' gibi Türkçe biçimli sayı mı?"""
    return bool(_TR_NUMBER.match((value or "").strip()))


def detect_number_format(text: str, delimiter: str) -> Dict[str, Optional[str]]:
    """
    Sayı biçimini örnek hücrelere bakarak tespit eder.

    Returns:
        {"decimal": "." | ",", "thousands": "." | "," | None}
    """
    lines = _sample_lines(text)
    if len(lines) < 2:
        return {"decimal": ".", "thousands": None}

    rows = list(csv.reader(io.StringIO("\n".join(lines[1:])), delimiter=delimiter))
    tr_hits = 0
    en_hits = 0
    tr_grouped = False
    en_grouped = False

    for row in rows:
        for cell in row:
            value = cell.strip()
            if not value:
                continue
            if _TR_NUMBER.match(value):
                tr_hits += 1
                tr_grouped = tr_grouped or "." in value
            elif _EN_NUMBER.match(value):
                en_hits += 1
                en_grouped = en_grouped or "," in value

    if tr_hits > en_hits:
        return {"decimal": ",", "thousands": "." if tr_grouped else None}
    if en_hits > 0:
        return {"decimal": ".", "thousands": "," if en_grouped else None}
    return {"decimal": ".", "thousands": None}


def detect_dialect(content: bytes) -> Dict[str, Any]:
    """
    Kodlama + ayırıcı + sayı biçimini birlikte tespit eder.

    Returns:
        {"encoding", "delimiter", "decimal", "thousands"}
    """
    encoding = detect_encoding(content)
    try:
        text = content[:_SAMPLE_BYTES].decode(encoding, errors="replace")
    except LookupError:
        encoding = "latin-1"
        text = content[:_SAMPLE_BYTES].decode(encoding, errors="replace")

    delimiter = detect_delimiter(text)
    numbers = detect_number_format(text, delimiter)

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "decimal": numbers["decimal"],
        "thousands": numbers["thousands"],
    }
