"""
api/core/converters/csv_normalize.py
------------------------------------
CSV → CSV: bozuk/yerelleştirilmiş CSV'yi platformun okuyabileceği hale getirir.

Kapattığı sorun: Türkiye'de Excel'den çıkan CSV `;` ayırıcı, Windows-1254
kodlama ve `1.234,56` sayı biçimi kullanır. Yükleme hattının varsayılanları
(`sep=","`, `encoding="utf-8"`) bu dosyayı tek sütuna çökertir ya da reddeder;
kullanıcı hatayı ancak analiz saçmaladığında fark eder.

Çıktı her zaman kanonik biçimdedir: UTF-8, tek ayırıcı, ondalık nokta.
"""

import io
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from api.core.converters.base import (
    BaseConverter,
    ConversionError,
    ConvertedFile,
    ConverterOption,
)
from api.core.converters.csv_dialect import detect_dialect
from api.core.converters.headers import (
    build_preview,
    flatten_columns,
    header_argument,
    looks_like_stray_label_row,
)

# Önizlemede gösterilecek satır sayısı
_PREVIEW_ROWS = 8

_ENCODING_CHOICES = [
    {"value": "utf-8", "label_key": "utf8"},
    {"value": "utf-8-sig", "label_key": "utf8Bom"},
    {"value": "cp1254", "label_key": "cp1254"},
    {"value": "iso-8859-9", "label_key": "iso88599"},
    {"value": "latin-1", "label_key": "latin1"},
]

_DELIMITER_CHOICES = [
    {"value": ",", "label_key": "comma"},
    {"value": ";", "label_key": "semicolon"},
    {"value": "\t", "label_key": "tab"},
    {"value": "|", "label_key": "pipe"},
]


def _read_frame(content: bytes, opts: Dict[str, Any]) -> pd.DataFrame:
    """Kullanıcı ayarlarıyla (ya da tespit edilenlerle) DataFrame okur."""
    thousands = opts.get("thousands") or None
    decimal = opts.get("decimal") or "."
    # pandas aynı karakteri hem ondalık hem binlik ayracı kabul etmez.
    if thousands == decimal:
        thousands = None

    try:
        return pd.read_csv(
            io.BytesIO(content),
            sep=opts.get("delimiter") or ",",
            encoding=opts.get("encoding") or "utf-8",
            decimal=decimal,
            thousands=thousands,
            header=header_argument(opts.get("has_header", True), opts.get("header_rows")),
            skiprows=int(opts.get("skip_rows") or 0) or None,
            skip_blank_lines=True,
        )
    except UnicodeDecodeError as e:
        raise ConversionError(
            f"Dosya '{opts.get('encoding')}' kodlamasıyla okunamadı: {e}"
        ) from e
    except Exception as e:
        raise ConversionError(f"CSV okunamadı: {e}") from e


def _clean_frame(df: pd.DataFrame, opts: Dict[str, Any]) -> pd.DataFrame:
    if not opts.get("has_header", True):
        df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
    # Çok satırlı başlık MultiIndex üretir; clean_headers kapalı olsa da indirgenmeli.
    elif opts.get("clean_headers", True) or isinstance(df.columns, pd.MultiIndex):
        df.columns = flatten_columns(df.columns)

    if opts.get("drop_empty", True):
        # Excel export'larının bıraktığı "Unnamed: 5" tipi boş sütunlar ve boş satırlar
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if opts.get("trim_values", True):
        for col in df.columns:
            # pandas 3'te metin sütunu `object` değil `StringDtype` — sayısal
            # olmayan her sütunda dener, str olmayan değerlere dokunmayız.
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    return df


def _collect_issues(
    df: pd.DataFrame,
    detected: Dict[str, Any],
    raw_columns: int,
    opts: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Arayüzde uyarı olarak gösterilecek bulgular (i18n kodu + parametre)."""
    issues: List[Dict[str, Any]] = []
    opts = opts or {}

    if df.shape[1] <= 1:
        issues.append({"code": "singleColumn", "params": {}})

    # Anket export'ları: ilk veri satırı aslında başlığın ikinci satırı olabilir.
    if int(opts.get("header_rows") or 1) == 1 and opts.get("has_header", True):
        stray = looks_like_stray_label_row(df)
        if stray:
            issues.append({"code": "possibleHeaderRow", "params": {"count": stray}})

    encoding = (detected.get("encoding") or "").lower()
    if encoding not in ("utf-8", "utf-8-sig"):
        issues.append({"code": "nonUtf8", "params": {"encoding": detected.get("encoding")}})

    if detected.get("decimal") == ",":
        issues.append({"code": "commaDecimal", "params": {}})

    dropped = raw_columns - df.shape[1]
    if dropped > 0:
        issues.append({"code": "emptyColumns", "params": {"count": dropped}})

    duplicates = len(df.columns) - len(set(map(str, df.columns)))
    if duplicates > 0:
        issues.append({"code": "duplicateHeaders", "params": {"count": duplicates}})

    return issues


class CsvNormalizeConverter(BaseConverter):
    id = "csv_normalize"
    source_extensions = (".csv", ".txt", ".tsv")
    target_extension = ".csv"
    target_media_type = "text/csv"
    label_key = "csv_normalize"
    description_key = "csv_normalize"
    produces_multiple = False
    max_file_mb = 300
    live_preview = True

    options = [
        ConverterOption(
            name="encoding",
            type="select",
            default="utf-8",
            default_source="detected.encoding",
            label_key="sourceEncoding",
            help_key="sourceEncoding",
            choices=_ENCODING_CHOICES,
        ),
        ConverterOption(
            name="delimiter",
            type="select",
            default=",",
            default_source="detected.delimiter",
            label_key="sourceDelimiter",
            help_key="sourceDelimiter",
            choices=_DELIMITER_CHOICES,
        ),
        ConverterOption(
            name="decimal",
            type="select",
            default=".",
            default_source="detected.decimal",
            label_key="decimal",
            help_key="decimal",
            choices=[
                {"value": ".", "label_key": "dot"},
                {"value": ",", "label_key": "comma"},
            ],
        ),
        ConverterOption(
            name="thousands",
            type="select",
            default="",
            default_source="detected.thousands",
            label_key="thousands",
            choices=[
                {"value": "", "label_key": "none"},
                {"value": ".", "label_key": "dot"},
                {"value": ",", "label_key": "comma"},
                {"value": " ", "label_key": "space"},
            ],
        ),
        ConverterOption(
            name="has_header",
            type="boolean",
            default=True,
            label_key="hasHeader",
            help_key="hasHeader",
        ),
        ConverterOption(
            name="header_rows",
            type="number",
            default=1,
            min=1,
            max=5,
            label_key="headerRows",
            help_key="headerRows",
        ),
        ConverterOption(
            name="skip_rows",
            type="number",
            default=0,
            min=0,
            max=1000,
            label_key="skipRows",
            help_key="skipRows",
        ),
        ConverterOption(
            name="output_delimiter",
            type="select",
            default=",",
            label_key="outputDelimiter",
            help_key="outputDelimiter",
            choices=_DELIMITER_CHOICES,
        ),
        ConverterOption(
            name="clean_headers",
            type="boolean",
            default=True,
            label_key="cleanHeaders",
            help_key="cleanHeaders",
        ),
        ConverterOption(
            name="drop_empty",
            type="boolean",
            default=True,
            label_key="dropEmpty",
            help_key="dropEmpty",
        ),
        ConverterOption(
            name="trim_values",
            type="boolean",
            default=True,
            label_key="trimValues",
            help_key="trimValues",
        ),
    ]

    # ------------------------------------------------------------------

    def _effective(self, content: bytes, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Kullanıcı ayarları + tespit edilenleri birleştirir.

        Kullanıcının açıkça verdiği değer her zaman kazanır; vermediklerinde
        tespit sonucu kullanılır (arayüz de aynı önceliği uygular).
        """
        detected = detect_dialect(content)
        resolved = self.resolve_options(options)
        provided = options or {}

        for key in ("encoding", "delimiter", "decimal", "thousands"):
            if provided.get(key) in (None, ""):
                value = detected.get(key)
                resolved[key] = "" if value is None else value

        return {"detected": detected, "options": resolved}

    def inspect(
        self, content: bytes, filename: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        state = self._effective(content, options)
        opts = state["options"]

        df = _read_frame(content, opts)
        raw_columns = df.shape[1]
        df = _clean_frame(df, opts)

        # Tespit edilen değil, gerçekten KULLANILAN ayarları döndür — arayüzdeki
        # rozetler ve varsayılanlar bununla beslenir.
        used = {
            "encoding": opts.get("encoding"),
            "delimiter": opts.get("delimiter"),
            "decimal": opts.get("decimal"),
            "thousands": opts.get("thousands") or None,
        }

        return {
            "detected": used,
            "preview": build_preview(df, _PREVIEW_ROWS),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "issues": _collect_issues(df, used, raw_columns, opts),
        }

    # ------------------------------------------------------------------

    def convert(self, content: bytes, filename: str, options: Dict[str, Any]) -> List[ConvertedFile]:
        state = self._effective(content, options)
        opts = state["options"]

        df = _clean_frame(_read_frame(content, opts), opts)
        if df.empty:
            raise ConversionError("Dönüştürülecek veri bulunamadı (dosya boş).")

        # Çıktı her zaman kanonik: UTF-8 + ondalık nokta.
        csv_bytes = df.to_csv(
            index=False, sep=opts.get("output_delimiter") or ","
        ).encode("utf-8", errors="replace")

        stem = re.sub(r"\.(csv|txt|tsv)$", "", filename or "data", flags=re.IGNORECASE)
        stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip("._") or "data"

        return [
            ConvertedFile(
                filename=f"{stem}_normalized{self.target_extension}",
                content=csv_bytes,
                media_type=self.target_media_type,
                meta={
                    "rows": int(len(df)),
                    "columns": int(df.shape[1]),
                    "size": len(csv_bytes),
                    "source_encoding": opts.get("encoding"),
                    "source_delimiter": opts.get("delimiter"),
                },
            )
        ]
