"""
api/core/converters/excel_to_csv.py
-----------------------------------
Excel (.xlsx / .xlsm / .xls) → CSV dönüşümü.

Çalışma kitabındaki her sayfa (sheet) ayrı bir CSV dosyasına dönüşür; kullanıcı
isterse yalnızca belirli sayfaları seçebilir.
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
from api.core.converters.headers import (
    build_preview,
    flatten_columns,
    header_argument,
    looks_like_stray_label_row,
)

# Bu boyutun üstünde `inspect` yalnızca sayfa adlarını döndürür; satır/sütun
# sayısı için tüm kitabı ayrıştırmak pahalı olur.
_LIGHT_INSPECT_BYTES = 25 * 1024 * 1024

# Önizlemede gösterilecek satır sayısı
_PREVIEW_ROWS = 8


def _safe_name(value: str) -> str:
    """Sayfa adını dosya adında güvenle kullanılabilir hale getirir."""
    cleaned = re.sub(r"[^\w\-. ]+", "_", (value or "").strip(), flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned or "sheet"


def _open_workbook(content: bytes) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(io.BytesIO(content))
    except ImportError as e:
        # openpyxl / xlrd kurulu değilse pandas ImportError fırlatır.
        raise ConversionError(
            f"Excel okuma motoru eksik: {e}. (openpyxl / xlrd kurulu olmalı)"
        ) from e
    except Exception as e:
        raise ConversionError(f"Excel dosyası açılamadı: {e}") from e


class ExcelToCsvConverter(BaseConverter):
    id = "excel_to_csv"
    source_extensions = (".xlsx", ".xlsm", ".xls")
    target_extension = ".csv"
    target_media_type = "text/csv"
    label_key = "excel_to_csv"
    description_key = "excel_to_csv"
    produces_multiple = True
    max_file_mb = 100
    live_preview = True

    options = [
        ConverterOption(
            name="sheets",
            type="multiselect",
            default=None,          # boş = tüm sayfalar
            source="sheets",       # seçenekler inspect çıktısından gelir
            label_key="sheets",
            help_key="sheets",
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
            name="delimiter",
            type="select",
            default=",",
            label_key="delimiter",
            choices=[
                {"value": ",", "label_key": "comma"},
                {"value": ";", "label_key": "semicolon"},
                {"value": "\t", "label_key": "tab"},
                {"value": "|", "label_key": "pipe"},
            ],
        ),
        ConverterOption(
            name="encoding",
            type="select",
            default="utf-8",
            label_key="encoding",
            choices=[
                {"value": "utf-8", "label_key": "utf8"},
                {"value": "utf-8-sig", "label_key": "utf8Bom"},
                {"value": "iso-8859-9", "label_key": "iso88599"},
            ],
        ),
        ConverterOption(
            name="clean_headers",
            type="boolean",
            default=True,
            label_key="cleanHeaders",
            help_key="cleanHeaders",
        ),
        ConverterOption(
            name="skip_empty_sheets",
            type="boolean",
            default=True,
            label_key="skipEmptySheets",
            help_key="skipEmptySheets",
        ),
    ]

    # ------------------------------------------------------------------

    def inspect(
        self, content: bytes, filename: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        xf = _open_workbook(content)
        detailed = len(content) <= _LIGHT_INSPECT_BYTES
        opts = self.resolve_options(options)

        sheets: List[Dict[str, Any]] = []
        for name in xf.sheet_names:
            entry: Dict[str, Any] = {"name": name}
            if detailed:
                try:
                    raw = xf.parse(name, header=None)
                except Exception as e:
                    entry["error"] = str(e)[:200]
                    sheets.append(entry)
                    continue
                entry["rows"] = max(len(raw) - 1, 0)  # ilk satır başlık varsayılır
                entry["columns"] = int(raw.shape[1])
                entry["empty"] = bool(raw.empty)
                entry["preview_columns"] = (
                    [str(v) for v in raw.iloc[0].tolist()[:12]] if len(raw) else []
                )
            sheets.append(entry)

        result: Dict[str, Any] = {"sheets": sheets, "detailed": detailed}

        # Seçili ilk sayfanın önizlemesi: kullanıcı dönüştürmeden ÖNCE başlık
        # satırlarının doğru ayrıldığını görebilsin (anket export'ları burada
        # sessizce bozuluyordu).
        selected = opts.get("sheets") or []
        if isinstance(selected, str):
            selected = [selected]
        preview_sheets = [s for s in xf.sheet_names if not selected or s in selected]
        if preview_sheets:
            try:
                df = xf.parse(
                    preview_sheets[0],
                    header=header_argument(opts.get("has_header", True), opts.get("header_rows")),
                    skiprows=int(opts.get("skip_rows") or 0) or None,
                )
                if opts.get("has_header", True):
                    df.columns = flatten_columns(df.columns)
                else:
                    df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]

                result["preview"] = build_preview(df, _PREVIEW_ROWS)
                result["rows"] = int(len(df))
                result["columns"] = int(df.shape[1])
                result["preview_sheet"] = preview_sheets[0]

                issues: List[Dict[str, Any]] = []
                if int(opts.get("header_rows") or 1) == 1 and opts.get("has_header", True):
                    stray = looks_like_stray_label_row(df)
                    if stray:
                        issues.append({"code": "possibleHeaderRow", "params": {"count": stray}})
                duplicates = len(df.columns) - len(set(map(str, df.columns)))
                if duplicates > 0:
                    issues.append({"code": "duplicateHeaders", "params": {"count": duplicates}})
                result["issues"] = issues
            except Exception as e:
                result["issues"] = [{"code": "previewFailed", "params": {"error": str(e)[:120]}}]

        return result

    # ------------------------------------------------------------------

    def convert(self, content: bytes, filename: str, options: Dict[str, Any]) -> List[ConvertedFile]:
        opts = self.resolve_options(options)
        xf = _open_workbook(content)

        selected = opts.get("sheets") or []
        if isinstance(selected, str):
            selected = [selected]
        target_sheets = [s for s in xf.sheet_names if not selected or s in selected]
        if not target_sheets:
            raise ConversionError("Seçilen sayfa çalışma kitabında bulunamadı.")

        delimiter = opts.get("delimiter") or ","
        encoding = opts.get("encoding") or "utf-8"
        has_header = bool(opts.get("has_header", True))
        skip_rows = int(opts.get("skip_rows") or 0)
        clean_headers = bool(opts.get("clean_headers", True))
        skip_empty = bool(opts.get("skip_empty_sheets", True))

        stem = re.sub(r"\.(xlsx|xlsm|xls)$", "", filename or "workbook", flags=re.IGNORECASE)
        stem = _safe_name(stem)
        single = len(target_sheets) == 1

        results: List[ConvertedFile] = []
        used_names: set = set()

        for sheet in target_sheets:
            try:
                df = xf.parse(
                    sheet,
                    header=header_argument(has_header, opts.get("header_rows")),
                    skiprows=skip_rows or None,
                )
            except Exception as e:
                raise ConversionError(f"'{sheet}' sayfası okunamadı: {e}") from e

            if df.empty and skip_empty:
                continue

            if has_header:
                # Çok satırlı başlık MultiIndex üretir; tek satıra indirgenir.
                if clean_headers or isinstance(df.columns, pd.MultiIndex):
                    df.columns = flatten_columns(df.columns)
            else:
                df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]

            csv_text = df.to_csv(index=False, sep=delimiter)
            csv_bytes = csv_text.encode(encoding, errors="replace")

            out_name = f"{stem}{self.target_extension}" if single else f"{stem}__{_safe_name(sheet)}{self.target_extension}"
            if out_name in used_names:
                out_name = f"{stem}__{_safe_name(sheet)}_{len(used_names)}{self.target_extension}"
            used_names.add(out_name)

            results.append(
                ConvertedFile(
                    filename=out_name,
                    content=csv_bytes,
                    media_type=self.target_media_type,
                    meta={
                        "sheet": sheet,
                        "rows": int(len(df)),
                        "columns": int(df.shape[1]),
                        "size": len(csv_bytes),
                    },
                )
            )

        if not results:
            raise ConversionError("Dönüştürülecek veri bulunamadı (tüm sayfalar boş).")

        return results
