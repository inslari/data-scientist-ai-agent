"""
api/core/converters/json_to_csv.py
----------------------------------
JSON / JSONL → CSV.

Zor kısım dönüşüm değil, **hangi kısmın satır olduğuna karar vermek**: API
çıktıları genelde `{"meta": {...}, "data": {"items": [ ... ]}}` gibi sarmalanmış
gelir; tabloya dönüşecek olan içerideki liste. `inspect` bu adayları bulup
kullanıcıya sunar (Excel'deki sayfa seçimiyle aynı kalıp), seçilen liste
`json_normalize` ile düzleştirilir.

Desteklenen kaynak biçimleri:
    - JSON  : tek belge (dizi ya da sarmalanmış dizi)
    - JSONL : satır başına bir JSON nesnesi (.jsonl / .ndjson)
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from api.core.converters.base import (
    BaseConverter,
    ConversionError,
    ConvertedFile,
    ConverterOption,
)
from api.core.converters.csv_dialect import detect_encoding
from api.core.converters.headers import build_preview

# Kök seviyeyi ifade eden yol (JSONPath'teki `$` ile aynı fikir)
ROOT_PATH = "$"

_PREVIEW_ROWS = 8
_MAX_DEPTH = 4          # aday ararken inilecek en fazla iç içe seviye
_SCAN_RECORDS = 200     # kolon birleşimi/aykırılık taraması için örneklem


def _decode(content: bytes) -> str:
    """JSON spec UTF-8 der, gerçek dosyalar demez — kodlamayı tespit ederek çöz."""
    encoding = detect_encoding(content)
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _parse(content: bytes, filename: str) -> Tuple[str, Any]:
    """
    (biçim, veri) döndürür. Biçim: "json" | "jsonl".

    Uzantıya körü körüne güvenmeyiz: `.json` uzantılı JSONL dosyaları yaygın.
    """
    text = _decode(content).strip()
    if not text:
        raise ConversionError("Dosya boş.")

    lower = (filename or "").lower()
    prefer_lines = lower.endswith((".jsonl", ".ndjson"))

    if not prefer_lines:
        try:
            return "json", json.loads(text)
        except json.JSONDecodeError:
            pass  # JSONL olabilir

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ConversionError(
                f"Geçersiz JSON — satır {line_no}: {e.msg}"
            ) from e

    if not records:
        raise ConversionError("Dosyada ayrıştırılabilir JSON bulunamadı.")
    return "jsonl", records


def _is_record_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict)


def _is_record_map(value: Any) -> bool:
    """
    `{"id1": {...}, "id2": {...}}` — anahtarla indekslenmiş kayıt sözlüğü.

    "Değerlerin hepsi sözlük" yetmez: `{"meta": {...}, "data": {...}}` da bunu
    sağlar ama kayıt listesi değildir. Kayıt sözlüğü olması için değerlerin
    BENZER ŞEKİLLİ olması gerekir — ortak alanlar üzerinden bakıyoruz.
    """
    if not isinstance(value, dict) or len(value) < 2:
        return False
    values = list(value.values())
    if not all(isinstance(v, dict) for v in values):
        return False

    key_sets = [frozenset(v.keys()) for v in values[:_SCAN_RECORDS]]
    shared = frozenset.intersection(*key_sets)
    if not shared:
        return False
    union = frozenset().union(*key_sets)
    return len(shared) / len(union) >= 0.5


def _records_at(data: Any, path: str) -> List[dict]:
    """Verilen yoldaki kayıt listesini çıkarır."""
    node = data
    if path and path != ROOT_PATH:
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                raise ConversionError(f"'{path}' yolu dosyada bulunamadı.")
            node = node[key]

    if _is_record_list(node):
        return node
    if _is_record_map(node):
        return list(node.values())
    if isinstance(node, dict):
        return [node]  # tek nesne → tek satır
    if isinstance(node, list):
        return [{"value": item} for item in node]  # skaler dizi → tek kolon
    raise ConversionError(f"'{path}' yolu tabloya dönüştürülebilir veri içermiyor.")


def _discover_paths(data: Any, prefix: str = "", depth: int = 0) -> List[str]:
    """Tabloya dönüşebilecek yolları (kayıt listeleri) bulur, en dış önce."""
    found: List[str] = []
    if _is_record_list(data) or _is_record_map(data):
        found.append(prefix or ROOT_PATH)
    if isinstance(data, dict) and depth < _MAX_DEPTH:
        for key, value in data.items():
            found.extend(_discover_paths(value, f"{prefix}.{key}" if prefix else key, depth + 1))
    return found


def _flatten(records: List[dict], opts: Dict[str, Any]) -> pd.DataFrame:
    separator = opts.get("separator") or "_"
    max_level = opts.get("max_level")
    max_level = int(max_level) if max_level not in (None, "", 0) else None

    if opts.get("flatten", True):
        df = pd.json_normalize(records, sep=separator, max_level=max_level)
    else:
        df = pd.DataFrame(records)

    # Kalan liste/sözlük hücreleri: CSV'de Python repr'i ("[1, 2]") istemeyiz.
    mode = opts.get("list_handling") or "json"
    for col in df.columns:
        if not df[col].map(lambda v: isinstance(v, (list, dict))).any():
            continue
        if mode == "drop":
            df = df.drop(columns=[col])
        elif mode == "join":
            df[col] = df[col].map(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else
                (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
            )
        else:  # "json"
            df[col] = df[col].map(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            )

    if opts.get("clean_headers", True):
        df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    return df


def _collect_issues(records: List[dict], df: pd.DataFrame, opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    separator = opts.get("separator") or "_"

    if len(records) == 1:
        issues.append({"code": "singleRecord", "params": {}})

    nested = [c for c in df.columns if separator in str(c)]
    if nested and opts.get("flatten", True):
        issues.append({
            "code": "nestedFlattened",
            "params": {"count": len(nested), "columns": ", ".join(map(str, nested[:3]))},
        })

    # Kayıtlar arası alan farkı — CSV'de eksik alanlar boş hücre olur.
    sample = records[:_SCAN_RECORDS]
    if len(sample) > 1:
        first_keys = set(sample[0].keys()) if isinstance(sample[0], dict) else set()
        all_keys = set()
        for record in sample:
            if isinstance(record, dict):
                all_keys |= set(record.keys())
        if all_keys - first_keys:
            issues.append({"code": "raggedRecords", "params": {"count": len(all_keys - first_keys)}})

    return issues


class JsonToCsvConverter(BaseConverter):
    id = "json_to_csv"
    source_extensions = (".json", ".jsonl", ".ndjson")
    target_extension = ".csv"
    target_media_type = "text/csv"
    label_key = "json_to_csv"
    description_key = "json_to_csv"
    produces_multiple = False
    max_file_mb = 100
    live_preview = True

    options = [
        ConverterOption(
            name="record_path",
            type="select",
            default=ROOT_PATH,
            source="record_paths",              # seçenekler dosyadan gelir
            default_source="detected.record_path",
            label_key="recordPath",
            help_key="recordPath",
        ),
        ConverterOption(
            name="flatten",
            type="boolean",
            default=True,
            label_key="flatten",
            help_key="flatten",
        ),
        ConverterOption(
            name="separator",
            type="select",
            default="_",
            label_key="separator",
            choices=[
                {"value": "_", "label_key": "underscore"},
                {"value": ".", "label_key": "dot"},
            ],
        ),
        ConverterOption(
            name="max_level",
            type="number",
            default=0,
            min=0,
            max=10,
            label_key="maxLevel",
            help_key="maxLevel",
        ),
        ConverterOption(
            name="list_handling",
            type="select",
            default="json",
            label_key="listHandling",
            help_key="listHandling",
            choices=[
                {"value": "json", "label_key": "asJson"},
                {"value": "join", "label_key": "joined"},
                {"value": "drop", "label_key": "dropped"},
            ],
        ),
        ConverterOption(
            name="output_delimiter",
            type="select",
            default=",",
            label_key="outputDelimiter",
            help_key="outputDelimiter",
            choices=[
                {"value": ",", "label_key": "comma"},
                {"value": ";", "label_key": "semicolon"},
                {"value": "\t", "label_key": "tab"},
                {"value": "|", "label_key": "pipe"},
            ],
        ),
        ConverterOption(
            name="clean_headers",
            type="boolean",
            default=True,
            label_key="cleanHeaders",
            help_key="cleanHeaders",
        ),
    ]

    # ------------------------------------------------------------------

    def _resolve_path(self, data: Any, options: Optional[Dict[str, Any]]) -> Tuple[List[str], str]:
        """
        Aday yolları bulur ve kullanılacak yolu seçer.

        Öncelik: kullanıcının seçimi > EN ÇOK SATIR veren aday. "En dıştaki"
        değil, çünkü bir belgede hem `errors: [1 kayıt]` hem `data.items:
        [500 kayıt]` olabilir — kullanıcının istediği neredeyse her zaman büyük
        olandır. Eşitlikte daha sığ olan kazanır.
        """
        paths = _discover_paths(data) or [ROOT_PATH]
        requested = (options or {}).get("record_path")
        if requested in paths:
            return paths, requested

        def score(path: str) -> Tuple[int, int]:
            try:
                rows = len(_records_at(data, path))
            except ConversionError:
                rows = 0
            return rows, -path.count(".")

        return paths, max(paths, key=score)

    def inspect(
        self, content: bytes, filename: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        source_format, data = _parse(content, filename)
        paths, path = self._resolve_path(data, options)
        opts = self.resolve_options(options)

        path_infos = []
        for candidate in paths:
            try:
                records = _records_at(data, candidate)
            except ConversionError:
                continue
            columns = set()
            for record in records[:_SCAN_RECORDS]:
                if isinstance(record, dict):
                    columns |= set(record.keys())
            path_infos.append({
                "name": candidate,
                "rows": len(records),
                "columns": len(columns),
            })

        records = _records_at(data, path)
        df = _flatten(records, opts)

        return {
            "record_paths": path_infos,
            "detected": {"format": source_format, "record_path": path},
            "preview": build_preview(df, _PREVIEW_ROWS),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "issues": _collect_issues(records, df, opts),
        }

    # ------------------------------------------------------------------

    def convert(self, content: bytes, filename: str, options: Dict[str, Any]) -> List[ConvertedFile]:
        _, data = _parse(content, filename)
        _, path = self._resolve_path(data, options)
        opts = self.resolve_options(options)

        df = _flatten(_records_at(data, path), opts)
        if df.empty:
            raise ConversionError("Dönüştürülecek kayıt bulunamadı.")

        csv_bytes = df.to_csv(
            index=False, sep=opts.get("output_delimiter") or ","
        ).encode("utf-8", errors="replace")

        stem = re.sub(r"\.(json|jsonl|ndjson)$", "", filename or "data", flags=re.IGNORECASE)
        stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip("._") or "data"

        return [
            ConvertedFile(
                filename=f"{stem}{self.target_extension}",
                content=csv_bytes,
                media_type=self.target_media_type,
                meta={
                    "rows": int(len(df)),
                    "columns": int(df.shape[1]),
                    "size": len(csv_bytes),
                    "record_path": path,
                },
            )
        ]
