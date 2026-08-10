"""
api/core/converters
-------------------
Dosya format dönüştürücü registry'si.

Yeni bir dönüşüm eklemek için `BaseConverter`'dan türeyen bir sınıf yazıp
aşağıdaki `_CONVERTERS` listesine eklemek yeterlidir; API uçları
(`/convert/...`) ve arayüz registry'yi otomatik yansıtır.
"""

from typing import Dict, List, Optional

from api.core.converters.base import (
    BaseConverter,
    ConversionError,
    ConvertedFile,
    ConverterOption,
)
from api.core.converters.csv_normalize import CsvNormalizeConverter
from api.core.converters.excel_to_csv import ExcelToCsvConverter
from api.core.converters.json_to_csv import JsonToCsvConverter

# Kayıtlı dönüştürücüler — sıralama arayüzdeki gösterim sırasıdır.
_CONVERTERS: List[BaseConverter] = [
    ExcelToCsvConverter(),
    CsvNormalizeConverter(),
    JsonToCsvConverter(),
]

CONVERTERS: Dict[str, BaseConverter] = {c.id: c for c in _CONVERTERS}


def list_converters() -> List[dict]:
    """Arayüzün dönüşüm listesini kurması için serileştirilmiş registry."""
    return [c.to_dict() for c in _CONVERTERS]


def get_converter(converter_id: str) -> Optional[BaseConverter]:
    return CONVERTERS.get(converter_id)


__all__ = [
    "BaseConverter",
    "ConversionError",
    "ConvertedFile",
    "ConverterOption",
    "CONVERTERS",
    "list_converters",
    "get_converter",
]
