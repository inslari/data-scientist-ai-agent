"""
api/core/converters/base.py
---------------------------
Dosya format dönüştürücüleri için ortak sözleşme.

Her dönüştürücü `BaseConverter`'dan türer ve registry'ye (`__init__.py`)
kaydedilir. Yeni bir dönüşüm eklemek için tek yapılması gereken:

    1. Bu sınıftan türeyen bir converter yaz (`inspect` + `convert`),
    2. `api/core/converters/__init__.py` içindeki `CONVERTERS` listesine ekle.

API ve arayüz tarafında hiçbir değişiklik gerekmez: `/convert/converters`
uçtan uca registry'yi yansıtır, arayüz de `options` şemasını jenerik olarak
render eder.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConvertedFile:
    """Dönüşüm sonucunda üretilen tek bir çıktı dosyası (bellek içi)."""

    filename: str
    content: bytes
    media_type: str = "text/csv"
    # Arayüzde gösterilecek serbest biçimli bilgi (satır/sütun sayısı, sayfa adı…)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConverterOption:
    """
    Dönüştürücünün kullanıcıdan aldığı tek bir ayar.

    Arayüz bu şemayı jenerik olarak render eder, dolayısıyla yeni bir ayar
    eklemek için frontend'e dokunmak gerekmez.

    type:
        "boolean"     — anahtar (checkbox)
        "text"        — serbest metin
        "number"      — sayı
        "select"      — `choices` içinden tek seçim
        "multiselect" — çoklu seçim; `source` verilirse seçenekler
                        `/convert/inspect` çıktısından okunur (örn. "sheets")
    """

    name: str
    type: str
    default: Any = None
    # i18n anahtarları — arayüz `converter.options.<label_key>` ile çevirir,
    # çeviri yoksa `name` gösterilir.
    label_key: Optional[str] = None
    help_key: Optional[str] = None
    choices: Optional[List[Dict[str, Any]]] = None  # [{"value": ..., "label_key": ...}]
    source: Optional[str] = None  # inspect çıktısındaki liste alanının adı
    # Varsayılanın inspect çıktısından geldiği noktalı yol (örn. "detected.delimiter").
    # Kullanıcı elle değiştirene kadar tespit edilen değer kullanılır.
    default_source: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "label_key": self.label_key or self.name,
            "help_key": self.help_key,
            "choices": self.choices,
            "source": self.source,
            "default_source": self.default_source,
            "min": self.min,
            "max": self.max,
        }


class ConversionError(Exception):
    """Kullanıcıya gösterilebilir dönüşüm hatası (500 değil, 400 üretir)."""


class BaseConverter:
    """Tüm dönüştürücülerin ortak arayüzü."""

    # Registry kimliği — API/arayüz bu id ile konuşur.
    id: str = ""
    # Kabul edilen kaynak uzantıları (küçük harf, nokta dahil).
    source_extensions: tuple = ()
    # Üretilen çıktının uzantısı ve MIME tipi.
    target_extension: str = ".csv"
    target_media_type: str = "text/csv"
    # i18n anahtarları (arayüz `converter.formats.<id>.label` şeklinde çevirir).
    label_key: str = ""
    description_key: str = ""
    # Tek kaynak dosyadan birden çok çıktı üretebilir mi (örn. her sayfa bir CSV)?
    produces_multiple: bool = False
    # Dosya boyutu üst sınırı (MB).
    max_file_mb: int = 100
    # True ise arayüz, ayar her değiştiğinde `inspect`'i yeniden çağırır ve
    # dönen `preview`'u gösterir (ayırıcı/kodlama denemesi canlı görünsün diye).
    live_preview: bool = False

    #: Kullanıcı ayarları
    options: List[ConverterOption] = []

    # ------------------------------------------------------------------
    # Alt sınıfların uygulayacağı kısım
    # ------------------------------------------------------------------

    def inspect(
        self, content: bytes, filename: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dönüşüm öncesi keşif: kaynak dosyanın yapısını döndürür.

        Dönen sözlüğün anahtarları `ConverterOption.source` ile eşleşirse
        arayüz o ayarın seçeneklerini buradan doldurur (örn. `{"sheets": [...]}`);
        `default_source` yolları da bu çıktıdan okunur (örn. `detected.delimiter`).

        `live_preview` açıksa `options` kullanıcının o anki ayarlarıyla gelir ve
        çıktıya `preview: {"columns": [...], "rows": [[...]]}` konması beklenir.
        """
        return {}

    def convert(self, content: bytes, filename: str, options: Dict[str, Any]) -> List[ConvertedFile]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Ortak yardımcılar
    # ------------------------------------------------------------------

    def accepts(self, filename: str) -> bool:
        lower = (filename or "").lower()
        return any(lower.endswith(ext) for ext in self.source_extensions)

    def resolve_options(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Eksik ayarları varsayılanlarla tamamlar."""
        resolved = {opt.name: opt.default for opt in self.options}
        for key, value in (options or {}).items():
            if key in resolved and value is not None:
                resolved[key] = value
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label_key": self.label_key or self.id,
            "description_key": self.description_key,
            "source_extensions": list(self.source_extensions),
            "target_extension": self.target_extension,
            "produces_multiple": self.produces_multiple,
            "max_file_mb": self.max_file_mb,
            "live_preview": self.live_preview,
            # CSV üreten dönüştürücüler sonucu doğrudan Veri Varlıkları'na yazabilir.
            "can_save_to_assets": self.target_extension == ".csv",
            "options": [opt.to_dict() for opt in self.options],
        }
