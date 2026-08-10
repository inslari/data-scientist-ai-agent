"""CSV normalize dönüştürücüsü + ayrıştırma tespiti testleri (DB'siz, LLM'siz).

Kapatılan hata: Türkiye'de Excel'den çıkan CSV `;` ayırıcı, Windows-1254
kodlama ve `1.234,56` sayı biçimi kullanır; `pd.read_csv` varsayılanları bu
dosyayı ya tek sütuna çökertir ya da UnicodeDecodeError ile reddeder.

A) TESPİT — kodlama / ayırıcı / sayı biçimi doğru bulunuyor mu.
B) INSPECT — önizleme gerçek sütunları gösteriyor, bulgular raporlanıyor.
C) CONVERT — çıktı kanonik (UTF-8, virgül, ondalık nokta) ve sayılar sayı.
D) KULLANICI AYARI — elle verilen ayar tespiti ezer.
E) INGEST FALLBACK — daha önce reddedilen dosya artık profillenebiliyor,
   çalışan dosyaların davranışı ise DEĞİŞMİYOR (regresyon sınırı).

    uv run python tests/test_csv_normalize.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_TMP_DB = Path(tempfile.gettempdir()) / "ds_agent_test_csv_normalize.sqlite"
os.environ["API_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from api.core.converters import get_converter  # noqa: E402
from api.core.converters.csv_dialect import (  # noqa: E402
    detect_delimiter,
    detect_dialect,
    detect_encoding,
)
from api.core.csv_ingest import detect_csv_warnings, profile_csv_bytes  # noqa: E402

# Türk Excel'inden çıkmış tipik dosya: ; ayırıcı, cp1254, ondalık virgül,
# binlik nokta, Türkçe karakterli başlıklar.
TR_CSV = (
    "Müşteri;Şehir;Tutar;Adet\n"
    "Ayşe Yılmaz;İstanbul;1.234,56;3\n"
    "Böri Çelik;Şanlıurfa;987,40;1\n"
    "Cem Öztürk;İzmir;12.000,00;7\n"
).encode("cp1254")

# Sorunsuz, klasik dosya — davranışı değişmemeli.
PLAIN_CSV = b"name,amount\nalice,10.5\nbob,20\n"


class DialectDetectionTest(unittest.TestCase):
    """A) TESPİT"""

    def test_detects_cp1254_encoding(self):
        self.assertIn(detect_encoding(TR_CSV), ("cp1254", "iso-8859-9", "windows-1254"))

    def test_utf8_stays_utf8(self):
        self.assertEqual(detect_encoding(PLAIN_CSV), "utf-8")

    def test_detects_semicolon_delimiter(self):
        text = TR_CSV.decode("cp1254")
        self.assertEqual(detect_delimiter(text), ";")

    def test_detects_comma_delimiter(self):
        self.assertEqual(detect_delimiter(PLAIN_CSV.decode()), ",")

    def test_detects_turkish_number_format(self):
        dialect = detect_dialect(TR_CSV)
        self.assertEqual(dialect["decimal"], ",")
        self.assertEqual(dialect["thousands"], ".")

    def test_plain_file_detected_as_dot_decimal(self):
        dialect = detect_dialect(PLAIN_CSV)
        self.assertEqual(dialect["decimal"], ".")
        self.assertIsNone(dialect["thousands"])

    def test_single_column_file_is_not_split(self):
        """Gerçekten tek sütunlu dosya uydurma bir ayırıcıya bölünmemeli."""
        content = "notlar\nbir iki üç\ndört beş\n".encode()
        self.assertEqual(detect_dialect(content)["delimiter"], ",")


class CsvNormalizeInspectTest(unittest.TestCase):
    """B) INSPECT"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("csv_normalize")
        assert cls.conv is not None

    def test_preview_shows_real_columns(self):
        data = self.conv.inspect(TR_CSV, "satislar.csv")
        self.assertEqual(data["preview"]["columns"], ["Müşteri", "Şehir", "Tutar", "Adet"])
        self.assertEqual(data["rows"], 3)
        self.assertEqual(data["columns"], 4)
        self.assertEqual(data["preview"]["rows"][0][0], "Ayşe Yılmaz")

    def test_used_settings_are_reported(self):
        used = self.conv.inspect(TR_CSV, "satislar.csv")["detected"]
        self.assertEqual(used["delimiter"], ";")
        self.assertEqual(used["decimal"], ",")
        self.assertIn(used["encoding"], ("cp1254", "iso-8859-9", "windows-1254"))

    def test_issues_flag_encoding_and_decimal(self):
        codes = {i["code"] for i in self.conv.inspect(TR_CSV, "satislar.csv")["issues"]}
        self.assertIn("nonUtf8", codes)
        self.assertIn("commaDecimal", codes)
        self.assertNotIn("singleColumn", codes)

    def test_clean_file_has_no_issues(self):
        codes = {i["code"] for i in self.conv.inspect(PLAIN_CSV, "plain.csv")["issues"]}
        self.assertEqual(codes, set())


class CsvNormalizeConvertTest(unittest.TestCase):
    """C) CONVERT + D) KULLANICI AYARI"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("csv_normalize")

    def test_output_is_canonical_utf8_comma(self):
        out = self.conv.convert(TR_CSV, "satislar.csv", {})[0]
        self.assertEqual(out.filename, "satislar_normalized.csv")

        text = out.content.decode("utf-8")  # kanonik çıktı: UTF-8
        lines = text.splitlines()
        self.assertEqual(lines[0], "Müşteri,Şehir,Tutar,Adet")
        self.assertEqual(lines[1], "Ayşe Yılmaz,İstanbul,1234.56,3")

    def test_numbers_are_parsed_as_numbers(self):
        import io

        out = self.conv.convert(TR_CSV, "satislar.csv", {})[0]
        df = pd.read_csv(io.BytesIO(out.content))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Tutar"]))
        self.assertAlmostEqual(df["Tutar"].sum(), 1234.56 + 987.40 + 12000.00, places=2)

    def test_output_delimiter_option(self):
        out = self.conv.convert(TR_CSV, "satislar.csv", {"output_delimiter": ";"})[0]
        self.assertEqual(out.content.decode("utf-8").splitlines()[0], "Müşteri;Şehir;Tutar;Adet")

    def test_explicit_option_overrides_detection(self):
        """Kullanıcı ayırıcıyı virgül derse tespit ezilir — dosya tek sütuna düşer."""
        out = self.conv.convert(TR_CSV, "satislar.csv", {"delimiter": ",", "encoding": "cp1254"})[0]
        header = out.content.decode("utf-8").splitlines()[0]
        self.assertNotIn(",", header.replace("Müşteri;Şehir;Tutar;Adet", ""))

    def test_empty_columns_are_dropped(self):
        content = b"a,b,Unnamed: 2\n1,2,\n3,4,\n"
        out = self.conv.convert(content, "excel_export.csv", {})[0]
        self.assertEqual(out.content.decode("utf-8").splitlines()[0], "a,b")

    def test_headers_are_trimmed(self):
        content = "  ad  ,  tutar \nx,1\n".encode()
        out = self.conv.convert(content, "bosluklu.csv", {})[0]
        self.assertEqual(out.content.decode("utf-8").splitlines()[0], "ad,tutar")

    def test_cell_values_are_trimmed(self):
        """pandas 3'te metin sütunu StringDtype — `dtype == object` kontrolü kırpmayı susturuyordu."""
        content = 'ad,sehir\n"  Ayşe  ","İstanbul  "\n'.encode()
        out = self.conv.convert(content, "bosluklu.csv", {})[0]
        self.assertEqual(out.content.decode("utf-8").splitlines()[1], "Ayşe,İstanbul")


class IngestFallbackTest(unittest.TestCase):
    """E) INGEST FALLBACK — yükleme hattının davranış sınırı"""

    def test_turkish_csv_no_longer_rejected(self):
        """Önce UnicodeDecodeError ile reddediliyordu; artık profillenebiliyor."""
        row_count, schema, stats, _ = profile_csv_bytes(TR_CSV)
        self.assertEqual(row_count, 3)
        self.assertEqual([c["name"] for c in schema], ["Müşteri", "Şehir", "Tutar", "Adet"])
        self.assertTrue(any(s["column_name"] == "Tutar" for s in stats))

    def test_plain_csv_behaviour_unchanged(self):
        row_count, schema, _, warnings = profile_csv_bytes(PLAIN_CSV)
        self.assertEqual(warnings, [])
        self.assertEqual(row_count, 2)
        self.assertEqual([c["name"] for c in schema], ["name", "amount"])

    def test_utf8_semicolon_file_is_left_alone(self):
        """
        UTF-8 + `;` dosya varsayılanlarla HATASIZ okunur (tek sütun olarak).
        Fallback devreye girmez — sessizce yeniden bölmek veriyi bozabilirdi;
        düzeltme kullanıcının onayıyla dönüştürücüde yapılır.
        """
        content = "ad;tutar\nx;1\ny;2\n".encode()
        _, schema, _, warnings = profile_csv_bytes(content)
        self.assertEqual([c["name"] for c in schema], ["ad;tutar"])
        # Veri değişmiyor ama sessiz de kalmıyoruz — kullanıcı uyarılıyor.
        self.assertEqual([w["code"] for w in warnings], ["singleColumn"])
        self.assertEqual(warnings[0]["params"]["delimiter"], ";")

        # Dönüştürücü aynı dosyayı doğru şekilde ayırır.
        out = get_converter("csv_normalize").convert(content, "x.csv", {})[0]
        self.assertEqual(out.content.decode("utf-8").splitlines()[0], "ad,tutar")


class UploadWarningTest(unittest.TestCase):
    """
    F) YÜKLEME UYARILARI — dosya kabul edildi ama yanlış okunmuş olabilir.

    Bu kanalın tek işi kullanıcıyı Dosya Dönüştürücü'ye yönlendirmek; veriye
    dokunmaz. Yanlış pozitif maliyeti yüksek (her yüklemede boş yere uyarı),
    o yüzden temiz dosyalarda kesinlikle susmalı.
    """

    def _codes(self, content: bytes):
        _, _, _, warnings = profile_csv_bytes(content)
        return {w["code"] for w in warnings}

    def test_clean_file_produces_no_warning(self):
        self.assertEqual(self._codes(PLAIN_CSV), set())

    def test_semicolon_file_warns_about_single_column(self):
        self.assertIn("singleColumn", self._codes("ad;tutar\nx;1\ny;2\n".encode()))

    def test_non_utf8_file_warns_about_encoding(self):
        self.assertIn("nonUtf8", self._codes(TR_CSV))

    def test_numbers_left_as_text_are_flagged(self):
        """`;` değil `,` ayırıcılı ama TR sayı biçimli dosya: sütun metin kalır."""
        content = 'ad,tutar\n"x","1.234,56"\n"y","987,40"\n"z","12.000,00"\n'.encode()
        codes = self._codes(content)
        self.assertIn("numbersAsText", codes)

    def test_genuinely_single_column_file_is_not_flagged(self):
        """Tek sütunlu meşru dosya uyarı üretmemeli (yanlış pozitif sınırı)."""
        content = "notlar\nbir iki üç\ndört beş\naltı yedi\n".encode()
        self.assertEqual(self._codes(content), set())

    def test_warning_params_carry_context(self):
        _, _, _, warnings = profile_csv_bytes(TR_CSV)
        encoding_warning = next(w for w in warnings if w["code"] == "nonUtf8")
        self.assertIn(encoding_warning["params"]["encoding"], ("cp1254", "iso-8859-9"))

    def test_detect_csv_warnings_is_reusable_standalone(self):
        import io as _io

        df = pd.read_csv(_io.BytesIO(PLAIN_CSV))
        self.assertEqual(detect_csv_warnings(PLAIN_CSV, df), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
