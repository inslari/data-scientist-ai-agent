"""Çok satırlı başlık (anket export'ları) testleri.

Kapatılan hata: anket/Forms export'larında başlık iki satırdır — üstte soru
metni, altta matris sorusunun alt etiketleri. Tek satır başlık varsayıldığında
alt etiket satırı VERİ olarak kalıyordu; sayı sütunları metne dönüyordu ve
kullanıcı bunu ancak analiz saçmaladığında fark ediyordu.

A) DÜZLEŞTİRME — MultiIndex başlık tek okunabilir ada iner.
B) BİRLEŞTİRİLMİŞ HÜCRE — soru yalnızca ilk sütunda duruyorsa sağa yayılır.
C) UÇTAN UCA — header_rows=2 ile etiket satırı veriden çıkar, sayılar sayı olur.
D) TESPİT — kullanıcı ayarı bilmese bile uyarı çıkar.
E) REGRESYON — tek satırlık normal başlıkta hiçbir şey değişmez.

    uv run python tests/test_multirow_header.py
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TMP_DB = Path(tempfile.gettempdir()) / "ds_agent_test_multirow_header.sqlite"
os.environ["API_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from api.core.converters import get_converter  # noqa: E402
from api.core.converters.headers import (  # noqa: E402
    flatten_columns,
    header_argument,
    looks_like_stray_label_row,
)

# Anket export'u: 1. satır soru metni, 2. satır matris alt etiketleri.
SURVEY_CSV = (
    "Zaman damgası,Yaşınız,Senaryo puanı,Senaryo puanı\n"
    ",,ihtimal1,ihtimal2\n"
    "2026-01-01,34,40,60\n"
    "2026-01-02,51,30,70\n"
    "2026-01-03,29,25,75\n"
).encode("utf-8")

# Birleştirilmiş hücre: soru sadece ilk sütunda yazılı, devamı boş.
MERGED_CSV = (
    "Zaman damgası,Yaşınız,Senaryo puanı,\n"
    ",,ihtimal1,ihtimal2\n"
    "2026-01-01,34,40,60\n"
    "2026-01-02,51,30,70\n"
    "2026-01-03,29,25,75\n"
).encode("utf-8")

PLAIN_CSV = b"ad,tutar\nAyse,10\nBora,20\nCem,30\n"


def _survey_workbook() -> bytes:
    """Aynı yapının Excel hali."""
    rows = [
        ["Zaman damgası", "Yaşınız", "Senaryo puanı", "Senaryo puanı"],
        ["", "", "ihtimal1", "ihtimal2"],
        ["2026-01-01", 34, 40, 60],
        ["2026-01-02", 51, 30, 70],
        ["2026-01-03", 29, 25, 75],
    ]
    buf = io.BytesIO()
    pd.DataFrame(rows[1:], columns=rows[0]).to_excel(buf, index=False, sheet_name="Yanıtlar")
    return buf.getvalue()


class FlattenTest(unittest.TestCase):
    """A) DÜZLEŞTİRME + B) BİRLEŞTİRİLMİŞ HÜCRE"""

    def test_multiindex_becomes_readable_names(self):
        df = pd.read_csv(io.BytesIO(SURVEY_CSV), header=[0, 1])
        self.assertEqual(
            flatten_columns(df.columns),
            ["Zaman damgası", "Yaşınız", "Senaryo puanı - ihtimal1", "Senaryo puanı - ihtimal2"],
        )

    def test_merged_cells_are_filled_rightwards(self):
        """Soru yalnızca ilk sütunda yazılıysa devam sütunlarına yayılır."""
        df = pd.read_csv(io.BytesIO(MERGED_CSV), header=[0, 1])
        self.assertEqual(
            flatten_columns(df.columns)[2:],
            ["Senaryo puanı - ihtimal1", "Senaryo puanı - ihtimal2"],
        )

    def test_duplicate_names_get_a_counter(self):
        self.assertEqual(flatten_columns(pd.Index(["a", "a", "b"])), ["a", "a_2", "b"])

    def test_header_argument_shapes(self):
        self.assertIsNone(header_argument(False, 3))
        self.assertEqual(header_argument(True, 1), 0)
        self.assertEqual(header_argument(True, 2), [0, 1])
        self.assertEqual(header_argument(True, None), 0)   # bozuk değer → tek satır
        self.assertEqual(header_argument(True, 99), [0, 1, 2, 3, 4])  # üst sınır


class CsvEndToEndTest(unittest.TestCase):
    """C) UÇTAN UCA — CSV"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("csv_normalize")

    def test_default_leaves_label_row_in_data(self):
        """Düzeltmeden önceki davranış — hatanın kendisi."""
        df = pd.read_csv(io.BytesIO(self.conv.convert(SURVEY_CSV, "anket.csv", {})[0].content))
        self.assertEqual(str(df["Senaryo puanı"][0]), "ihtimal1")

    def test_header_rows_two_removes_label_row(self):
        out = self.conv.convert(SURVEY_CSV, "anket.csv", {"header_rows": 2})[0]
        df = pd.read_csv(io.BytesIO(out.content))
        self.assertEqual(len(df), 3)
        self.assertIn("Senaryo puanı - ihtimal1", df.columns)
        self.assertNotIn("ihtimal1", df["Senaryo puanı - ihtimal1"].astype(str).tolist())

    def test_numbers_become_numbers(self):
        """Asıl kazanç: etiket satırı gidince sütun sayısal olur."""
        out = self.conv.convert(SURVEY_CSV, "anket.csv", {"header_rows": 2})[0]
        df = pd.read_csv(io.BytesIO(out.content))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Senaryo puanı - ihtimal1"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Yaşınız"]))

    def test_preview_reflects_header_rows(self):
        data = self.conv.inspect(SURVEY_CSV, "anket.csv", {"header_rows": 2})
        self.assertIn("Senaryo puanı - ihtimal1", data["preview"]["columns"])
        self.assertEqual(data["rows"], 3)


class ExcelEndToEndTest(unittest.TestCase):
    """C) UÇTAN UCA — Excel"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("excel_to_csv")
        cls.workbook = _survey_workbook()

    def test_header_rows_two_removes_label_row(self):
        out = self.conv.convert(cls_workbook := self.workbook, "anket.xlsx", {"header_rows": 2})[0]
        df = pd.read_csv(io.BytesIO(out.content))
        self.assertEqual(len(df), 3)
        self.assertIn("Senaryo puanı - ihtimal1", df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Senaryo puanı - ihtimal1"]))
        self.assertIsNotNone(cls_workbook)

    def test_inspect_now_returns_a_preview(self):
        """Excel tarafında önizleme yoktu — kullanıcı sonucu ancak dönüştürünce görüyordu."""
        data = self.conv.inspect(self.workbook, "anket.xlsx")
        self.assertIn("preview", data)
        self.assertEqual(data["preview_sheet"], "Yanıtlar")


class DetectionTest(unittest.TestCase):
    """D) TESPİT — kullanıcı ayarı bilmese bile uyarılmalı"""

    def test_stray_label_row_is_detected(self):
        df = pd.read_csv(io.BytesIO(SURVEY_CSV))
        self.assertGreaterEqual(looks_like_stray_label_row(df), 2)

    def test_clean_file_is_not_flagged(self):
        df = pd.read_csv(io.BytesIO(PLAIN_CSV))
        self.assertEqual(looks_like_stray_label_row(df), 0)

    def test_text_column_alone_is_not_enough(self):
        """Tek sütunda metin/sayı karışımı tesadüf olabilir — uyarı çıkmamalı."""
        content = b"ad,tutar\nnot,10\nBora,20\nCem,30\n"
        self.assertEqual(looks_like_stray_label_row(pd.read_csv(io.BytesIO(content))), 0)

    def test_csv_converter_reports_the_issue(self):
        codes = {i["code"] for i in get_converter("csv_normalize").inspect(SURVEY_CSV, "anket.csv")["issues"]}
        self.assertIn("possibleHeaderRow", codes)

    def test_excel_converter_reports_the_issue(self):
        data = get_converter("excel_to_csv").inspect(_survey_workbook(), "anket.xlsx")
        self.assertIn("possibleHeaderRow", {i["code"] for i in data["issues"]})

    def test_issue_disappears_once_fixed(self):
        codes = {
            i["code"]
            for i in get_converter("csv_normalize").inspect(SURVEY_CSV, "anket.csv", {"header_rows": 2})["issues"]
        }
        self.assertNotIn("possibleHeaderRow", codes)


class RegressionTest(unittest.TestCase):
    """E) REGRESYON — normal dosyalar etkilenmemeli"""

    def test_single_row_header_unchanged(self):
        out = get_converter("csv_normalize").convert(PLAIN_CSV, "duz.csv", {})[0]
        self.assertEqual(out.content.decode("utf-8").splitlines()[0], "ad,tutar")

    def test_no_issues_for_plain_file(self):
        codes = {i["code"] for i in get_converter("csv_normalize").inspect(PLAIN_CSV, "duz.csv")["issues"]}
        self.assertEqual(codes, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
