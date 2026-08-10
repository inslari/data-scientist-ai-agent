"""JSON / JSONL → CSV dönüştürücüsü testleri (DB'siz, LLM'siz).

Bu dönüşümde asıl risk ayrıştırma değil, **hangi kısmın satır olduğuna karar
vermek**: API çıktısı `{"meta": ..., "data": {"items": [...]}}` gibi sarmalanmış
gelir; yanlış yol seçilirse kullanıcı tek satırlık anlamsız bir tablo alır.

A) AYRIŞTIRMA — JSON / JSONL, uzantıya körü körüne güvenmeden.
B) YOL KEŞFİ — iç içe listeler ve anahtarla indekslenmiş kayıt sözlükleri.
C) DÜZLEŞTİRME — iç içe alanlar, listeler, derinlik sınırı.
D) ÇIKTI — CSV doğru ve kanonik.
E) BULGULAR — kullanıcıyı uyaran issue kodları.

    uv run python tests/test_json_to_csv.py
"""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TMP_DB = Path(tempfile.gettempdir()) / "ds_agent_test_json_to_csv.sqlite"
os.environ["API_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from api.core.converters import ConversionError, get_converter  # noqa: E402
from api.core.converters.json_to_csv import ROOT_PATH  # noqa: E402

# Sarmalanmış API çıktısı — satırlar `data.items` altında.
WRAPPED = json.dumps({
    "meta": {"page": 1, "total": 3},
    "data": {
        "items": [
            {"id": 1, "ad": "Ayşe", "adres": {"il": "İstanbul", "posta": "34000"}, "etiketler": ["vip", "yeni"]},
            {"id": 2, "ad": "Bora", "adres": {"il": "İzmir", "posta": "35000"}, "etiketler": []},
            {"id": 3, "ad": "Cem", "adres": {"il": "Ankara", "posta": "06000"}, "etiketler": ["vip"]},
        ]
    },
}, ensure_ascii=False).encode("utf-8")

# Kök seviyede dizi
ROOT_ARRAY = json.dumps(
    [{"id": 1, "tutar": 10.5}, {"id": 2, "tutar": 20}], ensure_ascii=False
).encode("utf-8")

# Satır başına bir nesne
JSONL = (
    '{"id": 1, "ad": "Ayşe"}\n'
    '{"id": 2, "ad": "Bora"}\n'
    '\n'
    '{"id": 3, "ad": "Cem"}\n'
).encode("utf-8")

# Anahtarla indekslenmiş kayıt sözlüğü (Firebase/dump kalıbı)
RECORD_MAP = json.dumps({
    "u1": {"ad": "Ayşe", "puan": 10},
    "u2": {"ad": "Bora", "puan": 20},
}, ensure_ascii=False).encode("utf-8")


def _read(out_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(out_bytes))


class ParsingTest(unittest.TestCase):
    """A) AYRIŞTIRMA"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("json_to_csv")
        assert cls.conv is not None

    def test_detects_plain_json(self):
        self.assertEqual(self.conv.inspect(ROOT_ARRAY, "a.json")["detected"]["format"], "json")

    def test_detects_jsonl(self):
        self.assertEqual(self.conv.inspect(JSONL, "a.jsonl")["detected"]["format"], "jsonl")

    def test_jsonl_content_with_json_extension_still_works(self):
        """`.json` uzantılı JSONL yaygın — uzantıya değil içeriğe bakılmalı."""
        data = self.conv.inspect(JSONL, "aslinda_jsonl.json")
        self.assertEqual(data["detected"]["format"], "jsonl")
        self.assertEqual(data["rows"], 3)

    def test_blank_lines_are_skipped(self):
        self.assertEqual(self.conv.inspect(JSONL, "a.jsonl")["rows"], 3)

    def test_broken_json_reports_line_number(self):
        content = b'{"id": 1}\n{"id": 2\n'
        with self.assertRaises(ConversionError) as ctx:
            self.conv.convert(content, "bozuk.jsonl", {})
        self.assertIn("satır 2", str(ctx.exception))

    def test_empty_file_is_rejected(self):
        with self.assertRaises(ConversionError):
            self.conv.convert(b"   ", "bos.json", {})


class RecordPathTest(unittest.TestCase):
    """B) YOL KEŞFİ"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("json_to_csv")

    def test_nested_list_is_discovered(self):
        data = self.conv.inspect(WRAPPED, "api.json")
        names = [p["name"] for p in data["record_paths"]]
        self.assertIn("data.items", names)
        self.assertEqual(data["detected"]["record_path"], "data.items")

    def test_discovered_path_reports_shape(self):
        info = next(
            p for p in self.conv.inspect(WRAPPED, "api.json")["record_paths"]
            if p["name"] == "data.items"
        )
        self.assertEqual(info["rows"], 3)
        self.assertEqual(info["columns"], 4)

    def test_root_array_uses_root_path(self):
        self.assertEqual(
            self.conv.inspect(ROOT_ARRAY, "a.json")["detected"]["record_path"], ROOT_PATH
        )

    def test_record_map_becomes_rows(self):
        """`{"u1": {...}, "u2": {...}}` → iki satır."""
        data = self.conv.inspect(RECORD_MAP, "users.json")
        self.assertEqual(data["rows"], 2)
        self.assertIn("ad", data["preview"]["columns"])

    def test_user_selected_path_wins(self):
        out = self.conv.convert(WRAPPED, "api.json", {"record_path": "data.items"})[0]
        self.assertEqual(len(_read(out.content)), 3)

    def test_unknown_path_falls_back_to_best_candidate(self):
        """Seçilen yol dosyada yoksa sessizce çökmek yerine en iyi adaya döner."""
        out = self.conv.convert(WRAPPED, "api.json", {"record_path": "yok.olmayan"})[0]
        self.assertEqual(len(_read(out.content)), 3)

    def test_largest_list_wins_over_shallower_one(self):
        """
        Belgede hem küçük hem büyük liste varsa varsayılan BÜYÜK olandır.
        "En dıştaki" kuralı burada yanlış cevabı verirdi (errors, 1 satır).
        """
        content = json.dumps({
            "errors": [{"kod": "E1"}],
            "data": {"items": [{"id": i} for i in range(50)]},
        }).encode()
        data = self.conv.inspect(content, "api.json")
        self.assertEqual(data["detected"]["record_path"], "data.items")
        self.assertEqual(data["rows"], 50)

    def test_wrapper_dict_is_not_mistaken_for_records(self):
        """`{"meta": {...}, "data": {...}}` kayıt sözlüğü DEĞİLDİR (şekilleri farklı)."""
        paths = [p["name"] for p in self.conv.inspect(WRAPPED, "api.json")["record_paths"]]
        self.assertNotIn(ROOT_PATH, paths)


class FlattenTest(unittest.TestCase):
    """C) DÜZLEŞTİRME"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("json_to_csv")

    def test_nested_objects_are_flattened(self):
        out = self.conv.convert(WRAPPED, "api.json", {})[0]
        df = _read(out.content)
        self.assertIn("adres_il", df.columns)
        self.assertEqual(df["adres_il"].tolist(), ["İstanbul", "İzmir", "Ankara"])

    def test_separator_option(self):
        out = self.conv.convert(WRAPPED, "api.json", {"separator": "."})[0]
        self.assertIn("adres.il", _read(out.content).columns)

    def test_flatten_off_keeps_nested_as_json(self):
        out = self.conv.convert(WRAPPED, "api.json", {"flatten": False})[0]
        df = _read(out.content)
        self.assertIn("adres", df.columns)
        self.assertNotIn("adres_il", df.columns)
        self.assertEqual(json.loads(df["adres"][0])["il"], "İstanbul")

    def test_lists_serialised_as_json_by_default(self):
        df = _read(self.conv.convert(WRAPPED, "api.json", {})[0].content)
        self.assertEqual(json.loads(df["etiketler"][0]), ["vip", "yeni"])

    def test_lists_can_be_joined(self):
        df = _read(self.conv.convert(WRAPPED, "api.json", {"list_handling": "join"})[0].content)
        self.assertEqual(df["etiketler"][0], "vip, yeni")

    def test_lists_can_be_dropped(self):
        df = _read(self.conv.convert(WRAPPED, "api.json", {"list_handling": "drop"})[0].content)
        self.assertNotIn("etiketler", df.columns)

    def test_max_level_limits_depth(self):
        content = json.dumps([{"a": {"b": {"c": 1}}}]).encode()
        df = _read(self.conv.convert(content, "derin.json", {"max_level": 1})[0].content)
        self.assertIn("a_b", df.columns)
        self.assertNotIn("a_b_c", df.columns)


class OutputTest(unittest.TestCase):
    """D) ÇIKTI"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("json_to_csv")

    def test_filename_and_utf8_output(self):
        out = self.conv.convert(WRAPPED, "musteriler.json", {})[0]
        self.assertEqual(out.filename, "musteriler.csv")
        self.assertIn("Ayşe", out.content.decode("utf-8"))

    def test_meta_reports_shape_and_path(self):
        out = self.conv.convert(WRAPPED, "api.json", {})[0]
        self.assertEqual(out.meta["rows"], 3)
        self.assertEqual(out.meta["record_path"], "data.items")

    def test_output_delimiter_option(self):
        out = self.conv.convert(ROOT_ARRAY, "a.json", {"output_delimiter": ";"})[0]
        self.assertEqual(out.content.decode("utf-8").splitlines()[0], "id;tutar")

    def test_jsonl_roundtrip(self):
        df = _read(self.conv.convert(JSONL, "a.jsonl", {})[0].content)
        self.assertEqual(df["ad"].tolist(), ["Ayşe", "Bora", "Cem"])


class IssuesTest(unittest.TestCase):
    """E) BULGULAR"""

    @classmethod
    def setUpClass(cls):
        cls.conv = get_converter("json_to_csv")

    def _codes(self, content: bytes, filename: str, options=None):
        return {i["code"] for i in self.conv.inspect(content, filename, options or {})["issues"]}

    def test_flattening_is_reported(self):
        self.assertIn("nestedFlattened", self._codes(WRAPPED, "api.json"))

    def test_single_record_is_reported(self):
        content = json.dumps({"ad": "Ayşe", "puan": 10}).encode()
        self.assertIn("singleRecord", self._codes(content, "tek.json"))

    def test_ragged_records_are_reported(self):
        content = b'{"a": 1}\n{"a": 2, "b": 3}\n'
        self.assertIn("raggedRecords", self._codes(content, "duzensiz.jsonl"))

    def test_flat_uniform_file_has_no_issues(self):
        self.assertEqual(self._codes(ROOT_ARRAY, "a.json"), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
