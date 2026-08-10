"""Excel → CSV dönüşüm uçları için uçtan uca testler (LLM'siz, uzak DB'siz).

Test edilen davranışlar:

A) REGISTRY — /convert/converters registry'yi ayar şemasıyla birlikte döndürür.
B) INSPECT  — Excel sayfaları, satır/sütun sayıları ve başlık önizlemesi.
C) DOWNLOAD — tek çıktı düz dosya, çoklu çıktı ZIP olarak iner.
D) ASSETS   — çıktı CSV varlığı olarak kaydedilir; data group + CSV kaydı oluşur.
E) DOĞRULAMA — desteklenmeyen uzantı, bilinmeyen converter, bozuk dosya.

Çalıştırma (sandbox DB zorunlu — asla uzak DB'ye bağlanma):

    API_DATABASE_URL="sqlite:///$(pwd)/data/test_convert.sqlite" \
        uv run python tests/test_convert_routes.py
"""

import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

# Uzak DB'ye kazara bağlanmayı engelle: test her zaman geçici bir sqlite kullanır.
_TMP_DB = Path(tempfile.gettempdir()) / "ds_agent_test_convert.sqlite"
os.environ["API_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.core.database import Base, SessionLocal, engine, get_db  # noqa: E402
from api.core.deps import get_current_user  # noqa: E402
from api.models.csv import CSVFileModel  # noqa: E402
from api.models.data_group import DataGroupModel  # noqa: E402
from api.models.user import UserModel  # noqa: E402
# create_all'ın tüm tabloları görmesi için (main.py ile aynı gerekçe)
from api.models.job import BackgroundJobModel  # noqa: F401,E402
from api.models.project import ProjectModel  # noqa: F401,E402
from api.models.mcp import MCPConfigModel  # noqa: F401,E402
from api.models.data_group_query import DataGroupQueryModel  # noqa: F401,E402
from api.models.db_intelligence_cache import DbIntelligenceCacheModel  # noqa: F401,E402
from api.models.log_entry import LogEntryModel  # noqa: F401,E402
from api.models.llm_usage import LLMUsageModel  # noqa: F401,E402
from api.models.ai_provider import AiProviderModel  # noqa: F401,E402
from api.models.ai_model import AiModelModel  # noqa: F401,E402
from api.models.ai_role_assignment import AiRoleAssignmentModel  # noqa: F401,E402
from api.models.password_reset_request import PendingPasswordResetModel  # noqa: F401,E402
from api.routes.convert import router as convert_router  # noqa: E402
from api.routes.csv import router as csv_router  # noqa: E402


def _build_workbook() -> bytes:
    """İki dolu, bir boş sayfası olan test çalışma kitabı."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"müşteri": ["Ayşe", "Bora"], "tutar ": [10.5, 20.0]}).to_excel(
            writer, sheet_name="Satışlar", index=False
        )
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_excel(
            writer, sheet_name="Detay", index=False
        )
        pd.DataFrame().to_excel(writer, sheet_name="Bos", index=False)
    return buf.getvalue()


class ConvertRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

        db = SessionLocal()
        user = db.query(UserModel).filter(UserModel.email == "convert-test@local").first()
        if not user:
            user = UserModel(email="convert-test@local", hashed_password="x", is_active=True)
            db.add(user)
            db.commit()
            db.refresh(user)
        cls.user_id = user.id
        db.close()

        app = FastAPI()
        app.include_router(convert_router)
        # Dönüşüm ve düz CSV yüklemesi aynı ingest hattını (api/core/csv_ingest)
        # paylaşır; ikisi de aynı uygulamada test edilir.
        app.include_router(csv_router)

        def _override_db():
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()

        def _override_user():
            db = SessionLocal()
            try:
                return db.query(UserModel).filter(UserModel.id == cls.user_id).first()
            finally:
                db.close()

        app.dependency_overrides[get_db] = _override_db
        app.dependency_overrides[get_current_user] = _override_user

        cls.client = TestClient(app)
        cls.workbook = _build_workbook()

    @classmethod
    def tearDownClass(cls):
        """Testin diske yazdığı CSV'leri ve DB satırlarını temizle."""
        db = SessionLocal()
        try:
            files = db.query(CSVFileModel).filter(CSVFileModel.user_id == cls.user_id).all()
            for f in files:
                if os.path.isabs(f.file_path) and os.path.exists(f.file_path):
                    os.remove(f.file_path)
                db.delete(f)
            for group in db.query(DataGroupModel).filter(DataGroupModel.user_id == cls.user_id).all():
                db.delete(group)
            db.commit()
        finally:
            db.close()

    # --- A) registry ---------------------------------------------------

    def test_a_converters_listed(self):
        res = self.client.get("/convert/converters")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        ids = [c["id"] for c in body["data"]]
        self.assertIn("excel_to_csv", ids)

        excel = next(c for c in body["data"] if c["id"] == "excel_to_csv")
        self.assertTrue(excel["can_save_to_assets"])
        self.assertIn(".xlsx", excel["source_extensions"])
        option_names = {o["name"] for o in excel["options"]}
        self.assertTrue({"sheets", "has_header", "delimiter"}.issubset(option_names))

    def test_a_response_model_does_not_drop_registry_fields(self):
        """
        Pydantic response_model bilmediği alanı SESSİZCE atar — registry'ye yeni
        bir alan eklenip şemaya eklenmezse arayüz onu hiç görmez (bu bir kez
        `default_source`/`live_preview` ile başımıza geldi).
        """
        from api.core.converters import list_converters

        served = {c["id"]: c for c in self.client.get("/convert/converters").json()["data"]}
        for registry_entry in list_converters():
            with self.subTest(converter=registry_entry["id"]):
                served_entry = served[registry_entry["id"]]
                self.assertEqual(set(registry_entry), set(served_entry))
                for opt, served_opt in zip(registry_entry["options"], served_entry["options"]):
                    self.assertEqual(set(opt), set(served_opt))

    # --- B) inspect ----------------------------------------------------

    def test_b_inspect_lists_sheets(self):
        res = self.client.post(
            "/convert/inspect",
            data={"converter_id": "excel_to_csv"},
            files={"file": ("rapor.xlsx", self.workbook, "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "success")

        sheets = {s["name"]: s for s in body["data"]["sheets"]}
        self.assertEqual(set(sheets), {"Satışlar", "Detay", "Bos"})
        self.assertEqual(sheets["Detay"]["rows"], 3)
        self.assertEqual(sheets["Detay"]["columns"], 2)
        self.assertTrue(sheets["Bos"]["empty"])
        self.assertEqual(sheets["Satışlar"]["preview_columns"][0], "müşteri")

    # --- C) indirme ----------------------------------------------------

    def test_c_download_single_sheet_is_plain_csv(self):
        res = self.client.post(
            "/convert/run",
            data={
                "converter_id": "excel_to_csv",
                "output": "download",
                "options": '{"sheets": ["Detay"]}',
            },
            files={"files": ("rapor.xlsx", self.workbook, "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"].split(";")[0], "text/csv")
        self.assertIn("rapor.csv", res.headers["content-disposition"])
        # Bu başlık olmadan tarayıcı Content-Disposition'ı JS'e vermez ve dosya
        # yanlış adla iner (CORS) — arayüz buna bağlı.
        self.assertIn("Content-Disposition", res.headers["access-control-expose-headers"])
        self.assertEqual(res.text.splitlines()[0], "a,b")

    def test_c_download_multi_sheet_is_zip(self):
        res = self.client.post(
            "/convert/run",
            data={"converter_id": "excel_to_csv", "output": "download"},
            files={"files": ("rapor.xlsx", self.workbook, "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/zip")
        # ZIP adı tek kaynak dosyadan türetilir.
        self.assertIn("rapor.zip", res.headers["content-disposition"])

        with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
            names = zf.namelist()
        # Boş sayfa atlanır → iki CSV kalır.
        self.assertEqual(len(names), 2)
        self.assertTrue(all(n.endswith(".csv") for n in names))

    def test_c_download_respects_delimiter_option(self):
        res = self.client.post(
            "/convert/run",
            data={
                "converter_id": "excel_to_csv",
                "output": "download",
                "options": '{"sheets": ["Detay"], "delimiter": ";"}',
            },
            files={"files": ("rapor.xlsx", self.workbook, "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.text.splitlines()[0], "a;b")

    # --- D) veri varlıklarına kaydetme ---------------------------------

    def test_d_save_to_assets_creates_group_and_files(self):
        res = self.client.post(
            "/convert/run",
            data={
                "converter_id": "excel_to_csv",
                "output": "assets",
                "group_name": "Dönüşüm Testi",
            },
            files={"files": ("rapor.xlsx", self.workbook, "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(len(body["data"]), 2)
        self.assertEqual(len(body["converted"]), 2)

        db = SessionLocal()
        try:
            group = db.query(DataGroupModel).filter(
                DataGroupModel.id == body["data_group_id"]
            ).first()
            self.assertIsNotNone(group)
            self.assertEqual(group.name, "Dönüşüm Testi")
            self.assertEqual(group.source_type, "csv")

            files = db.query(CSVFileModel).filter(
                CSVFileModel.data_group_id == group.id
            ).all()
            self.assertEqual(len(files), 2)
            # Şema ve satır sayısı CSV yüklemesindeki ile aynı hattan çıkar.
            detay = next(f for f in files if "Detay" in f.filename)
            self.assertEqual(detay.row_count, 3)
            self.assertEqual([c["name"] for c in detay.columns_schema], ["a", "b"])
        finally:
            db.close()

    # --- E) doğrulama --------------------------------------------------

    def test_e_rejects_unsupported_extension(self):
        res = self.client.post(
            "/convert/run",
            data={"converter_id": "excel_to_csv", "output": "download"},
            files={"files": ("veri.csv", b"a,b\n1,2\n", "text/csv")},
        )
        self.assertEqual(res.status_code, 400)

    def test_e_rejects_unknown_converter(self):
        res = self.client.post(
            "/convert/run",
            data={"converter_id": "pdf_to_gold", "output": "download"},
            files={"files": ("rapor.xlsx", self.workbook, "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 404)

    def test_e_corrupt_workbook_returns_400(self):
        res = self.client.post(
            "/convert/run",
            data={"converter_id": "excel_to_csv", "output": "download"},
            files={"files": ("bozuk.xlsx", b"definitely not excel", "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 400)

    # --- F) düz CSV yüklemesi (paylaşılan ingest hattı bozulmadı mı) ---

    def test_f_plain_csv_upload_still_works(self):
        """/csv/upload dönüşümle aynı ingest yardımcılarını kullanır — regresyon."""
        csv_bytes = "ad,tutar\nAyşe,10\nBora,20\n".encode("utf-8")
        res = self.client.post(
            "/csv/upload",
            data={"group_name": "Düz Yükleme"},
            files={"files": ("veri.csv", csv_bytes, "text/csv")},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(len(body["data"]), 1)

        uploaded = body["data"][0]
        self.assertEqual(uploaded["filename"], "veri.csv")
        self.assertEqual(uploaded["row_count"], 2)
        self.assertEqual([c["name"] for c in uploaded["columns_schema"]], ["ad", "tutar"])
        self.assertEqual(body["warnings"], [])

    def test_f_upload_warns_when_file_silently_collapses(self):
        """
        `;` ayırıcılı UTF-8 dosya hatasız ama TEK SÜTUN olarak yüklenir. Veriye
        dokunmuyoruz; arayüzün kullanıcıyı dönüştürücüye yönlendirebilmesi için
        uyarı yanıtta gelmeli (yükleme yine 'success').
        """
        res = self.client.post(
            "/csv/upload",
            data={"group_name": "Noktalı Virgüllü"},
            files={"files": ("bozuk.csv", "ad;tutar\nx;1\ny;2\n".encode("utf-8"), "text/csv")},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "success")

        codes = {w["code"] for w in body["warnings"]}
        self.assertIn("singleColumn", codes)
        warning = next(w for w in body["warnings"] if w["code"] == "singleColumn")
        self.assertEqual(warning["filename"], "bozuk.csv")
        self.assertEqual(warning["params"]["delimiter"], ";")

    def test_e_invalid_options_json_returns_400(self):
        res = self.client.post(
            "/convert/run",
            data={
                "converter_id": "excel_to_csv",
                "output": "download",
                "options": "{not json",
            },
            files={"files": ("rapor.xlsx", self.workbook, "application/vnd.ms-excel")},
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
