"""
Hafif, idempotent başlangıç migration'ı (text-analysis özelliği).
=================================================================

`Base.metadata.create_all` mevcut tablolara YENİ kolon EKLEMEZ. Bu yüzden
`background_jobs` tablosu önceden oluşmuş bir DB'de `text_analysis_id` kolonu
eksik kalır. Burada güvenli, idempotent bir ALTER ile kolonu ekliyoruz —
hem SQLite hem PostgreSQL'de çalışır, kolon zaten varsa sessizce geçer.

API ve worker süreçleri başlarken `create_all`'dan sonra bir kez çağrılır.
"""
from sqlalchemy import text

from api.core.database import engine
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def _add_column(conn, is_sqlite: bool, table: str, column: str, col_type: str) -> None:
    """Idempotent ALTER ... ADD COLUMN (sqlite + postgres)."""
    if is_sqlite:
        cols = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        existing = {row[1] for row in cols}  # row[1] = column name
        if cols and column not in existing:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
            logger.info(f"[migrate] {table}.{column} eklendi (sqlite)")
    else:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"))


def ensure_text_analysis_schema() -> None:
    is_sqlite = engine.url.get_backend_name() == "sqlite"
    # SQLite ALTER TABLE ... ADD COLUMN, kolon varsa hata verir; bu yüzden önce
    # kolon var mı diye bakarız. PostgreSQL'de IF NOT EXISTS destekli.
    json_type = "JSON" if is_sqlite else "JSONB"
    try:
        with engine.begin() as conn:
            _add_column(conn, is_sqlite, "background_jobs", "text_analysis_id", "INTEGER")
            # CSV kolon girişi için source_texts ile hizalı tarih/puan/etiket dizileri.
            _add_column(conn, is_sqlite, "text_analyses", "source_timestamps", json_type)
            _add_column(conn, is_sqlite, "text_analyses", "source_ratings", json_type)
            _add_column(conn, is_sqlite, "text_analyses", "source_labels", json_type)
    except Exception as exc:  # pragma: no cover - başlangıçta best-effort
        logger.warning(f"[migrate] text-analysis şema kolonları garanti edilemedi: {exc}")
