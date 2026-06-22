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


def ensure_text_analysis_schema() -> None:
    is_sqlite = engine.url.get_backend_name() == "sqlite"
    # SQLite ALTER TABLE ... ADD COLUMN, kolon varsa hata verir; bu yüzden önce
    # kolon var mı diye bakarız. PostgreSQL'de IF NOT EXISTS destekli.
    try:
        with engine.begin() as conn:
            if is_sqlite:
                cols = conn.execute(text("PRAGMA table_info(background_jobs)")).fetchall()
                existing = {row[1] for row in cols}  # row[1] = column name
                if "text_analysis_id" not in existing and cols:
                    conn.execute(text("ALTER TABLE background_jobs ADD COLUMN text_analysis_id INTEGER"))
                    logger.info("[migrate] background_jobs.text_analysis_id eklendi (sqlite)")
            else:
                conn.execute(text(
                    "ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS text_analysis_id INTEGER"
                ))
    except Exception as exc:  # pragma: no cover - başlangıçta best-effort
        logger.warning(f"[migrate] text_analysis_id kolonu garanti edilemedi: {exc}")
