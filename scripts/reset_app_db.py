"""
Proje/job geçmişini temizler.
users, mcp_configs, data_groups, csv_files tablolarına dokunmaz.

Çalıştır:
    cd data-scientist-ai-agent
    uv run python scripts/reset_app_db.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
# override=False: açıkça verilen API_DATABASE_URL .env'i ezsin (izole çalıştırma için).
load_dotenv(override=False)

from api.core.database import engine
from api.core.db_safety import assert_local_db
from sqlalchemy import text

# GÜVENLİK: bu script DELETE çalıştırır — remote (production) verisini yanlışlıkla
# silmeyi engelle. Bilinçli remote için ALLOW_REMOTE_DB_OPS=1.
assert_local_db(engine.url, context="reset_app_db.py")

TABLES = [
    "data_group_queries",
    "chat_messages",
    "background_jobs",
    "project_steps",
    "projects",
    "token_blacklist",
]


def reset():
    print("Temizleniyor...")
    with engine.begin() as conn:
        for table in TABLES:
            try:
                conn.execute(text(f'DELETE FROM "{table}"'))
                print(f"  ✓ {table}")
            except Exception as e:
                print(f"  ✗ {table}: {e}")
    print("Tamamlandı.")


if __name__ == "__main__":
    reset()
