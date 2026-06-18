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
load_dotenv(override=True)

from api.core.database import engine
from sqlalchemy import text

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
