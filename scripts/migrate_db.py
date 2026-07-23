import os
import sys

# Proje kökünü sys.path'e ekle ki 'scripts/' içinden çalıştırıldığında da
# 'api' paketi import edilebilsin (python scripts/migrate_db.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.core.database import engine
from api.core.db_safety import assert_local_db
from sqlalchemy import inspect
from sqlalchemy import text

# GÜVENLİK: remote (production) DB'ye yanlışlıkla migration uygulanmasını engelle.
# Bağlantı açılmadan, hiçbir ALTER çalışmadan önce durur. Bilinçli remote için ALLOW_REMOTE_DB_OPS=1.
assert_local_db(engine.url, context="migrate_db.py")

inspector = inspect(engine)

def check_table(table_name):
    print(f"\nTable: {table_name}")
    try:
        columns = inspector.get_columns(table_name)
        existing = [c['name'] for c in columns]
        print(f"  Existing columns: {existing}")
        return existing
    except Exception as e:
        print(f"Error: {e}")
        return []

existing_cols = check_table("projects")
data_group_cols = check_table("data_groups")
job_cols = check_table("background_jobs")
user_cols = check_table("users")

with engine.connect() as conn:
    if "query" not in existing_cols:
        print("Adding 'query' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN query TEXT"))

    if "intent" not in existing_cols:
        print("Adding 'intent' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN intent VARCHAR"))

    if "description" not in existing_cols:
        print("Adding 'description' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN description TEXT"))

    if "target_col" not in existing_cols:
        print("Adding 'target_col' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN target_col VARCHAR"))

    # Domain (sector) pack selection
    if "domain_pack" not in existing_cols:
        print("Adding 'domain_pack' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN domain_pack VARCHAR"))

    # Data modality axis (tabular | timeseries) + the time column for chronological split
    if "modality" not in existing_cols:
        print("Adding 'modality' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN modality VARCHAR"))

    if "time_col" not in existing_cols:
        print("Adding 'time_col' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN time_col VARCHAR"))

    if "entity_col" not in existing_cols:
        print("Adding 'entity_col' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN entity_col VARCHAR"))

    if "domain_pack" not in data_group_cols:
        print("Adding 'domain_pack' column to 'data_groups'...")
        conn.execute(text("ALTER TABLE data_groups ADD COLUMN domain_pack VARCHAR"))

    if "domain_doc" not in data_group_cols:
        print("Adding 'domain_doc' column to 'data_groups'...")
        conn.execute(text("ALTER TABLE data_groups ADD COLUMN domain_doc TEXT"))

    # Talk-to-Data örnek soru önerileri (LLM ile arka planda üretilir)
    if "ttd_suggestions" not in data_group_cols:
        print("Adding 'ttd_suggestions' column to 'data_groups'...")
        conn.execute(text("ALTER TABLE data_groups ADD COLUMN ttd_suggestions JSON"))

    # Data Visualization öneri kartları (LLM ile arka planda üretilir)
    if "viz_suggestions" not in data_group_cols:
        print("Adding 'viz_suggestions' column to 'data_groups'...")
        conn.execute(text("ALTER TABLE data_groups ADD COLUMN viz_suggestions JSON"))

    # Grubun varsayılan grafik renk teması (src/services/viz_themes.py)
    if "viz_theme" not in data_group_cols:
        print("Adding 'viz_theme' column to 'data_groups'...")
        conn.execute(text("ALTER TABLE data_groups ADD COLUMN viz_theme JSON"))

    # Rapor dili — run anında UI'dan job'a yazılır (tr|en)
    if "language" not in job_cols:
        print("Adding 'language' column to 'background_jobs'...")
        conn.execute(text("ALTER TABLE background_jobs ADD COLUMN language VARCHAR"))

    # Admin rolü — DB-managed LLM provider ayarları admin ile korunuyor.
    if "is_admin" not in user_cols:
        print("Adding 'is_admin' column to 'users'...")
        conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE NOT NULL"))

    conn.commit()

# Mevcut kurulumlarda henüz admin yoksa en eski kullanıcıyı admin yap (kapalı platform,
# ilk kullanıcı = admin). Idempotent: zaten bir admin varsa dokunmaz.
with engine.connect() as conn:
    conn.execute(text(
        "UPDATE users SET is_admin = TRUE "
        "WHERE id = (SELECT MIN(id) FROM users) "
        "AND NOT EXISTS (SELECT 1 FROM users WHERE is_admin = TRUE)"
    ))
    conn.commit()
print("Admin promotion ensured (oldest user).")

# Create data_group_queries table (raw SQL, idempotent)
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS data_group_queries (
            id SERIAL PRIMARY KEY,
            data_group_id INTEGER NOT NULL REFERENCES data_groups(id) ON DELETE CASCADE,
            title VARCHAR NOT NULL,
            description TEXT,
            sql TEXT NOT NULL,
            expert VARCHAR,
            pack_name VARCHAR,
            created_at INTEGER NOT NULL
        )
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_data_group_queries_data_group_id ON data_group_queries (data_group_id)"))
    conn.commit()
print("Table 'data_group_queries' ensured.")

# Create visualizations table (Data Visualization gallery, idempotent)
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS visualizations (
            id SERIAL PRIMARY KEY,
            data_group_id INTEGER NOT NULL REFERENCES data_groups(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL,
            user_query VARCHAR,
            spec JSON,
            image_b64 TEXT,
            chart JSON,
            sql TEXT,
            summary TEXT
        )
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_visualizations_data_group_id ON visualizations (data_group_id)"))
    conn.commit()

# Grafik bazında tema override'ı (tablo daha önce oluşturulmuş olabilir)
viz_cols = check_table("visualizations")
with engine.connect() as conn:
    if "theme" not in viz_cols:
        print("Adding 'theme' column to 'visualizations'...")
        conn.execute(text("ALTER TABLE visualizations ADD COLUMN theme JSON"))
    # SQL sonucu — tablo görünümü + LLM'siz yeniden render için
    if "data" not in viz_cols:
        print("Adding 'data' column to 'visualizations'...")
        conn.execute(text("ALTER TABLE visualizations ADD COLUMN data JSON"))
    conn.commit()

# Mod ayrımı kaldırıldı (tek akış); 'mode' kolonu artık okunmuyor/yazılmıyor ve her
# satırda NULL. Ayrı bir bağlantıda ve toleranslı: eski SQLite DROP COLUMN desteklemez,
# desteklemiyorsa ölü kolon zararsızca kalır.
if "mode" in viz_cols:
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE visualizations DROP COLUMN mode"))
            conn.commit()
        print("Dropped unused 'mode' column from 'visualizations'.")
    except Exception as e:
        print(f"Could not drop 'mode' column (harmless, leaving it): {e}")
print("Table 'visualizations' ensured.")

print("Migration completed.")
