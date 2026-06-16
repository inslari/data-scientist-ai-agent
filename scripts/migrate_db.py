from api.core.database import engine
from sqlalchemy import inspect
from sqlalchemy import text

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

    # Domain (sector) pack selection
    if "domain_pack" not in existing_cols:
        print("Adding 'domain_pack' column to 'projects'...")
        conn.execute(text("ALTER TABLE projects ADD COLUMN domain_pack VARCHAR"))

    if "domain_pack" not in data_group_cols:
        print("Adding 'domain_pack' column to 'data_groups'...")
        conn.execute(text("ALTER TABLE data_groups ADD COLUMN domain_pack VARCHAR"))

    if "domain_doc" not in data_group_cols:
        print("Adding 'domain_doc' column to 'data_groups'...")
        conn.execute(text("ALTER TABLE data_groups ADD COLUMN domain_doc TEXT"))

    conn.commit()

print("Migration completed.")
