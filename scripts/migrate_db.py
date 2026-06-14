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
        
    conn.commit()

print("Migration completed.")
