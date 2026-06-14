from api.core.database import engine
from sqlalchemy import inspect

inspector = inspect(engine)

def check_table(table_name):
    print(f"\nTable: {table_name}")
    try:
        columns = inspector.get_columns(table_name)
        for col in columns:
            print(f"  - {col['name']} ({col['type']})")
    except Exception as e:
        print(f"Error: {e}")

check_table("projects")
check_table("background_jobs")
