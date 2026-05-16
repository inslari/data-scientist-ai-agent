import os
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_DB_PATH = BASE_DIR / "data" / "meta.db"

def init_db():
    API_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(API_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_schemas (
            table_name TEXT PRIMARY KEY,
            schema_text TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_columns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT,
            column_name TEXT,
            data_type TEXT,
            description TEXT
        )
    ''')
    conn.commit()
    conn.close()

def parse_and_save_schema(schema_text: str):
    conn = sqlite3.connect(API_DB_PATH)
    cursor = conn.cursor()

    parts = re.split(r"- Table '([^']+)':", schema_text)
    for i in range(1, len(parts), 2):
        table_name = parts[i]
        schema_sql = parts[i+1].strip()
        
        cursor.execute("INSERT OR REPLACE INTO table_schemas (table_name, schema_text) VALUES (?, ?)", (table_name, schema_sql))
        
        match = re.search(r"\((.*)\)", schema_sql, re.DOTALL)
        if match:
            cols_str = match.group(1)
            cols = [c.strip() for c in cols_str.split(',')]
            for c in cols:
                if not c: continue
                tokens = c.split()
                if len(tokens) >= 2:
                    col_name = tokens[0]
                    if col_name.upper() in ("PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK"):
                        continue
                    data_type = tokens[1]
                    
                    cursor.execute("SELECT id FROM table_columns WHERE table_name = ? AND column_name = ?", (table_name, col_name))
                    if not cursor.fetchone():
                        cursor.execute("INSERT INTO table_columns (table_name, column_name, data_type, description) VALUES (?, ?, ?, ?)", 
                                       (table_name, col_name, data_type, ""))
    conn.commit()
    conn.close()
