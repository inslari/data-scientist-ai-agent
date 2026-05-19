import os
import re
import sqlite3
from pathlib import Path
from intelligence_engine import DatabaseIntelligenceEngine

BASE_DIR = Path(__file__).resolve().parent
API_DB_PATH = BASE_DIR / "data" / "meta.db"

def init_db():
    API_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(API_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_schemas (
            table_name TEXT PRIMARY KEY,
            schema_text TEXT,
            inferred_entity TEXT,
            description TEXT,
            data_quality_score REAL,
            analytics_potential_score REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_columns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT,
            column_name TEXT,
            data_type TEXT,
            description TEXT,
            semantic_type TEXT,
            nullability TEXT,
            uniqueness TEXT,
            importance_score REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS database_capabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capability TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS database_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_table TEXT,
            source_column TEXT,
            target_table TEXT,
            target_column TEXT,
            relationship_type TEXT
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
        
        cursor.execute("INSERT OR IGNORE INTO table_schemas (table_name, schema_text) VALUES (?, ?)", (table_name, schema_sql))
        cursor.execute("UPDATE table_schemas SET schema_text = ? WHERE table_name = ?", (schema_sql, table_name))
        
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
    
    # Run intelligence engine
    try:
        print("Running Database Intelligence Engine...")
        engine = DatabaseIntelligenceEngine()
        intelligence = engine.analyze_schema(schema_text)
        
        # Save table metadata
        for t in intelligence.tables:
            cursor.execute('''
                UPDATE table_schemas 
                SET inferred_entity = ?, description = ?, data_quality_score = ?, analytics_potential_score = ?
                WHERE table_name = ?
            ''', (t.inferred_entity, t.description, t.data_quality_score, t.analytics_potential_score, t.table_name))
            
        # Save column metadata
        for c in intelligence.columns:
            # We generate descriptions for columns by concatenating semantic meaning and uniqueness
            col_desc = f"Type: {c.semantic_type}. Uniqueness: {c.uniqueness}. Nullability: {c.nullability}."
            cursor.execute('''
                UPDATE table_columns 
                SET description = ?, semantic_type = ?, nullability = ?, uniqueness = ?, importance_score = ?
                WHERE table_name = ? AND column_name = ?
            ''', (col_desc, c.semantic_type, c.nullability, c.uniqueness, c.importance_score, c.table_name, c.column_name))
            
        # Save capabilities
        cursor.execute("DELETE FROM database_capabilities")
        for cap in intelligence.capabilities:
            cursor.execute("INSERT INTO database_capabilities (capability) VALUES (?)", (cap,))
            
        # Save insights
        cursor.execute("DELETE FROM database_insights")
        for ins in intelligence.global_insights:
            cursor.execute("INSERT INTO database_insights (insight) VALUES (?)", (ins,))
            
        # Save relationships
        cursor.execute("DELETE FROM relationships")
        for rel in intelligence.relationships:
            cursor.execute("INSERT INTO relationships (source_table, source_column, target_table, target_column, relationship_type) VALUES (?, ?, ?, ?, ?)",
                           (rel.source_table, rel.source_column, rel.target_table, rel.target_column, rel.relationship_type))
            
        print("Database Intelligence successfully processed and saved.")
    except Exception as e:
        print(f"Failed to run intelligence engine: {e}")
        
    conn.commit()
    conn.close()
