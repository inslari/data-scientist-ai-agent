import sqlite3
import json
import os
from datetime import datetime
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")

def init_db():
    """Initializes the SQLite database for experiment tracking."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            task TEXT,
            intent TEXT,
            model_name TEXT,
            metrics JSON,
            hyperparameters JSON,
            feature_names JSON,
            target_name TEXT,
            dataset_shape TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_experiment(
    task: str,
    intent: str,
    model_name: str,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    feature_names: list[str],
    target_name: str,
    dataset_shape: str
):
    """Logs an experiment run to the SQLite database."""
    if not os.path.exists(DB_PATH):
        init_db()
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO experiments (
            timestamp, task, intent, model_name, metrics, 
            hyperparameters, feature_names, target_name, dataset_shape
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        task,
        intent,
        model_name,
        json.dumps(metrics),
        json.dumps(hyperparameters),
        json.dumps(feature_names),
        target_name,
        dataset_shape
    ))
    
    conn.commit()
    conn.close()

def get_experiments() -> list[dict]:
    """Retrieves all experiments from the database."""
    if not os.path.exists(DB_PATH):
        return []
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        # Parse JSON fields
        for field in ["metrics", "hyperparameters", "feature_names"]:
            if row_dict[field]:
                try:
                    row_dict[field] = json.loads(row_dict[field])
                except Exception:
                    pass
        results.append(row_dict)
        
    conn.close()
    return results
