"""
Experiment History Tracker
--------------------------
Logs ML experiment runs to the history database.
Uses the db_provider module for provider-agnostic database access.
"""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import text

from src.core.db_provider import history_db_provider


def init_db():
    """Initializes the database for experiment tracking."""
    create_sql = """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            task TEXT,
            intent TEXT,
            model_name TEXT,
            metrics TEXT,
            hyperparameters TEXT,
            feature_names TEXT,
            target_name TEXT,
            dataset_shape TEXT
        )
    """
    # Adjust for providers that use different auto-increment syntax
    if history_db_provider.provider == "postgresql":
        create_sql = """
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                timestamp TEXT,
                task TEXT,
                intent TEXT,
                model_name TEXT,
                metrics TEXT,
                hyperparameters TEXT,
                feature_names TEXT,
                target_name TEXT,
                dataset_shape TEXT
            )
        """
    elif history_db_provider.provider == "mysql":
        create_sql = """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                timestamp TEXT,
                task TEXT,
                intent TEXT,
                model_name TEXT,
                metrics TEXT,
                hyperparameters TEXT,
                feature_names TEXT,
                target_name TEXT,
                dataset_shape TEXT
            )
        """
    elif history_db_provider.provider == "mssql":
        # MSSQL doesn't support IF NOT EXISTS in CREATE TABLE directly
        create_sql = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='experiments' AND xtype='U')
            CREATE TABLE experiments (
                id INT IDENTITY(1,1) PRIMARY KEY,
                timestamp NVARCHAR(255),
                task NVARCHAR(255),
                intent NVARCHAR(255),
                model_name NVARCHAR(255),
                metrics NVARCHAR(MAX),
                hyperparameters NVARCHAR(MAX),
                feature_names NVARCHAR(MAX),
                target_name NVARCHAR(255),
                dataset_shape NVARCHAR(255)
            )
        """

    with history_db_provider.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()


def log_experiment(
    task: str,
    intent: str,
    model_name: str,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    feature_names: list[str],
    target_name: str,
    dataset_shape: str,
):
    """Logs an experiment run to the database."""
    # Ensure table exists
    init_db()

    insert_sql = """
        INSERT INTO experiments (
            timestamp, task, intent, model_name, metrics, 
            hyperparameters, feature_names, target_name, dataset_shape
        ) VALUES (:timestamp, :task, :intent, :model_name, :metrics, 
                  :hyperparameters, :feature_names, :target_name, :dataset_shape)
    """

    with history_db_provider.connect() as conn:
        conn.execute(text(insert_sql), {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "intent": intent,
            "model_name": model_name,
            "metrics": json.dumps(metrics),
            "hyperparameters": json.dumps(hyperparameters),
            "feature_names": json.dumps(feature_names),
            "target_name": target_name,
            "dataset_shape": dataset_shape,
        })
        conn.commit()


def get_experiments() -> list[dict]:
    """Retrieves all experiments from the database."""
    try:
        init_db()  # Ensure table exists

        with history_db_provider.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM experiments ORDER BY timestamp DESC")
            )
            columns = list(result.keys())
            rows = result.fetchall()

        results = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            # Parse JSON fields
            for field in ["metrics", "hyperparameters", "feature_names"]:
                if row_dict.get(field):
                    try:
                        row_dict[field] = json.loads(row_dict[field])
                    except Exception:
                        pass
            results.append(row_dict)

        return results
    except Exception:
        return []
