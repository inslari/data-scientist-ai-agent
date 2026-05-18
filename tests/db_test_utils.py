import json
import sqlite3
from pathlib import Path


def init_results_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS intent_discovery_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            agent_state TEXT,
            agent_output TEXT,
            stored_by TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_intent_discovery_result(path: Path, state: dict, output: dict, stored_by: str = "integration_test") -> None:
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO intent_discovery_results (agent_state, agent_output, stored_by) VALUES (?, ?, ?)",
        (json.dumps(state, ensure_ascii=False), json.dumps(output, ensure_ascii=False), stored_by),
    )
    conn.commit()
    conn.close()
