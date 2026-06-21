import sqlalchemy
import json
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://dsuser:HfkTzazQ8wHqyTMoRNvI@188.132.198.201:5432/dsdb?options=-csearch_path%3Dapp')
with engine.connect() as conn:
    row = conn.execute(text('SELECT id, status, modality, time_col FROM projects ORDER BY id DESC LIMIT 1')).fetchone()
    
    steps = conn.execute(text('SELECT node_name, step_data FROM project_steps WHERE project_id = :pid'), {"pid": row.id}).fetchall()
    
    with open('error_log.txt', 'w', encoding='utf-8') as f:
        f.write(f"Project ID: {row.id}\nStatus: {row.status}\n")
        for s in steps:
            f.write(f"Node: {s.node_name}\n")
            if s.step_data:
                data = s.step_data
                if "error" in data:
                    f.write(f"  Error: {data['error']}\n")
