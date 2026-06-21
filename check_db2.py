import sqlalchemy
import json
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://dsuser:HfkTzazQ8wHqyTMoRNvI@188.132.198.201:5432/dsdb?options=-csearch_path%3Dapp')
with engine.connect() as conn:
    row = conn.execute(text('SELECT id, status, modality, time_col FROM projects ORDER BY id DESC LIMIT 1')).fetchone()
    print('Project ID:', row.id)
    print('Status:', row.status)
    print('Modality:', row.modality)
    print('Time Col:', row.time_col)
    
    print("\n--- Project Steps ---")
    steps = conn.execute(text('SELECT node_name, step_data FROM project_steps WHERE project_id = :pid'), {"pid": row.id}).fetchall()
    for s in steps:
        print(f"Node: {s.node_name}")
        if s.step_data:
            data = s.step_data
            if "error" in data:
                print("  Error:", data["error"])
            if "metrics" in data:
                print("  Metrics:", {k:v for k,v in data["metrics"].items() if k not in ["all_model_scores", "feature_importance", "per_class_report"]})
            if "evaluation_report" in data:
                print("  Eval Metrics:", data["evaluation_report"].get("test_metrics"))
