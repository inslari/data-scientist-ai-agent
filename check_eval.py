import sqlalchemy
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://dsuser:HfkTzazQ8wHqyTMoRNvI@188.132.198.201:5432/dsdb?options=-csearch_path%3Dapp')
with engine.connect() as conn:
    row = conn.execute(text("SELECT step_data, node_name FROM project_steps WHERE project_id = (SELECT id FROM projects ORDER BY id DESC LIMIT 1) AND node_name IN ('automl_node', 'evaluation_node') ORDER BY id")).fetchall()
    for r in row:
        print(f"--- {r.node_name} ---")
        data = r.step_data if r else {}
        if r.node_name == 'automl_node':
            print('Model:', data.get('model_name'))
            print('Error:', data.get('error'))
        if r.node_name == 'evaluation_node':
            print('Eval Report Keys:', list(data.get('evaluation_report', {}).keys()))
            print('Error:', data.get('error'))
