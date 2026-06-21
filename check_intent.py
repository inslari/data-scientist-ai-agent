import sqlalchemy
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://dsuser:HfkTzazQ8wHqyTMoRNvI@188.132.198.201:5432/dsdb?options=-csearch_path%3Dapp')
with engine.connect() as conn:
    row = conn.execute(text("SELECT step_data FROM project_steps WHERE project_id = (SELECT id FROM projects ORDER BY id DESC LIMIT 1) AND node_name = 'intent_node'")).fetchone()
    data = row.step_data if row else {}
    print('Detected Intent:', data.get('intent'))
    print('Target Col:', data.get('target_col'))
