import sqlalchemy
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://dsuser:HfkTzazQ8wHqyTMoRNvI@188.132.198.201:5432/dsdb?options=-csearch_path%3Dapp')
with engine.connect() as conn:
    row = conn.execute(text('SELECT id, status, modality, time_col, metrics, evaluation_report, error FROM projects ORDER BY id DESC LIMIT 1')).fetchone()
    print('ID:', row.id)
    print('Status:', row.status)
    print('Modality:', row.modality)
    print('Time Col:', row.time_col)
    print('Error:', row.error)
    print('Metrics:', str(row.metrics)[:200])
    print('Eval:', str(row.evaluation_report)[:200])
