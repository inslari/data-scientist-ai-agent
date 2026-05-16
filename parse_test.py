import re

schema_text = """Database Schema:
- Table 'customers': CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    full_name TEXT,
    email TEXT,
    phone TEXT,
    city TEXT,
    birth_date TEXT,
    age INTEGER,
    created_at TEXT
)
- Table 'categories': CREATE TABLE categories (
    category_id TEXT PRIMARY KEY,
    name TEXT,
    parent TEXT
)
"""

parts = re.split(r"- Table '([^']+)':", schema_text)
for i in range(1, len(parts), 2):
    table_name = parts[i]
    schema_sql = parts[i+1].strip()
    print(f"Table: {table_name}")
    match = re.search(r"\((.*)\)", schema_sql, re.DOTALL)
    if match:
        cols_str = match.group(1)
        cols = [c.strip() for c in cols_str.split(',')]
        for c in cols:
            if not c: continue
            tokens = c.split()
            if len(tokens) >= 2:
                print(f"  Col: {tokens[0]}, Type: {tokens[1]}")
