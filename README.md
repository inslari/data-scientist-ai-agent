# Data Scientist AI Agent

This project contains two independent services:

1. `src/mcp_server.py` — MCP-based database tool server
2. `app.py` — Streamlit dashboard (Agent UI)

## Running

1. Start the MCP server:

```bash
uv run python src/mcp_server.py
```

By default, the MCP server listens on `http://127.0.0.1:9001/mcp`.

2. Start the Streamlit application:

```bash
uv run streamlit run app.py
```

Streamlit runs on `http://localhost:8501` by default.

> Note: MCP server and the Streamlit app are two independent processes. You should run the MCP server first, then the Streamlit dashboard.

## Database Configuration

The project supports dynamic database providers. You can easily switch between SQLite, PostgreSQL, MySQL, and SQL Server without changing the code.

Configure your database connection by editing the `.env` file at the root of the project:

```env
# Main Database (Dataset)
DB_PROVIDER=postgresql # Options: sqlite, postgresql, mysql, mssql
DB_CONNECTION_STRING=postgresql://username:password@localhost:5432/my_database

# History Database (Experiment Logs)
HISTORY_DB_PROVIDER=sqlite
HISTORY_DB_CONNECTION_STRING=sqlite:///data/history.db
```

### Multi-Database Support (Simultaneous Connections)

You can define multiple databases to be queried simultaneously by the AI Agent using the `DATABASES` environment variable in JSON format:

```env
DATABASES='{"default": {"provider": "sqlite", "connection_string": "sqlite:///data/adult_income.db"}, "secondary": {"provider": "postgresql", "connection_string": "postgresql://user:password@localhost/mydb"}}'
```

When configured, the AI Agent reads the schema for all connected databases and will automatically generate and route SQL queries to the correct database based on your question.

**Loading data to a specific database:**
You can specify the target database when loading a Kaggle dataset by using the `--db` flag:
```bash
uv run python src/setup_database.py uciml/adult-census-income --db secondary
```


**Supported Providers:**
- **`sqlite`**: Requires a standard sqlite URI (e.g., `sqlite:///data/my_db.db`)
- **`postgresql`**: Requires `psycopg2-binary` (installed by default)
- **`mysql`**: Requires `pymysql` (installed by default)
- **`mssql`**: Requires `pyodbc` and corresponding ODBC drivers (installed by default)

## Testing

- Test the MCP connection:

```bash
curl http://127.0.0.1:8000/mcp/schema
```

- Run a SQL query:

```bash
curl -X POST http://127.0.0.1:8000/mcp/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "SELECT * FROM customers LIMIT 3"}'
```

- Run the intent discovery integration test and save results to the DB:

```bash
python -m unittest tests.test_intent_discovery_agent
```

`mcp_server.py` and `main.py` work independently; FastAPI only connects to the MCP server over HTTP.
