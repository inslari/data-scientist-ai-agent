# Data Scientist AI Agent

A production-ready, multi-user AI agent that automatically performs end-to-end data science workflows (EDA, feature engineering, AutoML, model evaluation) on your datasets.

The system is composed of **three independent processes** that must be run separately:

| Process | File | Description |
|---|---|---|
| API Server | `main.py` | FastAPI REST backend — serves the Backoffice UI |
| Background Worker | `worker_main.py` | Runs AI/ML jobs asynchronously, decoupled from the API |
| MCP Tool Server | `src/mcp_server.py` | Provides database tools (SQL, schema) to the AI Agent via MCP |

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL (recommended for production; SQLite supported for local development)

---

## Environment Setup

Copy and configure your environment variables:

```bash
cp .env.example .env
```

Key variables in `.env`:

```env
# FastAPI / Worker shared database (PostgreSQL recommended)
API_DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/ds_agent

# LangGraph checkpoint store (same Postgres instance recommended for multi-user safety)
# Falls back to SQLite if not set or if SQLite is used above

# LLM API Key
OPENAI_API_KEY=sk-...
# or
GOOGLE_API_KEY=...
```

---

## Running

The three processes must be started **in separate terminal windows**.

### 1. Start the MCP Tool Server

The MCP server provides SQL execution and database schema tools to the AI agent.

```bash
uv run python src/mcp_server.py
```

By default, listens on `http://127.0.0.1:9001/mcp`.

---

### 2. Start the API Server

The REST API serves the Backoffice frontend and handles user authentication, project management, and data uploads.

```bash
uv run python main.py
```

API is available at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

> **Production tip:** Use `gunicorn` with multiple Uvicorn workers for higher throughput:
> ```bash
> uv run gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
> ```

---

### 3. Start the Background Worker

The worker polls the database for pending AI analysis jobs and executes the LangGraph pipeline. It is **completely decoupled from the API**, so heavy ML operations never block web traffic.

```bash
uv run python worker_main.py
```

> **Multi-user safety:** The worker uses PostgreSQL's `SELECT ... FOR UPDATE SKIP LOCKED` to ensure no two worker instances pick up the same job simultaneously. You can safely run multiple worker instances in parallel.

> **On crash recovery:** At startup, the worker automatically re-queues any jobs that were left in `running` or `pending` state from a previous crash.

---

## Database Configuration

The project uses **two database connections**:

1. **Application DB** (`API_DATABASE_URL`): Stores users, projects, data groups, CSV files, and job queues. Shared by both `main.py` and `worker_main.py`.

2. **Dataset DBs** (MCP server): The AI agent can query your actual datasets via the MCP tool server. Configure these in your `.env`:

```env
# Single database
DB_PROVIDER=postgresql
DB_CONNECTION_STRING=postgresql://username:password@localhost:5432/my_dataset

# Or multiple databases simultaneously (JSON format)
DATABASES='{"default": {"provider": "sqlite", "connection_string": "sqlite:///data/my_data.db"}, "secondary": {"provider": "postgresql", "connection_string": "postgresql://user:password@localhost/mydb"}}'
```

**Supported dataset providers:** `sqlite`, `postgresql`, `mysql`, `mssql`

---

## Architecture Overview

```
[ Backoffice UI (React) ]
         |
         ▼
[ main.py  — FastAPI API Server ]
         |
         | writes job to background_jobs table
         ▼
[ PostgreSQL — Shared Database ]
         |
         | worker polls FOR UPDATE SKIP LOCKED
         ▼
[ worker_main.py — Background Worker ]
         |
         | LangGraph pipeline calls MCP tools
         ▼
[ src/mcp_server.py — MCP Tool Server ]
         |
         ▼
[ Dataset Databases (SQLite / PostgreSQL / MySQL / MSSQL) ]
```

---

## Health Check

```bash
curl http://localhost:8000/health
```

---

## Testing the MCP Server

```bash
# Check schema
curl http://127.0.0.1:9001/mcp/schema

# Run a SQL query
curl -X POST http://127.0.0.1:9001/mcp/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "SELECT * FROM customers LIMIT 3"}'
```
