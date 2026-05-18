# Data Scientist AI Agent

This project contains two independent services:

1. `mcp_server.py` — MCP-based tool server
2. `main.py` / `api.py` — FastAPI application

## Running

1. Start the MCP server:

```bash
uv run python3 mcp_server.py
```

By default, the MCP server listens on `http://127.0.0.1:9001/mcp`.

2. Start the FastAPI application:

```bash
uv run python3 main.py
```

FastAPI runs on `http://127.0.0.1:8000` by default.

> Note: MCP server and FastAPI are two independent projects. You can simply run the MCP server first, then FastAPI.

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
