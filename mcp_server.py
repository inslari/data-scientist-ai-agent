from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "data" / "adult_income.db"

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "9001"))
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")

mcp = FastMCP(
    "SQLite MCP Server",
    host=MCP_SERVER_HOST,
    port=MCP_SERVER_PORT,
    streamable_http_path=MCP_SERVER_PATH,
)


@mcp.tool()
def get_schema() -> str:
    """Get the schema of all tables in the SQLite database. Use this to understand what columns exist."""
    try:
        if not DB_PATH.exists():
            return f"Error: Database not found at {DB_PATH}"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()

        schema_info = "Database Schema:\n"
        for table_name, table_sql in tables:
            schema_info += f"- Table '{table_name}': {table_sql}\n"
        return schema_info
    except Exception as e:
        return f"Error getting schema: {e}"


@mcp.tool()
def run_sql_query(query: str) -> str:
    """Execute a SELECT query on the SQLite database and return the results. Only SELECT queries are allowed."""
    try:
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        column_names = [description[0] for description in cursor.description]
        conn.close()

        output = f"Columns: {', '.join(column_names)}\n"
        output += f"Returned {len(results)} rows.\nResults:\n"

        MAX_DISPLAY_ROWS = 10_000
        for row in results[:MAX_DISPLAY_ROWS]:
            output += str(row) + "\n"

        if len(results) > MAX_DISPLAY_ROWS:
            output += f"... (Showing only first {MAX_DISPLAY_ROWS} rows)"

        return output
    except Exception as e:
        return f"SQL Error: {e}"


def start_mcp_server(transport: str = "streamable-http", **kwargs) -> None:
    """Start the MCP server with the given transport."""
    mcp.run(transport=transport, **kwargs)


if __name__ == "__main__":
    start_mcp_server(transport="streamable-http")
