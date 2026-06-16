from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

# Add project root to sys.path so we can import src.* when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sqlalchemy import text

from src.core.config import settings  # type: ignore[import-not-found]
from src.core.db_provider import (  # type: ignore[import-not-found]
    DatabaseManager,
    db_manager as _shared_db_manager,
)

load_dotenv(override=True)

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "9001"))
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")

# Dataset DB served by the MCP tool layer.
# In local mode (ENVIRONMENT=local) the MCP server serves data/financial.db only;
# the rest of the application keeps using settings.DATABASES (e.g. PostgreSQL).
if settings.ENVIRONMENT == "local":
    db_manager = DatabaseManager({
        "default": {
            "provider": "sqlite",
            "connection_string": f"sqlite:///{settings.LOCAL_FINANCIAL_DB_PATH}",
        }
    })
    print(f"[MCP] ENVIRONMENT=local → serving SQLite dataset: {settings.LOCAL_FINANCIAL_DB_PATH}", flush=True)
else:
    db_manager = _shared_db_manager

mcp = FastMCP(
    "Database MCP Server",
    host=MCP_SERVER_HOST,
    port=MCP_SERVER_PORT,
    streamable_http_path=MCP_SERVER_PATH,
    # One-shot clients (api/routes/mcp.py:_call_mcp_tool) connect → call → close.
    # Plain JSON responses + stateless sessions avoid the SSE "ClosedResourceError"
    # / "unhandled errors in a TaskGroup" the server raised when the SSE stream was
    # torn down before completing.
    json_response=True,
    stateless_http=True,
)


@mcp.tool()
def get_schema() -> str:
    """Get the schema of all tables in the database. Use this to understand what columns exist."""
    try:
        output = ""
        for db_name, provider in db_manager.get_all_providers().items():
            output += f"Database: '{db_name}'\n"
            output += provider.build_full_schema_text()
            output += "\n"
        return output
    except Exception as e:
        return f"Error getting schema: {e}"


@mcp.tool()
def get_column_stats(table_name: str) -> str:
    """
    Get per-column statistics for a specific table.
    Numeric columns: min, max, avg, null count.
    Text columns: top 10 most frequent distinct values with counts.
    Use this to understand what each column actually contains before writing SQL.
    """
    try:
        output = f"Column Statistics for table '{table_name}':\n"
        found_table = False
        
        for db_name, provider in db_manager.get_all_providers().items():
            with provider.connect() as conn:
                # Check if table exists in this DB
                tables_result = conn.execute(text(provider.get_table_names_query()))
                tables = [r[0] for r in tables_result]
                
                if table_name not in tables:
                    continue
                    
                found_table = True
                output += f"\n=== Database: '{db_name}' ===\n"
                output += f"Table: {table_name}\n"

                # Get column info
                col_result = conn.execute(
                    text(provider.get_column_info_query(table_name))
                )
                cols = col_result.fetchall()

                # Total rows
                count_result = conn.execute(
                    text(provider.get_count_query(table_name))
                )
                count_row = count_result.fetchone()
                total_rows = count_row[0] if count_row else 0
                output += f"  Total rows: {total_rows:,}\n"

                for col in cols:
                    if provider.provider == "sqlite":
                        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
                        col_name = col[1]
                        col_type = col[2] or ""
                    else:
                        # information_schema returns: column_name, data_type, is_nullable, column_default
                        col_name = col[0]
                        col_type = col[1] or ""

                    is_numeric = provider.is_numeric_type(col_type)

                    try:
                        # Null count
                        null_result = conn.execute(
                            text(provider.get_null_count_query(table_name, col_name))
                        )
                        null_row = null_result.fetchone()
                        null_count = null_row[0] if null_row else 0
                        null_pct = round(null_count / total_rows * 100, 1) if total_rows else 0

                        if is_numeric:
                            stats_result = conn.execute(
                                text(provider.get_column_stats_numeric_query(table_name, col_name))
                            )
                            stats_row = stats_result.fetchone()
                            mn, mx, avg, n_distinct = stats_row if stats_row else (0, 0, 0, 0)
                            output += (
                                f'  "{col_name}" [numeric]: '
                                f'min={mn}, max={mx}, avg={avg}, '
                                f'distinct={n_distinct}, nulls={null_count} ({null_pct}%)\n'
                            )
                        else:
                            # Top 10 most frequent values
                            top_result = conn.execute(
                                text(provider.get_column_stats_text_top_values_query(table_name, col_name))
                            )
                            top_vals = top_result.fetchall()
                            vals_str = ", ".join(
                                f'"{v}"({c})' for v, c in top_vals
                            )
                            distinct_result = conn.execute(
                                text(provider.get_distinct_count_query(table_name, col_name))
                            )
                            distinct_row = distinct_result.fetchone()
                            n_distinct = distinct_row[0] if distinct_row else 0
                            output += (
                                f'  "{col_name}" [text]: '
                                f'distinct={n_distinct}, nulls={null_count} ({null_pct}%), '
                                f'top values=[{vals_str}]\n'
                            )
                    except Exception as col_err:
                        output += f'  "{col_name}": error — {col_err}\n'

        if not found_table:
            return f"Error: Table '{table_name}' not found in any database."

        return output

    except Exception as e:
        return f"Error getting column stats: {e}"


@mcp.tool()
def run_sql_query(query: str, db_name: str = "default") -> str:
    """Execute a SELECT query on the specified database and return the results. Only SELECT queries are allowed."""
    try:
        query_upper = query.strip().upper()
        if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH") or query_upper.startswith("PRAGMA")):
            return "Error: Only SELECT, WITH, and PRAGMA queries are allowed."

        try:
            provider = db_manager.get_provider(db_name)
        except ValueError as e:
            return f"Error: {e}"

        with provider.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            column_names = list(result.keys())

        output = f"Columns: {', '.join(column_names)}\n"
        output += f"Returned {len(rows)} rows.\nResults:\n"

        MAX_DISPLAY_ROWS = 10_000
        for row in rows[:MAX_DISPLAY_ROWS]:
            output += str(tuple(row)) + "\n"

        if len(rows) > MAX_DISPLAY_ROWS:
            output += f"... (Showing only first {MAX_DISPLAY_ROWS} rows)"

        return output
    except Exception as e:
        return f"SQL Error: {e}"


def start_mcp_server(transport: Literal["sse", "stdio", "streamable-http"] = "streamable-http", **kwargs) -> None:
    """Start the MCP server with the given transport."""
    print(f"Starting MCP Server on http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}{MCP_SERVER_PATH}", flush=True)
    db_names = list(db_manager.get_all_providers().keys())
    print(f"Transport: {transport} | Databases: {', '.join(db_names)}", flush=True)
    print("Press Ctrl+C to stop...", flush=True)
    mcp.run(transport=transport, **kwargs)



if __name__ == "__main__":
    start_mcp_server(transport="streamable-http")