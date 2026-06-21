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

from src.core.db_provider import db_manager  # type: ignore[import-not-found]
from src.core.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv(override=True)

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "9001"))
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")

mcp = FastMCP(
    "Database MCP Server",
    host=MCP_SERVER_HOST,
    port=MCP_SERVER_PORT,
    streamable_http_path=MCP_SERVER_PATH,
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
def get_column_stats() -> str:
    """
    Get per-column statistics for all tables.
    Numeric columns: min, max, avg, null count.
    Text columns: top 10 most frequent distinct values with counts.
    Use this to understand what each column actually contains before writing SQL.
    """
    try:
        output = "Column Statistics:\n"
        for db_name, provider in db_manager.get_all_providers().items():
            output += f"\n=== Database: '{db_name}' ===\n"
            with provider.connect() as conn:
                # Get table names
                tables_result = conn.execute(text(provider.get_table_names_query()))
                tables = [r[0] for r in tables_result]

                for table in tables:
                    output += f"\nTable: {table}\n"

                    # Get column info
                    col_result = conn.execute(
                        text(provider.get_column_info_query(table))
                    )
                    cols = col_result.fetchall()

                    # Total rows
                    count_result = conn.execute(
                        text(provider.get_count_query(table))
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
                                text(provider.get_null_count_query(table, col_name))
                            )
                            null_row = null_result.fetchone()
                            null_count = null_row[0] if null_row else 0
                            null_pct = round(null_count / total_rows * 100, 1) if total_rows else 0

                            if is_numeric:
                                stats_result = conn.execute(
                                    text(provider.get_column_stats_numeric_query(table, col_name))
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
                                    text(provider.get_column_stats_text_top_values_query(table, col_name))
                                )
                                top_vals = top_result.fetchall()
                                vals_str = ", ".join(
                                    f'"{v}"({c})' for v, c in top_vals
                                )
                                distinct_result = conn.execute(
                                    text(provider.get_distinct_count_query(table, col_name))
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
    logger.info(f"Starting MCP Server on http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}{MCP_SERVER_PATH}")
    db_names = list(db_manager.get_all_providers().keys())
    logger.info(f"Transport: {transport} | Databases: {', '.join(db_names)}")
    logger.info("Press Ctrl+C to stop...")
    mcp.run(transport=transport, **kwargs)



if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging(process="mcp")
    start_mcp_server(transport="streamable-http")