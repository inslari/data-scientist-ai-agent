from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
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
def get_column_stats() -> str:
    """
    Get per-column statistics for all tables.
    Numeric columns: min, max, avg, null count.
    Text columns: top 10 most frequent distinct values with counts.
    Use this to understand what each column actually contains before writing SQL.
    """
    try:
        if not DB_PATH.exists():
            return f"Error: Database not found at {DB_PATH}"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        output = "Column Statistics:\n"

        for table in tables:
            output += f"\nTable: {table}\n"
            cursor.execute(f"PRAGMA table_info('{table}')")
            cols = cursor.fetchall()

            # Total rows 
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            total_rows = cursor.fetchone()[0]
            output += f"  Total rows: {total_rows:,}\n"

            for col in cols:
                col_name = col[1]
                col_type = col[2].upper() if col[2] else ""
                is_numeric = any(t in col_type for t in
                                 ("INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE", "DECIMAL"))
                try:
                    # Null count
                    cursor.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL'
                    )
                    null_count = cursor.fetchone()[0]
                    null_pct   = round(null_count / total_rows * 100, 1) if total_rows else 0

                    if is_numeric:
                        cursor.execute(f"""
                            SELECT
                                MIN("{col_name}"),
                                MAX("{col_name}"),
                                ROUND(AVG("{col_name}"), 4),
                                COUNT(DISTINCT "{col_name}")
                            FROM "{table}"
                        """)
                        mn, mx, avg, n_distinct = cursor.fetchone()
                        output += (
                            f'  "{col_name}" [numeric]: '
                            f'min={mn}, max={mx}, avg={avg}, '
                            f'distinct={n_distinct}, nulls={null_count} ({null_pct}%)\n'
                        )
                    else:
                        # Top 10 most frequent values
                        cursor.execute(f"""
                            SELECT "{col_name}", COUNT(*) as cnt
                            FROM "{table}"
                            WHERE "{col_name}" IS NOT NULL
                            GROUP BY "{col_name}"
                            ORDER BY cnt DESC
                            LIMIT 10
                        """)
                        top_vals = cursor.fetchall()
                        vals_str = ", ".join(
                            f'"{v}"({c})' for v, c in top_vals
                        )
                        cursor.execute(
                            f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}"'
                        )
                        n_distinct = cursor.fetchone()[0]
                        output += (
                            f'  "{col_name}" [text]: '
                            f'distinct={n_distinct}, nulls={null_count} ({null_pct}%), '
                            f'top values=[{vals_str}]\n'
                        )
                except Exception as col_err:
                    output += f'  "{col_name}": error — {col_err}\n'

        conn.close()
        return output

    except Exception as e:
        return f"Error getting column stats: {e}"


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