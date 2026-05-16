from mcp.server.fastmcp import FastMCP
import sqlite3
import os

mcp = FastMCP("SQLite MCP Server")

DB_PATH = "data/ecommerce.db"

@mcp.tool()
def get_schema() -> str:
    """Get the schema of all tables in the SQLite database. Use this to understand what columns exist."""
    try:
        if not os.path.exists(DB_PATH):
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
        for row in results[:50]: # Limit
            output += str(row) + "\n"
            
        if len(results) > 50:
            output += "... (Showing only first 50 rows)"
            
        return output
    except Exception as e:
        return f"SQL Error: {e}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
