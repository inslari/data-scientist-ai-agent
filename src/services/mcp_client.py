import os
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_PORT = os.getenv("MCP_SERVER_PORT", "9001")
MCP_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")
MCP_URL  = f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}"

async def _async_run_query(sql: str, db_name: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("run_sql_query", {"query": sql, "db_name": db_name})
            raw_text = getattr(result.content[0], "text", "") if result.content else ""

    lines   = raw_text.strip().splitlines()
    columns: list[str]              = []
    rows:    list[tuple[Any, ...]]  = []

    for i, line in enumerate(lines):
        if line.startswith("Columns:"):
            columns = [c.strip() for c in line.replace("Columns:", "").split(",")]
        elif line.startswith("Results:") or line.startswith("("):
            data_lines = lines[i + 1:] if line.startswith("Results:") else lines[i:]
            for dl in data_lines:
                dl = dl.strip()
                if not dl or dl.startswith("..."):
                    continue
                if dl.startswith("(") and dl.endswith(")"):
                    try:
                        rows.append(ast.literal_eval(dl))
                    except Exception:
                        pass
            break

    return columns, rows

def _run_async(coro):
    """Streamlit'in sync context'inden async coroutine çalıştır."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

def mcp_run_query(sql: str, db_name: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Senkron (blocking) MCP çağrısı."""
    return _run_async(_async_run_query(sql, db_name))