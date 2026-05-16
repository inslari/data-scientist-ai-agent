from __future__ import annotations

import os
import re
import sqlite3
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv(override=True)

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "9001"))
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")
MCP_SERVER_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}{MCP_SERVER_PATH}"

from meta_db import init_db, parse_and_save_schema

async def call_mcp_tool(tool_name: str, params: dict) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        async with streamable_http_client(MCP_SERVER_URL, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, params)
                return result.content[0].text

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        schema_text = await call_mcp_tool("get_schema", {})
        parse_and_save_schema(schema_text)
        print("Schema successfully fetched from MCP and saved to API DB on startup.")
    except Exception as e:
        print(f"Warning: Failed to fetch schema from MCP on startup. Is MCP running? Error: {e}")
    yield

app = FastAPI(title="Data Scientist AI Agent API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str



@app.get("/")
def read_root() -> dict[str, object]:
    return {
        "status": "ok",
        "description": "FastAPI is running independently and can call an independent MCP server.",
        "mcp_server_url": MCP_SERVER_URL,
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "healthy", "mcp_server_url": MCP_SERVER_URL}


@app.get("/mcp/schema")
async def get_schema() -> dict[str, str]:
    try:
        schema_text = await call_mcp_tool("get_schema", {})
        return {"schema": schema_text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/mcp/query")
async def run_sql_query(request: QueryRequest) -> dict[str, str]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    try:
        query_result = await call_mcp_tool("run_sql_query", {"query": request.query})
        return {"result": query_result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
