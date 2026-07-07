"""Tools the expert agents use to investigate the dataset.

The single source-agnostic tool is ``run_sql``: it runs a read-only query through
``run_data_group_query_logic`` (api/schemas/data_group.py), which routes by the
data group's ``source_type`` — MCP groups hit the MCP server, CSV groups run on
an in-memory DuckDB over the registered CSV views. Both return the same
``Columns: …\\n(row)…`` text, so the agents are source-agnostic.

We deliberately do NOT use ``services.mcp_client.mcp_run_query`` here — that is
MCP-only and would break CSV data groups.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.core.sql_safety import validate_read_only

_MAX_RESULT_CHARS = 3000


def _is_read_only(sql: str) -> bool:
    ok, _ = validate_read_only(sql)
    return ok


def make_sql_tool(data_group_id: int, user_id: int):
    """Build an async ``run_sql`` tool bound to a specific data group/user.

    Works for both MCP and CSV data groups. Returns a LangChain tool suitable
    for ``create_agent``.
    """

    @tool
    async def run_sql(sql: str) -> str:
        """Run a read-only SQL SELECT query against THIS dataset and return the rows.

        Use this to investigate the data: compute counts, distributions, group-bys,
        min/max/avg, null rates, and class balance. Only SELECT / WITH queries are
        allowed. Prefer aggregate queries and always add a LIMIT when selecting raw rows.
        """
        ok, reason = validate_read_only(sql)
        if not ok:
            return f"ERROR: {reason}"

        # Imported lazily to avoid import cycles at module load.
        from api.core.database import SessionLocal
        from api.schemas.data_group import run_data_group_query_logic, QueryRequest

        db = SessionLocal()
        try:
            res = await run_data_group_query_logic(
                data_group_id, QueryRequest(query=sql), db, user_id
            )
        except Exception as e:  # pragma: no cover - defensive
            return f"ERROR running query: {e}"
        finally:
            db.close()

        if res.get("status") != "success":
            return f"ERROR: {res.get('result', 'query failed')}"

        text = str(res.get("result", "")).strip()
        if len(text) > _MAX_RESULT_CHARS:
            text = text[:_MAX_RESULT_CHARS] + "\n… (truncated)"
        return text or "Query returned no rows."

    return run_sql
