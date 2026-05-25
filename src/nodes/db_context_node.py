from __future__ import annotations

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.intelligence_engine import DatabaseIntelligenceEngine  # type: ignore[import-not-found]
from src.state import AgentState  # type: ignore[import-not-found]

MCP_URL = "http://127.0.0.1:9001/mcp"


async def db_context_node(state: AgentState) -> AgentState:
    """Async node — LangGraph ainvoke ile uyumlu."""
    
    # Fetch schema + sample data
    raw_schema, sample_data = await _fetch_schema_and_samples()

    engine   = DatabaseIntelligenceEngine()
    db_model = engine.analyze_schema(
        schema_text=raw_schema,
        sample_data=sample_data,
    )

    intent          = state.get("intent", "eda")
    relevant_tables = _filter_tables_by_intent(db_model, intent)

    return {
        **state,
        "raw_schema":      raw_schema,
        "db_intelligence": db_model.model_dump(),
        "relevant_tables": relevant_tables,
        "current_step":    "db_context_node",
    }


async def _fetch_schema_and_samples() -> tuple[str, str]:
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result     = await session.call_tool("get_schema", {})
            raw_schema = getattr(result.content[0], "text", "") if result.content else ""

            table_info = []
            current_db = "default"
            for line in raw_schema.splitlines():
                if line.strip().startswith("Database: '"):
                    try:
                        current_db = line.split("'")[1]
                    except IndexError:
                        continue
                elif line.strip().startswith("- Table '"):
                    try:
                        table_name = line.split("'")[1]
                        table_info.append((current_db, table_name))
                    except IndexError:
                        continue

            sample_parts = []
            for db_name, table in table_info[:5]:
                try:
                    r = await session.call_tool(
                        "run_sql_query",
                        {"query": f'SELECT * FROM "{table}" LIMIT 3', "db_name": db_name}
                    )
                    r_text = getattr(r.content[0], "text", "") if r.content else ""
                    sample_parts.append(f"Database: {db_name} | Table: {table}\n{r_text}")
                except Exception:
                    pass

            sample_data = "\n\n".join(sample_parts) if sample_parts else "No sample data."
            return raw_schema, sample_data


def _filter_tables_by_intent(db_model, intent: str) -> list[str]:
    scored = sorted(
        db_model.tables,
        key=lambda t: t.analytics_potential_score,
        reverse=True,
    )
    if intent in ("eda", "report"):
        return [t.table_name for t in scored]
    return [t.table_name for t in scored[:3]]