from __future__ import annotations

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.intelligence_engine import DatabaseIntelligenceEngine
from src.state import AgentState

MCP_CONFIG = {
    "sqlite": {
        "url": "http://127.0.0.1:9001/mcp",
        "transport": "streamable_http",
    }
}


async def _fetch_schema_and_samples() -> tuple[str, str]:
    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}

    schema_result = await tool_map["get_schema"].ainvoke({})
    raw_schema = schema_result if isinstance(schema_result, str) else str(schema_result)

    table_names = []
    for line in raw_schema.splitlines():
        if line.strip().startswith("- Table '"):
            try:
                name = line.split("'")[1]
                table_names.append(name)
            except IndexError:
                continue

    sample_parts = []
    for table in table_names[:5]:
        try:
            sample_result = await tool_map["run_sql_query"].ainvoke(
                {"query": f"SELECT * FROM {table} LIMIT 3"}
            )
            sample_parts.append(f"Table: {table}\n{sample_result}")
        except Exception:
            pass

    sample_data = "\n\n".join(sample_parts) if sample_parts else "No sample data."
    return raw_schema, sample_data


def db_context_node(state: AgentState) -> AgentState:
    raw_schema, sample_data = asyncio.run(_fetch_schema_and_samples())

    engine = DatabaseIntelligenceEngine()
    db_model = engine.analyze_schema(
        schema_text=raw_schema,
        sample_data=sample_data,
    )

    intent = state.get("intent", "eda")
    relevant_tables = _filter_tables_by_intent(db_model, intent)

    return {
        **state,
        "raw_schema": raw_schema,
        "db_intelligence": db_model.model_dump(),
        "relevant_tables": relevant_tables,
        "current_step": "db_context_node",
    }


def _filter_tables_by_intent(db_model, intent: str) -> list[str]:
    scored = sorted(
        db_model.tables,
        key=lambda t: t.analytics_potential_score,
        reverse=True,
    )
    if intent in ("eda", "report"):
        return [t.table_name for t in scored]
    else:
        return [t.table_name for t in scored[:3]]