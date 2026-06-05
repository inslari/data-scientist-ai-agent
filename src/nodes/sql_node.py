import json
from ..state import AgentState
from ..core.db_provider import db_manager
from ..core.llm import get_llm
from ..core.config import settings
from ..core.prompt_loader import create_chat_prompt
from ..schemas.sql_schemas import SQLGenerationResult, ChartConfig
from langchain_core.runnables import RunnableConfig
from api.core.database import SessionLocal
from api.routes.data_group import run_data_group_query, QueryRequest
import ast
from ..services.chart_utils import build_chart_payload


def _build_column_profiles(db_intelligence: dict) -> str:
    """
    state['db_intelligence'] dict'inden kolon semantik profillerini
    LLM'in okuyabileceği düz metin formatına çevirir.
    """
    columns = db_intelligence.get("columns", [])
    if not columns:
        return "Kolon profili mevcut değil."

    lines = []
    for col in columns:
        table       = col.get("table_name", "?")
        name        = col.get("column_name", "?")
        dtype       = col.get("data_type", "?")
        sem_type    = col.get("semantic_type", "?")
        importance  = col.get("importance_score", 0)
        nullability = col.get("nullability", "?")
        uniqueness  = col.get("uniqueness", "?")
        description = col.get("description", "")
        sample_vals = col.get("sample_values", "")

        line = (
            f'  - {table}."{name}" | type={dtype} | semantic={sem_type} '
            f'| importance={importance:.2f} | null={nullability} | unique={uniqueness}'
        )
        if description:
            line += f'\n      description: {description}'
        if sample_vals:
            line += f'\n      sample_values: {sample_vals}'
        lines.append(line)

    global_insights = db_intelligence.get("global_insights", [])
    header = "Global insights:\n" + "\n".join(f"  * {g}" for g in global_insights) + "\n\n" if global_insights else ""
    return header + "Columns:\n" + "\n".join(lines)


async def sql_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    SQL Query Node:
      1. db_intelligence'ı state'den okur
      2. LLM ile kolon profillerini kullanarak SQL üretir
      3. MCP'de sorguyu çalıştırır
      4. LLM ile chart konfigürasyonu belirler
      5. State'e yazar
    """
    print("\n========================================\n[NODE START] sql_node başlıyor...\n========================================")
    llm        = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
    user_query = state["user_query"]
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        return {**state, "error": "data_group_id is missing", "current_step": "sql_node"}


    # db_intelligence taken from db_context_node
    db_intelligence  = state.get("db_intelligence", {})
    raw_schema       = state.get("raw_schema", "Schema not available.")
    column_profiles  = _build_column_profiles(db_intelligence)

    try:
        db_providers_info = ", ".join([f"{k} ({v.provider_display_name})" for k, v in db_manager.get_all_providers().items()])
        from src.agents.sql_agent import sql_agent
        
        prompt_content = f"""
        User Query: {user_query}
        
        Schema Context:
        {raw_schema}
        
        Column Profiles:
        {column_profiles}
        
        DB Providers Available:
        {db_providers_info}
        """
        
        agent_result = sql_agent.invoke(
            {"messages": [{"role": "user", "content": prompt_content}]},
            config=config
        )
        
        sql = ""
        target_db_name = "default"
        data_warning = ""
        is_approximation = False
        approximation_warning = ""
        
        for msg in reversed(agent_result["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == "submit_final_sql":
                        args = tool_call["args"]
                        sql = args.get("sql_query", "").strip()
                        target_db_name = args.get("db_name", "default").strip()
                        data_warning = args.get("data_warning", "")
                        is_approximation = args.get("is_approximation", False)
                        approximation_warning = args.get("approximation_warning", "")
                        break
            if sql:
                break

        if not sql:
             raise ValueError("Agent did not submit a final SQL query via tool.")
             
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {**state, "error": f"SQL üretim hatası: {e}", "current_step": "sql_node"}

    # If no valid columns found, return warning
    if "NO_VALID_COLUMN" in sql:
        warning_msg = data_warning or "Bu veri setinde isteğinize uygun kolon bulunamadı."
        return {
            **state,
            "sql_query_raw":     sql,
            "sql_query_columns": [],
            "sql_query_results": [],
            "sql_query_chart": {
                "type":    "warning",
                "title":   "Uygun Kolon Bulunamadı",
                "summary": warning_msg,
            },
            "sql_query_summary": warning_msg,
            "current_step":      "sql_node",
        }

    # 4. Execute the query
    try:
        db = SessionLocal()
        try:
            req = QueryRequest(query=sql)
            res = await run_data_group_query(data_group_id, req, db)
            raw_text = str(res.get("result", "")) if res.get("status") == "success" else ""
            
            lines = raw_text.strip().splitlines()
            columns = []
            rows = []
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
        finally:
            db.close()
    except Exception as e:
        return {**state, "error": f"SQL çalıştırma hatası: {e}", "current_step": "sql_node"}

    if not rows:
        return {
            **state,
            "sql_query_raw":     sql,
            "sql_query_columns": columns,
            "sql_query_results": [],
            "sql_query_chart":   {
                "type": "table", "columns": columns, "rows": [],
                "title": "Sonuç bulunamadı", "summary": "Sorgu sonuç döndürmedi.",
            },
            "sql_query_summary": "Sorgu sonuç döndürmedi.",
            "current_step":      "sql_node",
        }

    # 4. Chart configuration
    sample_rows = rows[:5]
    try:
        CHART_CFG_PROMPT = create_chat_prompt("chart_configuration.md", "Grafik konfigürasyonunu döndür.")
        chart_chain = CHART_CFG_PROMPT | llm.with_structured_output(ChartConfig)
        chart_cfg: ChartConfig = chart_chain.invoke({  # type: ignore[assignment]
            "columns":     ", ".join(columns),
            "sample_rows": json.dumps([list(r) for r in sample_rows], ensure_ascii=False),
            "user_query":  user_query,
        })
    except Exception as e:
        chart_cfg = ChartConfig(
            type="table",
            x=columns[0] if columns else "",
            y="",
            title="Sorgu Sonuçları"
        )

    # 5. Payload + state
    chart_payload    = build_chart_payload(chart_cfg.model_dump(), rows, columns)
    results_as_dicts = [dict(zip(columns, row)) for row in rows]

    return {
        **state,
        "sql_query_raw":     sql,
        "sql_query_columns": columns,
        "sql_query_results": results_as_dicts,
        "sql_query_chart":   chart_payload or {},
        "sql_query_summary": "",
        "sql_query_warning": approximation_warning if is_approximation else "",
        "current_step":      "sql_node",
    }