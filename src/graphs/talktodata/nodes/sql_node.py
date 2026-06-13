import ast
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from src.core.db_provider import db_manager
from api.core.database import SessionLocal
from api.schemas.data_group import (
    run_data_group_query_logic as run_data_group_query,
    QueryRequest as DGQueryRequest,
)

from ..state import TTDState, _SQLGenerationResult
from ..utils import _get_llm, _build_column_profiles, _StructuredJsonParser

TTD_SQL_SYSTEM = """\
You are a SQL database expert. Available databases: {db_providers}.
Your job is to understand the user's question and generate an appropriate SELECT query along with the target database name (db_name) to run it on.

The raw database schema and semantic column profiles are provided below.

=== REQUIRED: Apply these 3 steps in order for every query ===

STEP 1 — Determine exactly what the user is asking for.
  Clarify what data type and unit the concept should be in.
  Example: "revenue" → numeric, currency | "age" → numeric, years | "city" → text

STEP 2 — Is there a column in the schema that TRULY matches this concept?
  Evaluate every column using the provided {schema} and {column_profiles}.
  A column is only considered suitable if ALL of the following hold:
    a) The column name or description DIRECTLY matches the user's concept semantically
    b) The column's actual values (min, max, avg, sample values) are CONSISTENT
       with the expected meaning and unit (e.g. "revenue" requires real monetary numeric values)
    c) The data type is appropriate (numeric concept → numeric column; text concept → text column)

  COMMON MISLEADING SITUATIONS — these are NOT suitable columns:
    - Unit mismatch: If the user wants a monetary value, a column containing hours/counts/ratios is NOT suitable
    - Category vs number: If the user wants a real numeric value, a column with labels/ranges is NOT suitable
      (e.g. "low/medium/high", "<X", ">Y" groupings are not numeric values)
    - Name trap: A keyword in the column name is NOT enough — the actual content must also match
    - Proxy column: Do not pick the "closest" or "most related" column — if there's no direct match, it's NOT suitable

STEP 3 — Choose one of three cases:

  [CASE A] Exact match found:
    -> Write a normal SELECT query
    -> is_approximation = false, approximation_warning = "", data_warning = ""

  [CASE B] No exact match, but a meaningful approximation is possible:
    -> Write a query that answers the topic indirectly using available columns
       (e.g. ratio, distribution, category breakdown)
    -> is_approximation = true
    -> approximation_warning: "This dataset has no direct column for [requested concept].
       Instead, [used column/approach] is used to show [what was calculated]."
    -> data_warning = ""

  [CASE C] No meaningful alternative exists:
    -> sql = SELECT 'NO_VALID_COLUMN' AS warning
    -> data_warning = "No suitable column found for [concept] in this dataset.
                       Available columns: [...]. Alternative: [possible query suggestion]"
    -> is_approximation = false, approximation_warning = ""

=== General Rules ===
- Only write SELECT queries
- Use GROUP BY for aggregations. If returning raw, unaggregated data, ALWAYS append LIMIT 20.
- Unless the user specifies otherwise, ALWAYS order the results to show the most recent data first (find an appropriate date/time/timestamp column and use ORDER BY ... DESC).
- Use column names exactly as they appear in the schema (case-sensitive)
- Always wrap column names containing dots, spaces, or special characters in double quotes
- If the question is ambiguous, apply the most logical interpretation
- Write SQL in English regardless of the question language
- CRITICAL: You must directly return the structured output. DO NOT attempt to call any external tools or hallucinate tool names like 'select_column_from_table' or 'search'. You do NOT have web access. ONLY output the valid JSON matching the schema.

--- Chat History Context ---
{chat_history}

--- Long-Term Memory / User Preferences ---
{long_term_memory}

--- Raw Schema ---
{schema}

--- Column Semantic Profiles ---
{column_profiles}

{validation_fix_hint}
"""

TTD_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TTD_SQL_SYSTEM),
    ("human", "User question: {user_query}{sql_error_msg}"),
])

async def ttd_sql_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    LLM → SQL üretir → MCP'de çalıştırır → LIMIT/ORDER BY uygular.
    """
    print("\n[TTD] ttd_sql_node başlıyor...")

    data_group_id = config.get("configurable", {}).get("data_group_id") or state.get("data_group_id")
    if not data_group_id:
        return {**state, "error": "data_group_id missing", "current_step": "ttd_sql_node"}  # type: ignore[return-value]

    llm             = _get_llm(temperature=0)
    user_query      = state["user_query"]
    db_intelligence = state.get("db_intelligence") or {}
    raw_schema      = state.get("raw_schema") or "Schema not available."
    column_profiles = _build_column_profiles(db_intelligence)

    # Format context
    chat_history_str = ""
    for m in state.get("messages") or []:
        chat_history_str += f"{m.get('role', 'user').upper()}: {m.get('content', '')}\n"
    if not chat_history_str:
        chat_history_str = "No previous messages."

    memory_str = state.get("long_term_memory") or "No specific user preferences saved yet."

    sql_error_msg = ""
    if state.get("sql_error"):
        sql_error_msg = (
            f"\n\n[URGENT] Your previous query failed to execute.\n"
            f"Failed Query: {state.get('sql_query_raw')}\n"
            f"Error: {state['sql_error']}\n\n"
            f"Please analyze the error and correct the query."
        )

    # Validator'dan gelen düzeltme ipucunu ekle
    validation_fix_hint = ""
    if state.get("validation_fix_hint"):
        validation_fix_hint = (
            f"\n\n[VALIDATOR FEEDBACK] The previous query result was rejected:\n"
            f"Issue: {state.get('validation_issue')}\n"
            f"Suggested fix: {state['validation_fix_hint']}\n"
            f"Please rewrite the query addressing this issue."
        )

    # SQL generation
    try:
        db_providers_info = ", ".join([
            f"{k} ({v.provider_display_name})"
            for k, v in db_manager.get_all_providers().items()
        ])
        invoke_kwargs = {
            "schema":              raw_schema,
            "column_profiles":     column_profiles,
            "user_query":          user_query,
            "db_providers":        db_providers_info,
            "sql_error_msg":       sql_error_msg,
            "chat_history":        chat_history_str,
            "long_term_memory":    memory_str,
            "validation_fix_hint": validation_fix_hint,
        }
        sql_result: _SQLGenerationResult
        # Strategy 1: structured output (tool-call based — most reliable when supported)
        try:
            sql_chain_structured = TTD_SQL_PROMPT | llm.with_structured_output(_SQLGenerationResult)
            sql_result = sql_chain_structured.invoke(invoke_kwargs)  # type: ignore[assignment]
        except Exception as e1:
            print(f"[TTD] with_structured_output failed ({e1}), falling back to text parser")
            # Strategy 2: text-based parser — handles plain text / markdown JSON responses
            sql_chain_text = TTD_SQL_PROMPT | llm | _StructuredJsonParser(pydantic_object=_SQLGenerationResult)
            sql_result = sql_chain_text.invoke(invoke_kwargs)  # type: ignore[assignment]
        sql            = sql_result.sql.strip()
        data_warning   = sql_result.data_warning or ""
        is_approx      = sql_result.is_approximation
        approx_warning = sql_result.approximation_warning or ""
    except Exception as e:
        retry_count = state.get("sql_retry_count") or 0
        print(f"[TTD] SQL generation failed (retry {retry_count}): {e}")
        return {
            **state,
            "sql_error": f"SQL generation error: {e}",
            "sql_retry_count": retry_count + 1,
            "current_step": "ttd_sql_node"
        }  # type: ignore[return-value]

    # NO_VALID_COLUMN guard
    if "NO_VALID_COLUMN" in sql:
        msg = data_warning or "No suitable column found for your query."
        return {
            **state,
            "sql_query_raw":     sql,
            "sql_query_columns": [],
            "sql_query_results": [],
            "sql_query_chart":   {"type": "warning", "title": "No Valid Column", "summary": msg},
            "sql_query_summary": msg,
            "sql_query_warning": "",
            "current_step":      "ttd_sql_node",
        }  # type: ignore[return-value]

    # Execute query
    try:
        db = SessionLocal()
        try:
            req = DGQueryRequest(query=sql)
            res = await run_data_group_query(data_group_id, req, db)

            if res.get("status") == "error":
                raise Exception(res.get("result"))

            raw_txt = str(res.get("result", "")) if res.get("status") == "success" else ""

            lines   = raw_txt.strip().splitlines()
            columns: list[str] = []
            rows:    list      = []

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
        retry_count = state.get("sql_retry_count") or 0
        print(f"[TTD] SQL execution failed (retry {retry_count}): {e}")
        return {
            **state,
            "sql_error": str(e),
            "sql_retry_count": retry_count + 1,
            "sql_query_raw": sql,
            "current_step": "ttd_sql_node"
        }  # type: ignore[return-value]

    final_warning    = approx_warning if is_approx else ""
    results_as_dicts = [dict(zip(columns, row)) for row in rows]

    return {
        **state,
        "sql_error":            None,
        "validation_fix_hint":  None,  # clear previous fix hint on success
        "sql_query_raw":        sql,
        "sql_query_columns":    columns,
        "sql_query_results":    results_as_dicts,
        "sql_query_warning":    final_warning,
        "current_step":         "ttd_sql_node",
        "sql_query_chart":      {},
        "sql_query_summary":    "",
        "ttd_raw_rows":          rows,
    }  # type: ignore[return-value]
