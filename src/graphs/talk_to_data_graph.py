"""
talk_to_data_graph.py
─────────────────────
Talk to Data için bağımsız LangGraph pipeline.

Mevcut graph.py (ML pipeline) ile HİÇBİR bağlantısı yoktur.
Gerekli kodlar (node mantığı) doğrudan bu dosyada kopyalanmıştır.

Flow:
  START
    └─ ttd_db_context_node     (schema cache kontrolü → db_intelligence üret)
         └─ ttd_intent_classifier_node  (soruyu sınıflandır: chitchat/schema/sql/analytical)
              ├─ [chitchat / schema_question] → ttd_interpreter_node → END
              └─ [sql_query / analytical]
                   └─ ttd_sql_node     (SQL üret → MCP'de çalıştır)
                        └─ ttd_result_validator_node  (sonuçları doğrula)
                             ├─ [invalid / retry] → ttd_sql_node (max 3x)
                             └─ [valid]
                                  └─ ttd_chart_node   (grafik konfigürasyonu)
                                       └─ ttd_interpreter_node  (LLM veriyi yorumlar)
                                            └─ END
"""

from __future__ import annotations

import ast
import json
import re
from typing import Optional, Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

# Project imports
from src.core.llm import get_llm
from src.core.config import settings
from src.core.db_provider import db_manager
from src.intelligence_engine import DatabaseIntelligenceEngine
from api.core.database import SessionLocal
from api.schemas.data_group import (
    get_data_group_schema_logic as get_data_group_schema,
    run_data_group_query_logic as run_data_group_query,
    QueryRequest as DGQueryRequest,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. State
# ─────────────────────────────────────────────────────────────────────────────

class TTDState(TypedDict):
    # Input
    user_query: str
    data_group_id: Optional[int]
    messages: Optional[list]
    long_term_memory: Optional[str]

    # Schema / DB intelligence (cache'den veya canlı)
    raw_schema: Optional[str]
    db_intelligence: Optional[dict]

    # Intent classification
    intent: Optional[str]                    # "chitchat" | "schema_question" | "sql_query" | "analytical"
    needs_multiple_queries: Optional[bool]
    clarification_needed: Optional[bool]
    clarification_question: Optional[str]

    # SQL node output
    sql_query_raw: Optional[str]
    sql_query_columns: Optional[list]
    sql_query_results: Optional[list]
    sql_query_warning: Optional[str]

    # Validator node output
    validation_passed: Optional[bool]
    validation_issue: Optional[str]          # "empty_result" | "wrong_columns" | "too_few_rows" | None
    validation_fix_hint: Optional[str]       # Validator'ın SQL node'a gönderdiği düzeltme ipucu

    # Chart node output
    sql_query_chart: Optional[dict]

    # Interpreter output
    sql_query_summary: Optional[str]

    # Control
    current_step: Optional[str]
    error: Optional[str]
    sql_error: Optional[str]
    sql_retry_count: Optional[int]
    ttd_raw_rows: Optional[list]  # temporary: raw tuple rows passed from sql_node to chart_node


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prompt constants
# ─────────────────────────────────────────────────────────────────────────────

TTD_INTENT_SYSTEM = """\
You are an intent classifier for a data analytics assistant.

The user has sent a message. Your job is to classify what kind of response is needed.

Available intent labels:
- "chitchat"         → Greetings, small talk, questions unrelated to data (e.g. "hello", "thank you", "how are you?")
- "schema_question"  → User asks about the structure, tables, or columns of the database (e.g. "what tables do you have?", "what does the income column mean?")
- "sql_query"        → User wants to query, count, filter, or aggregate data from the database (e.g. "show me top 10 occupations", "how many records are there?")
- "analytical"       → User wants insights, relationships, correlations, or trends that may require complex queries (e.g. "what is the relationship between age and income?")

--- Database Schema (for context) ---
{raw_schema}

--- Chat History ---
{chat_history}

--- User Message ---
{user_query}

Classification rules:
- If the user says hello/thanks/goodbye → "chitchat"
- If the user asks ABOUT the database structure → "schema_question"
- If the user asks for data retrieval or aggregation → "sql_query"
- If the user asks for patterns, correlations, or comparisons → "analytical"
- When uncertain between sql_query and analytical, prefer "sql_query"
- needs_multiple_queries: true ONLY if the question clearly requires 2+ separate SQL queries
- clarification_needed: true ONLY if the question is too vague to answer without asking the user
"""

TTD_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TTD_INTENT_SYSTEM),
    ("human", "Classify the intent of the user message above. Return structured output."),
])


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


TTD_VALIDATOR_SYSTEM = """\
You are a SQL result validator for a data analytics assistant.

The user asked a question, SQL was executed, and results came back.
Your job is to evaluate whether the results actually answer the user's question.

User question: {user_query}
Executed SQL: {sql}
Result columns: {columns}
Row count: {row_count}
Sample rows (first 5): {sample_rows}

Validation rules:
- is_valid = false if:
    * row_count = 0 and the question expects data to exist
    * columns returned have no relation to what the user asked
    * results are obviously wrong (e.g. user asked for names but got numbers)
- is_valid = true if results reasonably answer the question (even if approximate)
- issue must be one of: "empty_result", "wrong_columns", "too_few_rows", or null
- suggested_fix: a SHORT hint for the SQL generator on how to fix the query (e.g. "Remove the WHERE clause — it may be filtering too aggressively")
  If is_valid = true, suggested_fix must be null.
"""

TTD_VALIDATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TTD_VALIDATOR_SYSTEM),
    ("human", "Validate the SQL results above. Return structured output."),
])


TTD_CHART_SYSTEM = """\
You are a data visualization expert. Look at the SQL results and choose the most meaningful chart type.

Chart type guide:
- Categorical column + numeric column → bar
- Time series / ordered data + numeric → line
- Ratio / share distribution (≤8 categories) → pie
- Relationship between two numeric columns → scatter
- Multiple columns / raw tabular data → table

Column list: {columns}
First 5 rows: {sample_rows}
User question: {user_query}
"""

TTD_CHART_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TTD_CHART_SYSTEM),
    ("human", "Return the chart configuration."),
])


TTD_INTERPRETER_SYSTEM = """\
You are a Data Scientist and Analysis Expert. The user has asked a question, and data has been fetched from the database to answer it.

Your task: Review the fetched data and provide a **natural, clear, and direct response** to the user's question.

Rules:
- Only use the provided data to make interpretations. Do not hallucinate or make up information not present in the data.
- The output should only contain the interpretation text. Do not use unnecessary introductions (like "Hello, here are the results:").
- Format numerical values to be easily readable (e.g., write 1,000 instead of 1000).
- Highlight the main summaries and notable details. Do not read out every single row line by line (the user will already see the table on their screen).
- Keep the response short, concise, and focused.

---
Additionally, you have access to a "Long-Term Memory". If the user expresses a general preference (e.g., "always show me USD", "I prefer line charts", "don't show me empty rows"), update the long-term memory string. If there are no new preferences, return the existing memory exactly as it is.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0):
    """Returns a configured LLM instance."""
    return get_llm(
        provider=settings.INTELLIGENCE_ENGINE_PROVIDER,
        model_name=settings.INTELLIGENCE_ENGINE_MODEL,
        temperature=temperature
    )


def _build_chart_payload(chart_cfg: dict, rows: list, columns: list) -> dict | None:
    """chart_utils.build_chart_payload — kopyalanmış."""
    try:
        ctype  = chart_cfg.get("type", "bar").lower()
        x_col  = chart_cfg.get("x")
        y_col  = chart_cfg.get("y")
        title  = chart_cfg.get("title", "Query Results")

        if not x_col or not y_col:
            return None
        if x_col not in columns or y_col not in columns:
            return None

        x_idx = columns.index(x_col)
        y_idx = columns.index(y_col)

        x_data = [str(r[x_idx]) for r in rows]
        y_data = [float(r[y_idx]) if r[y_idx] is not None else 0.0 for r in rows]

        if ctype == "pie":
            option = {
                "title":   {"text": title, "left": "center"},
                "tooltip": {"trigger": "item"},
                "series":  [{"type": "pie", "radius": "50%",
                              "data": [{"name": x_data[i], "value": y_data[i]} for i in range(len(x_data))]}],
            }
        else:
            option = {
                "title":   {"text": title, "left": "center"},
                "tooltip": {"trigger": "axis"},
                "xAxis":   {"type": "category", "data": x_data, "name": x_col},
                "yAxis":   {"type": "value",    "name": y_col},
                "series":  [{"data": y_data, "type": ctype}],
            }
        return option
    except Exception as e:
        print(f"[ttd chart] build error: {e}")
        return None


def _build_column_profiles(db_intelligence: dict) -> str:
    columns = db_intelligence.get("columns", [])
    if not columns:
        return "No column profiles available."
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class _IntentResult(BaseModel):
    intent: str = Field(description="One of: chitchat, schema_question, sql_query, analytical")
    needs_multiple_queries: bool = Field(default=False, description="True if the question requires 2+ separate SQL queries")
    clarification_needed: bool = Field(default=False, description="True if the question is too vague to answer without asking")
    clarification_question: Optional[str] = Field(default=None, description="The clarification question to ask the user, if needed")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"intent", "needs_multiple_queries", "clarification_needed", "clarification_question"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class _ValidationResult(BaseModel):
    is_valid: bool = Field(description="True if the SQL results adequately answer the user's question")
    issue: Optional[str] = Field(default=None, description="One of: empty_result, wrong_columns, too_few_rows, or null")
    suggested_fix: Optional[str] = Field(default=None, description="A short hint for the SQL generator on how to fix the query")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"is_valid", "issue", "suggested_fix"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class _SQLGenerationResult(BaseModel):
    sql: str = Field(description="SQL query")
    db_name: str = Field(description="Target database/connection name")
    data_warning: Optional[str] = Field(default="", description="Warning if no valid column found")
    is_approximation: bool = Field(default=False)
    approximation_warning: Optional[str] = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, values: Any) -> Any:
        """Accept 'query' as an alias for 'sql' in case the LLM uses the wrong key."""
        if isinstance(values, dict):
            if "sql" not in values and "query" in values:
                values["sql"] = values.pop("query")
            allowed = {"sql", "db_name", "data_warning", "is_approximation", "approximation_warning"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class _ChartConfig(BaseModel):
    type: str  = Field(description="Chart type: bar, line, pie, scatter, table")
    x: str     = Field(description="X-axis column name")
    y: str     = Field(description="Y-axis column name")
    title: str = Field(description="Chart title")

    @model_validator(mode="before")
    @classmethod
    def _drop_extra(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"type", "x", "y", "title"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class _InterpreterResult(BaseModel):
    summary: str = Field(description="The natural language interpretation of the data.")
    updated_long_term_memory: Optional[str] = Field(default=None, description="The updated long-term memory string, keeping previous preferences and adding new ones if any.")

    @model_validator(mode="before")
    @classmethod
    def _drop_extra(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"summary", "updated_long_term_memory"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class _StructuredJsonParser(BaseOutputParser):
    """
    Robust parser that handles LLMs which:
    - Return raw text instead of JSON
    - Wrap JSON in markdown code fences
    - Return plain text (handled via fallback_field)
    - Use wrong field names (handled by model_validators above)
    """
    pydantic_object: type
    fallback_field: Optional[str] = None  # If set, plain-text responses are wrapped as {fallback_field: text}

    class Config:
        arbitrary_types_allowed = True

    def parse(self, text: str):
        text = text.strip()
        # Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        # Try to find a JSON object in the text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self.pydantic_object.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass  # Fall through to plain-text handling
        # No JSON found — use fallback_field if configured
        if self.fallback_field:
            return self.pydantic_object.model_validate({self.fallback_field: text})
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")

    def get_format_instructions(self) -> str:
        from langchain_core.output_parsers import PydanticOutputParser
        return PydanticOutputParser(pydantic_object=self.pydantic_object).get_format_instructions()


def _parse_with_fallback(llm, prompt_value, pydantic_cls, fallback_field: Optional[str] = None):
    """
    Try with_structured_output first, fall back to _StructuredJsonParser.
    Returns an instance of pydantic_cls.
    """
    try:
        return (prompt_value | llm.with_structured_output(pydantic_cls)).invoke({})
    except Exception as e1:
        print(f"[TTD] with_structured_output failed ({type(e1).__name__}), trying text parser")
        raw_text = llm.invoke(prompt_value).content
        return _StructuredJsonParser(
            pydantic_object=pydantic_cls,
            fallback_field=fallback_field,
        ).parse(raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Nodes
# ─────────────────────────────────────────────────────────────────────────────

async def ttd_db_context_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    Schema + db_intelligence üretir.
    State'te zaten db_intelligence varsa (cache hit) atlar.
    """
    print("\n[TTD] ttd_db_context_node başlıyor...")

    if state.get("db_intelligence") and state.get("raw_schema"):
        print("[TTD] db_context: cache hit, skipping LLM analysis")
        return {**state, "current_step": "ttd_db_context_node"}  # type: ignore[return-value]

    data_group_id = config.get("configurable", {}).get("data_group_id") or state.get("data_group_id")
    if not data_group_id:
        return {**state, "error": "data_group_id missing", "current_step": "ttd_db_context_node"}  # type: ignore[return-value]

    db = SessionLocal()
    try:
        schema_result = await get_data_group_schema(data_group_id, db)
        raw_schema = schema_result.get("schema", "")

        # sample data
        table_names = []
        for line in raw_schema.splitlines():
            ls = line.strip()
            if ls.startswith("- Table '"):
                try:
                    table_names.append(ls.split("'")[1])
                except IndexError:
                    pass
            elif ls.startswith("Table: "):
                table_names.append(ls.split("Table: ")[1].strip())

        sample_parts = []
        for tname in table_names[:5]:
            try:
                req = DGQueryRequest(query=f'SELECT * FROM "{tname}" LIMIT 3')
                r = await run_data_group_query(data_group_id, req, db)
                if r.get("status") == "success":
                    txt = str(r.get("result", ""))[:500]
                    sample_parts.append(f"Table: {tname}\n{txt}")
            except Exception:
                pass
        sample_data = "\n\n".join(sample_parts) or "No sample data."
    finally:
        db.close()

    engine = DatabaseIntelligenceEngine()
    try:
        db_model = engine.analyze_schema(schema_text=raw_schema, sample_data=sample_data)
    except Exception:
        db_model = engine.analyze_schema(schema_text=raw_schema, sample_data="Sample data omitted.")

    print("[TTD] db_context: LLM analysis complete")
    return {
        **state,
        "raw_schema":      raw_schema,
        "db_intelligence": db_model.model_dump(),
        "current_step":    "ttd_db_context_node",
    }  # type: ignore[return-value]


async def ttd_intent_classifier_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    Kullanıcının sorusunu sınıflandırır.
    - chitchat        → doğrudan interpreter'a git
    - schema_question → schema'dan cevap ver, interpreter'a git
    - sql_query       → sql node'a git
    - analytical      → sql node'a git (karmaşık query)
    """
    print("\n[TTD] ttd_intent_classifier_node başlıyor...")

    llm        = _get_llm(temperature=0)
    user_query = state["user_query"]
    raw_schema = state.get("raw_schema") or "No schema available."

    chat_history_str = ""
    for m in state.get("messages") or []:
        chat_history_str += f"{m.get('role', 'user').upper()}: {m.get('content', '')}\n"
    if not chat_history_str:
        chat_history_str = "No previous messages."

    invoke_kwargs = {
        "user_query":   user_query,
        "raw_schema":   raw_schema,
        "chat_history": chat_history_str,
    }

    try:
        # Strategy 1: structured output
        intent_result: _IntentResult = (
            TTD_INTENT_PROMPT | llm.with_structured_output(_IntentResult)
        ).invoke(invoke_kwargs)  # type: ignore[assignment]
    except Exception as e1:
        print(f"[TTD] intent structured output failed ({type(e1).__name__}), trying text parser")
        try:
            raw_text = llm.invoke(
                TTD_INTENT_PROMPT.format_messages(**invoke_kwargs)
            ).content
            intent_result = _StructuredJsonParser(pydantic_object=_IntentResult).parse(raw_text)  # type: ignore[assignment]
        except Exception as e2:
            print(f"[TTD] intent classifier failed ({e2}), defaulting to sql_query")
            intent_result = _IntentResult(intent="sql_query")  # type: ignore[assignment]

    print(f"[TTD] intent: {intent_result.intent} | multi={intent_result.needs_multiple_queries} | clarify={intent_result.clarification_needed}")

    # Eğer clarification gerekiyorsa, doğrudan cevap ver
    if intent_result.clarification_needed and intent_result.clarification_question:
        return {
            **state,
            "intent":                 intent_result.intent,
            "needs_multiple_queries": intent_result.needs_multiple_queries,
            "clarification_needed":   True,
            "clarification_question": intent_result.clarification_question,
            "sql_query_summary":      intent_result.clarification_question,
            "sql_query_results":      [],
            "sql_query_columns":      [],
            "sql_query_chart":        {},
            "current_step":           "ttd_intent_classifier_node",
        }  # type: ignore[return-value]

    return {
        **state,
        "intent":                 intent_result.intent,
        "needs_multiple_queries": intent_result.needs_multiple_queries,
        "clarification_needed":   intent_result.clarification_needed,
        "clarification_question": intent_result.clarification_question,
        "current_step":           "ttd_intent_classifier_node",
    }  # type: ignore[return-value]


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


async def ttd_result_validator_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    SQL sonuçlarının kullanıcının sorusunu gerçekten cevapladığını doğrular.
    Geçersiz sonuçlarda fix hint üretir ve sql_node'a geri döner.
    """
    print("\n[TTD] ttd_result_validator_node başlıyor...")

    user_query = state["user_query"]
    sql        = state.get("sql_query_raw") or ""
    columns    = state.get("sql_query_columns") or []
    results    = state.get("sql_query_results") or []
    raw_rows   = state.get("ttd_raw_rows") or []

    # Eğer SQL zaten hata verdiyse validasyona gerek yok
    if state.get("sql_error"):
        return {**state, "validation_passed": False, "current_step": "ttd_result_validator_node"}  # type: ignore[return-value]

    sample_rows = raw_rows[:5]
    row_count   = len(results)

    llm = _get_llm(temperature=0)
    invoke_kwargs = {
        "user_query":  user_query,
        "sql":         sql,
        "columns":     ", ".join(columns) if columns else "none",
        "row_count":   str(row_count),
        "sample_rows": json.dumps([list(r) for r in sample_rows], ensure_ascii=False),
    }

    try:
        # Strategy 1: structured output
        try:
            val_result: _ValidationResult = (
                TTD_VALIDATOR_PROMPT | llm.with_structured_output(_ValidationResult)
            ).invoke(invoke_kwargs)  # type: ignore[assignment]
        except Exception as e1:
            print(f"[TTD] validator structured output failed ({type(e1).__name__}), trying text parser")
            raw_text = llm.invoke(
                TTD_VALIDATOR_PROMPT.format_messages(**invoke_kwargs)
            ).content
            val_result = _StructuredJsonParser(pydantic_object=_ValidationResult).parse(raw_text)  # type: ignore[assignment]
    except Exception as e:
        print(f"[TTD] validator failed ({e}), assuming valid")
        val_result = _ValidationResult(is_valid=True)

    print(f"[TTD] validation: is_valid={val_result.is_valid} | issue={val_result.issue}")

    return {
        **state,
        "validation_passed":    val_result.is_valid,
        "validation_issue":     val_result.issue,
        "validation_fix_hint":  val_result.suggested_fix if not val_result.is_valid else None,
        "current_step":         "ttd_result_validator_node",
    }  # type: ignore[return-value]


def _infer_chart_from_text(text: str, columns: list, user_query: str) -> "_ChartConfig":
    """Parse chart type keywords from a plain-text LLM response."""
    text_lower = text.lower()
    if "pie" in text_lower:
        ctype = "pie"
    elif "line" in text_lower or "time series" in text_lower or "trend" in text_lower:
        ctype = "line"
    elif "scatter" in text_lower:
        ctype = "scatter"
    elif "bar" in text_lower or "column" in text_lower or "histogram" in text_lower:
        ctype = "bar"
    else:
        ctype = "table"

    x_col = columns[0] if columns else ""
    y_col = columns[1] if len(columns) > 1 else x_col
    title = user_query[:60] if user_query else "Query Results"
    return _ChartConfig(type=ctype, x=x_col, y=y_col, title=title)


async def ttd_chart_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    Grafik konfigürasyonunu LLM ile belirler ve ECharts payload üretir.
    """
    print("\n[TTD] ttd_chart_node başlıyor...")

    llm        = _get_llm(temperature=0)
    user_query = state["user_query"]
    columns    = state.get("sql_query_columns", [])
    raw_rows   = state.get("ttd_raw_rows", [])

    if not columns or not raw_rows:
        return {**state, "current_step": "ttd_chart_node"}  # type: ignore[return-value]

    sample_rows = raw_rows[:5]

    chart_invoke_kwargs = {
        "columns":     ", ".join(columns),
        "sample_rows": json.dumps([list(r) for r in sample_rows], ensure_ascii=False),
        "user_query":  user_query,
    }
    chart_cfg: _ChartConfig
    try:
        # Strategy 1: tool-call structured output
        chart_cfg = (TTD_CHART_PROMPT | llm.with_structured_output(_ChartConfig)).invoke(chart_invoke_kwargs)  # type: ignore[assignment]
    except Exception as e1:
        print(f"[TTD] chart structured output failed ({type(e1).__name__}), trying text parser")
        try:
            # Strategy 2: text-based JSON parser
            chart_cfg = (TTD_CHART_PROMPT | llm | _StructuredJsonParser(pydantic_object=_ChartConfig)).invoke(chart_invoke_kwargs)  # type: ignore[assignment]
        except Exception as e2:
            print(f"[TTD] chart text parser failed ({type(e2).__name__}), inferring from prose")
            try:
                # Strategy 3: keyword inference from prose response
                raw_text = llm.invoke(
                    TTD_CHART_PROMPT.format_messages(**chart_invoke_kwargs)
                ).content
                chart_cfg = _infer_chart_from_text(raw_text, columns, user_query)
            except Exception as e3:
                print(f"[TTD] chart inference failed ({e3}), defaulting to table")
                chart_cfg = _ChartConfig(type="table", x=columns[0] if columns else "", y="", title="Query Results")

    chart_payload = _build_chart_payload(chart_cfg.model_dump(), raw_rows, columns)

    updated = dict(state)
    updated["sql_query_chart"] = chart_payload or {}
    updated["current_step"]    = "ttd_chart_node"
    updated.pop("ttd_raw_rows", None)  # type: ignore[call-overload]
    return updated  # type: ignore[return-value]


async def ttd_interpreter_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    SQL sonuçlarını LLM ile yorumlar → sql_query_summary doldurur.
    Chitchat ve schema_question intent'leri için de çalışır.
    """
    print("\n[TTD] ttd_interpreter_node başlıyor...")

    intent    = state.get("intent") or "sql_query"
    rows      = state.get("sql_query_results") or []
    sql       = state.get("sql_query_raw") or ""
    query     = state.get("user_query") or ""
    error_msg = state.get("error") or ""
    raw_schema = state.get("raw_schema") or ""

    # Clarification yanıtı zaten set edilmişse pas geç
    if state.get("clarification_needed") and state.get("sql_query_summary"):
        return {**state, "current_step": "ttd_interpreter_node"}  # type: ignore[return-value]

    if error_msg or (not rows and intent not in ("chitchat", "schema_question")):
        if not state.get("sql_query_summary"):
            count   = len(rows)
            summary = f"Query completed, {count} row(s) returned." if count else "Query returned no results."
            return {**state, "sql_query_summary": summary, "current_step": "ttd_interpreter_node"}  # type: ignore[return-value]
        return {**state, "current_step": "ttd_interpreter_node"}  # type: ignore[return-value]

    try:
        llm = _get_llm(temperature=0.2)

        if intent == "chitchat":
            user_msg = f"The user said: \"{query}\"\nPlease respond naturally and helpfully in the same language."
        elif intent == "schema_question":
            user_msg = (
                f"The user asked about the database structure: \"{query}\"\n\n"
                f"Database Schema:\n{raw_schema}\n\n"
                f"Please answer the question directly based on the schema information above."
            )
        else:
            sample   = rows[:20]
            data_str = json.dumps(sample, ensure_ascii=False, indent=2)
            user_msg = (
                f"Current Long-Term Memory:\n{state.get('long_term_memory') or 'None'}\n\n"
                f"User Query: {query}\n\n"
                f"Executed SQL:\n{sql}\n\n"
                f"Returned Data:\n{data_str}\n\n"
                "Please interpret this data and update the long-term memory if needed."
            )

        raw_response = llm.invoke([
            ("system", TTD_INTERPRETER_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        # Accept both JSON and plain-text responses from the LLM
        response: _InterpreterResult = _StructuredJsonParser(
            pydantic_object=_InterpreterResult,
            fallback_field="summary",
        ).parse(raw_response.content)  # type: ignore

        summary        = response.summary.strip()
        updated_memory = response.updated_long_term_memory
    except Exception as e:
        print(f"[TTD] interpreter error: {e}")
        count          = len(rows)
        summary        = f"Query completed, {count} row(s) returned."
        updated_memory = state.get("long_term_memory")

    return {
        **state,
        "sql_query_summary": summary,
        "long_term_memory":  updated_memory,
        "current_step":      "ttd_interpreter_node",
    }  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_intent(state: TTDState) -> str:
    """
    Intent sınıflandırmasına göre routing:
    - clarification_needed → interpreter (soruyu sor)
    - chitchat / schema_question → interpreter (SQL gerekmez)
    - sql_query / analytical → sql_node
    """
    if state.get("clarification_needed"):
        return "ttd_interpreter_node"
    intent = state.get("intent") or "sql_query"
    if intent in ("chitchat", "schema_question"):
        return "ttd_interpreter_node"
    return "ttd_sql_node"


def _route_after_sql(state: TTDState) -> str:
    """SQL hatası varsa retry, yoksa validator'a git."""
    if state.get("sql_error") and (state.get("sql_retry_count") or 0) < 3:
        return "ttd_sql_node"
    return "ttd_result_validator_node"


def _route_after_validator(state: TTDState) -> str:
    """
    Validator sonucuna göre routing:
    - Geçersiz + retry hakkı varsa → sql_node
    - Geçerli veya retry hakkı bittiyse → chart_node
    """
    retry_count = state.get("sql_retry_count") or 0
    if not state.get("validation_passed") and retry_count < 3:
        print(f"[TTD] validator rejected results (issue={state.get('validation_issue')}), retrying SQL (attempt {retry_count + 1})")
        # retry_count'u artır ki sonsuz döngüye girmesin
        return "ttd_sql_node"
    return "ttd_chart_node"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ttd_graph():
    """
    Talk to Data LangGraph pipeline.

    START → db_context → intent_classifier
                              ├─ [chitchat/schema] → interpreter → END
                              └─ [sql/analytical]  → sql_node
                                                         └─ result_validator
                                                              ├─ [invalid] → sql_node (retry)
                                                              └─ [valid]   → chart → interpreter → END
    """
    g = StateGraph(TTDState)  # type: ignore[type-var]

    g.add_node("ttd_db_context_node",        ttd_db_context_node)
    g.add_node("ttd_intent_classifier_node", ttd_intent_classifier_node)
    g.add_node("ttd_sql_node",               ttd_sql_node)
    g.add_node("ttd_result_validator_node",  ttd_result_validator_node)
    g.add_node("ttd_chart_node",             ttd_chart_node)
    g.add_node("ttd_interpreter_node",       ttd_interpreter_node)

    g.add_edge(START,                          "ttd_db_context_node")
    g.add_edge("ttd_db_context_node",          "ttd_intent_classifier_node")
    g.add_conditional_edges("ttd_intent_classifier_node", _route_after_intent)
    g.add_conditional_edges("ttd_sql_node",               _route_after_sql)
    g.add_conditional_edges("ttd_result_validator_node",  _route_after_validator)
    g.add_edge("ttd_chart_node",               "ttd_interpreter_node")
    g.add_edge("ttd_interpreter_node",         END)

    return g.compile()
