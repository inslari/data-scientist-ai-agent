from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

class TTDState(TypedDict):
    # Input
    user_query: str
    data_group_id: Optional[int]
    messages: Optional[list]
    long_term_memory: Optional[str]

    # Domain (Sector) Pack — pluggable Merchant Intelligence layer
    domain_pack: Optional[str]        # Active pack name ("none" disables the layer)
    merchant_context: Optional[dict]  # MerchantContext.to_dict() — selected expert(s) & framing
    domain_doc: Optional[str]         # Per-data-group dataset description (markdown context)

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
