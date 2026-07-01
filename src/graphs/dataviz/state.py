from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


class VizState(TypedDict):
    """
    Data Visualization graph state.

    TTDState ile aynı veri/şema/SQL anahtarlarını KORUR — böylece Talk-to-Data'nın
    context ve validator node'ları (ttd_db_context_node, ttd_result_validator_node)
    doğrudan yeniden kullanılabilir. Görselleştirmeye özgü alanlar ayrıca eklenir.
    """
    # Input
    user_query: str
    data_group_id: Optional[int]
    messages: Optional[list]
    long_term_memory: Optional[str]

    # Kart seçildiyse hazır öneri objesi (viz_suggestion_generator çıktısı);
    # serbest doğal dil isteğinde None.
    selected_suggestion: Optional[dict]

    # Domain (Sector) Pack — Talk-to-Data ile aynı pluggable katman
    domain_pack: Optional[str]
    merchant_context: Optional[dict]
    domain_doc: Optional[str]

    # Schema / DB intelligence (cache'den veya canlı)
    raw_schema: Optional[str]
    db_intelligence: Optional[dict]

    # Plan node çıktısı (viz_plan_node)
    viz_spec: Optional[dict]                  # _VizSpec.model_dump()
    clarification_needed: Optional[bool]
    clarification_question: Optional[str]

    # SQL node çıktısı (TTDState ile aynı anahtarlar → validator reuse)
    sql_query_raw: Optional[str]
    sql_query_columns: Optional[list]
    sql_query_results: Optional[list]
    sql_query_warning: Optional[str]

    # Validator çıktısı (ttd_result_validator_node ile aynı anahtarlar)
    validation_passed: Optional[bool]
    validation_issue: Optional[str]
    validation_fix_hint: Optional[str]

    # Render node çıktısı (viz_render_node)
    viz_image_b64: Optional[str]              # matplotlib PNG (base64)
    viz_chart: Optional[dict]                 # recharts payload | None (hibrit)
    viz_render_error: Optional[str]

    # Interpreter çıktısı
    viz_summary: Optional[str]

    # Control
    current_step: Optional[str]
    error: Optional[str]
    sql_error: Optional[str]
    sql_retry_count: Optional[int]
    ttd_raw_rows: Optional[list]              # sql_node → render_node ham tuple satırları


# ─── Structured LLM output modelleri (talktodata deseni: _drop_extra validator) ──

_CHART_TYPES = "bar | line | pie | scatter | hist | box | heatmap | grouped_bar"
_AGGS = "none | sum | mean | count"


class _VizSpec(BaseModel):
    chart_type: str = Field(description=f"One of: {_CHART_TYPES}")
    x: Optional[str] = Field(default=None, description="Column for the X axis / category / labels")
    y: Optional[str] = Field(default=None, description="Column for the Y axis / numeric value")
    group: Optional[str] = Field(default=None, description="Optional column to group/hue by (grouped_bar)")
    agg: str = Field(default="none", description=f"Aggregation to apply to y grouped by x: {_AGGS}")
    title: str = Field(default="Visualization", description="Chart title")
    sql: str = Field(description="A single SELECT query that returns exactly the columns needed to draw this chart")

    @model_validator(mode="before")
    @classmethod
    def _drop_extra(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "sql" not in values and "query" in values:
                values["sql"] = values.pop("query")
            allowed = {"chart_type", "x", "y", "group", "agg", "title", "sql"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class _VizPlanResult(BaseModel):
    spec: _VizSpec = Field(description="The visualization specification")
    clarification_needed: bool = Field(default=False, description="True if the request is too vague to build a chart")
    clarification_question: Optional[str] = Field(default=None, description="Question to ask the user, if clarification is needed")

    @model_validator(mode="before")
    @classmethod
    def _drop_extra(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"spec", "clarification_needed", "clarification_question"}
            return {k: v for k, v in values.items() if k in allowed}
        return values


class _VizInterpretation(BaseModel):
    summary: str = Field(description="A short, natural-language reading of what the chart shows.")

    @model_validator(mode="before")
    @classmethod
    def _drop_extra(cls, values: Any) -> Any:
        if isinstance(values, dict):
            allowed = {"summary"}
            return {k: v for k, v in values.items() if k in allowed}
        return values
