import json
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from ..state import TTDState, _ChartConfig
from ..utils import _get_llm, _build_chart_payload, _infer_chart_from_text, _StructuredJsonParser

from src.core.logging_config import get_logger
logger = get_logger(__name__)

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

async def ttd_chart_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    Grafik konfigürasyonunu LLM ile belirler ve ECharts payload üretir.
    """
    logger.info("\n[TTD] ttd_chart_node başlıyor...")

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
        logger.error(f"[TTD] chart structured output failed ({type(e1).__name__}), trying text parser")
        try:
            # Strategy 2: text-based JSON parser
            chart_cfg = (TTD_CHART_PROMPT | llm | _StructuredJsonParser(pydantic_object=_ChartConfig)).invoke(chart_invoke_kwargs)  # type: ignore[assignment]
        except Exception as e2:
            logger.error(f"[TTD] chart text parser failed ({type(e2).__name__}), inferring from prose")
            try:
                # Strategy 3: keyword inference from prose response
                raw_text = llm.invoke(
                    TTD_CHART_PROMPT.format_messages(**chart_invoke_kwargs)
                ).content
                chart_cfg = _infer_chart_from_text(raw_text, columns, user_query)
            except Exception as e3:
                logger.error(f"[TTD] chart inference failed ({e3}), defaulting to table")
                chart_cfg = _ChartConfig(type="table", x=columns[0] if columns else "", y="", title="Query Results")

    chart_payload = _build_chart_payload(chart_cfg.model_dump(), raw_rows, columns)

    updated = dict(state)
    updated["sql_query_chart"] = chart_payload or {}
    updated["current_step"]    = "ttd_chart_node"
    updated.pop("ttd_raw_rows", None)  # type: ignore[call-overload]
    return updated  # type: ignore[return-value]
