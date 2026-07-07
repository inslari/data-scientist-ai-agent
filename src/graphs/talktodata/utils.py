import json
import re
from typing import Optional

from langchain_core.output_parsers import BaseOutputParser

from src.core.llm import get_llm
from src.core.config import settings

from src.core.logging_config import get_logger
logger = get_logger(__name__)

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
        logger.error(f"[ttd chart] build error: {e}")
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

def _infer_chart_from_text(text: str, columns: list, user_query: str):
    """Parse chart type keywords from a plain-text LLM response."""
    from .state import _ChartConfig
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

# The structured-output parser + fallback helper now live in src.core.structured
# so the main pipeline shares one implementation (H4). Kept under the old private
# names here so the TTD nodes that import them keep working unchanged.
from src.core.structured import StructuredJsonParser as _StructuredJsonParser  # noqa: E402,F401
from src.core.structured import invoke_structured as _parse_with_fallback  # noqa: E402,F401
