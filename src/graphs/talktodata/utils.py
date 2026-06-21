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
        logger.error(f"[TTD] with_structured_output failed ({type(e1).__name__}), trying text parser")
        raw_text = llm.invoke(prompt_value).content
        return _StructuredJsonParser(
            pydantic_object=pydantic_cls,
            fallback_field=fallback_field,
        ).parse(raw_text)
