"""
Data Visualization öneri üreticisi.

Bir data group için, şemasına bakarak kullanıcının üretebileceği görselleştirme
önerileri (kart) üretir. ttd_suggestion_generator ile aynı deseni izler, ama string
soru yerine yapılandırılmış öneri objeleri döndürür:

    {"title", "description", "chart_type", "x", "y", "agg"}

Üretim arka planda (worker job'ı) çalışır; sonuç DataGroupModel.viz_suggestions
JSON kolonuna (list[dict]) yazılır. Data Visualization sayfası boşken kart olarak gösterilir.
"""
from __future__ import annotations

import json
import re

from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.llm import get_llm

from src.core.logging_config import get_logger
logger = get_logger(__name__)

SUGGESTION_COUNT = 6
_MAX_SCHEMA_CHARS = 6000
_VALID_CHART_TYPES = {"bar", "line", "pie", "scatter", "hist", "box", "heatmap", "grouped_bar"}
_VALID_AGGS = {"none", "sum", "mean", "count"}

_SYSTEM_PROMPT = """You are a data-visualization analyst. You will be given a database/CSV schema.

Your task: Produce {count} DIVERSE, concrete chart ideas a user could generate from this data.

Rules:
- Base every idea on tables/columns that ACTUALLY exist in the schema; never invent column names.
- Diversity: cover different chart types (bar, line, pie, scatter, hist, box, grouped_bar) and
  different columns. Prefer charts that reveal something useful (distributions, comparisons, trends, breakdowns).
- Each idea has:
    - "title": a short chart title (<= 8 words)
    - "description": one clear sentence on what it shows
    - "chart_type": one of bar | line | pie | scatter | hist | box | heatmap | grouped_bar
    - "x": the category/axis/label column name (exactly as in schema)
    - "y": the numeric value column name, or null (e.g. for hist)
    - "agg": one of none | sum | mean | count
- Return ONLY a JSON array of {count} objects. No prose, no markdown.

Example:
[{{"title":"Revenue by region","description":"Total revenue for each region.","chart_type":"bar","x":"region","y":"revenue","agg":"sum"}}]
"""


def _coerce_suggestion(obj: dict) -> dict | None:
    """Bir LLM öneri objesini normalize eder; geçersizse None."""
    if not isinstance(obj, dict):
        return None
    title = str(obj.get("title", "")).strip()
    if not title:
        return None
    chart_type = str(obj.get("chart_type", "bar")).strip().lower()
    if chart_type not in _VALID_CHART_TYPES:
        chart_type = "bar"
    agg = str(obj.get("agg", "none")).strip().lower()
    if agg not in _VALID_AGGS:
        agg = "none"
    x = obj.get("x")
    y = obj.get("y")
    return {
        "title":       title[:80],
        "description": str(obj.get("description", "")).strip()[:200],
        "chart_type":  chart_type,
        "x":           str(x).strip() if x else None,
        "y":           str(y).strip() if y else None,
        "agg":         agg,
    }


def _parse_suggestions(text: str) -> list[dict]:
    """LLM çıktısından öneri objesi listesini çıkarır (fence/fazla metne dayanıklı)."""
    if not text:
        return []
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            out = [s for s in (_coerce_suggestion(o) for o in data) if s]
            return out[:SUGGESTION_COUNT]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


async def generate_viz_suggestions(group_id: int, db: Session, user_id: int) -> list[dict]:
    """
    Data group şemasından {SUGGESTION_COUNT} görselleştirme önerisi üretir.
    Hata durumunda boş liste döner (best-effort).
    """
    from api.schemas.data_group import get_data_group_schema_logic

    try:
        schema_result = await get_data_group_schema_logic(group_id, db, user_id)
        raw_schema = (schema_result or {}).get("schema", "") if isinstance(schema_result, dict) else ""
    except Exception as e:
        logger.info(f"[viz_suggest] Şema çekilemedi (group {group_id}): {str(e)[:150]}")
        return []

    if not raw_schema or not raw_schema.strip():
        logger.info(f"[viz_suggest] Boş şema (group {group_id}); öneri üretilmedi.")
        return []

    schema_text = raw_schema[:_MAX_SCHEMA_CHARS]
    llm = get_llm(
        model_name=settings.DEFAULT_LLM_MODEL,
        temperature=0.4,
        provider=settings.DEFAULT_LLM_PROVIDER,
    )
    system = _SYSTEM_PROMPT.format(count=SUGGESTION_COUNT)
    try:
        resp = await llm.ainvoke([
            ("system", system),
            ("human", f"Schema:\n{schema_text}\n\nProduce {SUGGESTION_COUNT} chart ideas as a JSON array."),
        ])
        content = resp.content if hasattr(resp, "content") else str(resp)
        suggestions = _parse_suggestions(content if isinstance(content, str) else str(content))
        logger.info(f"[viz_suggest] group {group_id}: {len(suggestions)} öneri üretildi.")
        return suggestions
    except Exception as e:
        logger.info(f"[viz_suggest] LLM üretimi başarısız (group {group_id}): {str(e)[:150]}")
        return []
