import json
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from src.graphs.talktodata.utils import _get_llm, _StructuredJsonParser

from ..state import VizState, _VizInterpretation

from src.core.logging_config import get_logger
logger = get_logger(__name__)

VIZ_INTERPRETER_SYSTEM = """\
You are a data analyst. A chart was just generated to answer the user's request.

Write a SHORT (2-4 sentences), direct reading of what the chart shows: the headline
takeaway, the biggest/smallest values, and any obvious trend or outlier. Use the data
provided. Do not restate every row and do not introduce with pleasantries. Format numbers
readably (e.g. 1,000 not 1000).
"""


async def viz_interpreter_node(state: VizState, config: RunnableConfig) -> VizState:
    """
    Üretilen grafiği + veriyi doğal dille yorumlar → viz_summary doldurur.
    Clarification gerekiyorsa soruyu summary olarak döner.
    """
    logger.info("\n[VIZ] viz_interpreter_node başlıyor...")

    # Clarification: kullanıcıya soruyu göster, LLM'e gitme
    if state.get("clarification_needed"):
        question = state.get("clarification_question") or "Could you clarify what you'd like to visualize?"
        return {**state, "viz_summary": question, "current_step": "viz_interpreter_node"}  # type: ignore[return-value]

    # Upstream error / veri yok
    if state.get("error") or state.get("viz_render_error"):
        msg = state.get("viz_render_error") or state.get("error") or "Could not build the visualization."
        return {**state, "viz_summary": f"The chart could not be generated: {msg}", "current_step": "viz_interpreter_node"}  # type: ignore[return-value]

    rows = state.get("sql_query_results") or []
    if not rows:
        return {**state, "viz_summary": "The query returned no data, so there is nothing to chart.", "current_step": "viz_interpreter_node"}  # type: ignore[return-value]

    spec  = state.get("viz_spec") or {}
    query = state.get("user_query") or ""
    sample = rows[:30]

    user_msg = (
        f"User request: {query}\n\n"
        f"Chart spec: {json.dumps(spec, ensure_ascii=False)}\n\n"
        f"Data (up to 30 rows):\n{json.dumps(sample, ensure_ascii=False, indent=2)}\n\n"
        "Interpret the chart."
    )
    messages = [("system", VIZ_INTERPRETER_SYSTEM), HumanMessage(content=user_msg)]

    try:
        llm = _get_llm(temperature=0.2)
        try:
            response: _VizInterpretation = llm.with_structured_output(_VizInterpretation).invoke(messages)  # type: ignore
        except Exception as e_struct:
            logger.info(f"[VIZ] structured interpreter başarısız ({type(e_struct).__name__}); metin fallback")
            raw = llm.invoke(messages)
            response = _StructuredJsonParser(pydantic_object=_VizInterpretation, fallback_field="summary").parse(raw.content)  # type: ignore
        summary = response.summary.strip()
    except Exception as e:
        logger.error(f"[VIZ] interpreter error: {e}")
        summary = spec.get("title") or f"Chart generated from {len(rows)} row(s)."

    return {**state, "viz_summary": summary, "current_step": "viz_interpreter_node"}  # type: ignore[return-value]
