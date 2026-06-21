import json
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from ..state import TTDState, _ValidationResult
from ..utils import _get_llm, _StructuredJsonParser

from src.core.logging_config import get_logger
logger = get_logger(__name__)

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

async def ttd_result_validator_node(state: TTDState, config: RunnableConfig) -> TTDState:
    """
    SQL sonuçlarının kullanıcının sorusunu gerçekten cevapladığını doğrular.
    Geçersiz sonuçlarda fix hint üretir ve sql_node'a geri döner.
    """
    logger.info("\n[TTD] ttd_result_validator_node başlıyor...")

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
            logger.error(f"[TTD] validator structured output failed ({type(e1).__name__}), trying text parser")
            raw_text = llm.invoke(
                TTD_VALIDATOR_PROMPT.format_messages(**invoke_kwargs)
            ).content
            val_result = _StructuredJsonParser(pydantic_object=_ValidationResult).parse(raw_text)  # type: ignore[assignment]
    except Exception as e:
        logger.error(f"[TTD] validator failed ({e}), assuming valid")
        val_result = _ValidationResult(is_valid=True)

    retry_count = state.get("sql_retry_count") or 0
    logger.info(f"[TTD] validation: is_valid={val_result.is_valid} | issue={val_result.issue} | retry_count={retry_count}")

    return {
        **state,
        "validation_passed":    val_result.is_valid,
        "validation_issue":     val_result.issue,
        "validation_fix_hint":  val_result.suggested_fix if not val_result.is_valid else None,
        "sql_retry_count":      retry_count + 1 if not val_result.is_valid else retry_count,
        "current_step":         "ttd_result_validator_node",
    }  # type: ignore[return-value]
