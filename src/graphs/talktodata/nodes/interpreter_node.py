import json
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from ..state import TTDState, _InterpreterResult
from ..utils import _get_llm, _StructuredJsonParser
from src.domains import get_active_pack
from src.domains.base import MerchantContext

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

    # Domain pack (varsa) yorum merceğini system prompt'a ekle (veri yorumlama için)
    system_prompt = TTD_INTERPRETER_SYSTEM
    if intent not in ("chitchat", "schema_question"):
        try:
            pack = get_active_pack(state.get("domain_pack"))
            mc = MerchantContext.from_dict(state.get("merchant_context"))
            if pack.name != "none" and mc is not None:
                lens = pack.interpret_lens(mc, intent)
                if lens:
                    system_prompt = TTD_INTERPRETER_SYSTEM + "\n\n" + lens.system_suffix
        except Exception as e:  # pragma: no cover - defensive
            print(f"[TTD] Domain pack yorum merceği başarısız ({e}); atlanıyor.")

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
            ("system", system_prompt),
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
