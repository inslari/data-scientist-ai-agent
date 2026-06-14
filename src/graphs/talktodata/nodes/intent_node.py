from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from ..state import TTDState, _IntentResult
from ..utils import _get_llm, _StructuredJsonParser

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
