from ..state import AgentState
from ..core.llm import get_llm
from ..core.config import settings
from ..core.prompt_loader import create_chat_prompt
from ..schemas.intent_schemas import IntentResult


def intent_node(state: AgentState) -> AgentState:
    """
    Node 1: Kullanıcının sorgusunu analiz edip intent'i belirler.
    """
    print("\n========================================\n[NODE START] intent_node başlıyor...\n========================================")
    llm = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(IntentResult)

    INTENT_PROMPT = create_chat_prompt("intent_discovery.md", "Kullanıcı sorusu: {user_query}")
    chain = INTENT_PROMPT | structured_llm

    result: IntentResult = chain.invoke({"user_query": state["user_query"]})  # type: ignore[assignment]

    return {
        **state,
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "current_step": "intent_node",
    }