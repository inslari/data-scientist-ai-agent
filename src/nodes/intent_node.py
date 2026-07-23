from src.core.nodes import node
from ..state import AgentState
from ..core.llm import get_llm
from ..core.config import settings
from ..core.prompt_loader import create_chat_prompt
from ..core.structured import invoke_structured
from ..schemas.intent_schemas import IntentResult
from ..domains import get_active_pack

from src.core.logging_config import get_logger
logger = get_logger(__name__)


# UI'dan açıkça set edilebilen intent'ler — LLM tespitine gerek yok
_UI_EXPLICIT_INTENTS = {"deep_analysis"}


def _select_merchant_context(state: AgentState):
    """Aktif domain pack varsa expert seçimi yapar.

    Döndürür: (merchant_context_dict | None, intent_prompt_eki: str).
    Pack 'none' ise veya hata olursa pipeline'ı etkilemez (boş döner).
    """
    try:
        pack = get_active_pack(state.get("domain_pack"))
        if pack.name == "none":
            return None, ""
        domain_doc = state.get("domain_doc") or ""
        ctx = pack.select_experts(state.get("user_query", ""), state.get("db_intelligence"), domain_doc)
        logger.info(f"[intent_node] Domain pack '{pack.name}' aktif → expert(ler): "
              f"{ctx.council if ctx.is_council else [ctx.primary_expert]}")
        return ctx.to_dict(), pack.enrich_intent_prompt(ctx, domain_doc)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[intent_node] Domain pack seçimi başarısız ({e}); atlanıyor.")
        return None, ""


@node("intent_node")
def intent_node(state: AgentState) -> AgentState:
    """
    Node 1: Kullanıcının sorgusunu analiz edip intent'i belirler.

    Eğer intent UI'dan açıkça verilmişse (ör. "Veri Analizi" modu seçilmiş),
    LLM tespiti atlanır — kullanıcı ne istediğini zaten söylemiştir.
    """

    # Domain pack (varsa) expert seçimi — UI explicit intent durumunda da çalışır
    merchant_context, domain_context = _select_merchant_context(state)

    # UI'dan açık seçim varsa LLM'e sorma
    incoming_intent = state.get("intent")
    if incoming_intent in _UI_EXPLICIT_INTENTS:
        logger.warning(f"[intent_node] UI'dan açık intent: '{incoming_intent}' — LLM atlandı")
        return {
            "intent": incoming_intent,
            "intent_confidence": 1.0,
            "merchant_context": merchant_context,
            "current_step": "intent_node",
        }

    llm = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0, role="intelligence")

    INTENT_PROMPT = create_chat_prompt(
        "intent_discovery.md", "Kullanıcı sorusu: {user_query}\n\n{domain_context}"
    )
    prompt_value = INTENT_PROMPT.invoke(
        {"user_query": state["user_query"], "domain_context": domain_context or ""}
    )

    # Structured output with a text-parser fallback (C1/H4).
    result: IntentResult = invoke_structured(llm, prompt_value, IntentResult)

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "merchant_context": merchant_context,
        "current_step": "intent_node",
    }