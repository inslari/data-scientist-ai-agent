from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm
from src.core.config import settings
from src.graphs.talktodata.utils import _StructuredJsonParser

from src.core.logging_config import get_logger
logger = get_logger(__name__)


def log_token_usage(msg, where: str) -> None:
    """Best-effort token/maliyet izleme. AIMessage'ta usage varsa loglar."""
    try:
        usage = getattr(msg, "usage_metadata", None) or {}
        if usage:
            logger.info(f"[TA tokens] {where}: in={usage.get('input_tokens')} "
                        f"out={usage.get('output_tokens')} total={usage.get('total_tokens')}")
    except Exception:
        pass


_LANG_NAMES = {"tr": "Türkçe", "en": "English"}


def language_directive(lang: str) -> str:
    """
    Çıktı dilini LLM'e dayatan kısa yönerge. classify/theme/report promptlarına
    eklenir; özet, aspect ve tema adları bu dilde üretilir.
    """
    name = _LANG_NAMES.get((lang or "tr").lower(), "Türkçe")
    return f"ÖNEMLİ: Tüm serbest-metin çıktılarını (özet, aspect, tema adı, açıklama) {name} dilinde üret."


def _get_llm(temperature: float = 0):
    """
    Text-analysis'e ÖZEL LLM. TEXT_ANALYSIS_LLM_PROVIDER/MODEL ile ayrı seçilebilir;
    set edilmezse intelligence engine modeline düşer (config'deki fallback).
    """
    return get_llm(
        provider=settings.TEXT_ANALYSIS_LLM_PROVIDER,
        model_name=settings.TEXT_ANALYSIS_LLM_MODEL,
        temperature=temperature,
    )


def structured_invoke(prompt: ChatPromptTemplate, schema, invoke_kwargs: dict):
    """
    with_structured_output dener; başarısız olursa metin tabanlı JSON parser'a
    düşer (TTD ile aynı dayanıklılık stratejisi).
    """
    llm = _get_llm(temperature=0)
    try:
        return (prompt | llm.with_structured_output(schema)).invoke(invoke_kwargs)
    except Exception:
        raw_text = llm.invoke(prompt.format_messages(**invoke_kwargs)).content
        return _StructuredJsonParser(pydantic_object=schema).parse(raw_text)
