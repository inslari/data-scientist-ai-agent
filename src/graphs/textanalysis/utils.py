from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm
from src.core.config import settings
from src.graphs.talktodata.utils import _StructuredJsonParser


def _get_llm(temperature: float = 0):
    """Text-analysis için yapılandırılmış LLM (intelligence engine modeli)."""
    return get_llm(
        provider=settings.INTELLIGENCE_ENGINE_PROVIDER,
        model_name=settings.INTELLIGENCE_ENGINE_MODEL,
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
