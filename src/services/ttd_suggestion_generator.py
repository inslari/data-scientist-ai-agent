"""
Talk-to-Data örnek soru üreticisi.

Bir data group için, şemasına bakarak kullanıcının sorabileceği 5 adet doğal dil
örnek sorusu üretir. Bu öneriler data group'a özeldir ve TTD sohbeti boşken
(mesaj yokken) boş ekranda gösterilir.

Üretim arka planda (worker job'ı) çalışır; sonuç DataGroupModel.ttd_suggestions
JSON kolonuna (list[str]) yazılır.
"""
from __future__ import annotations

import json
import re

from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.llm import get_llm

from src.core.logging_config import get_logger
logger = get_logger(__name__)

SUGGESTION_COUNT = 5
_MAX_SCHEMA_CHARS = 6000  # LLM'e gönderilen şemayı sınırla (uzun şemalarda halüsinasyon/maliyet)

_SYSTEM_PROMPT = """You are a data analyst assistant. You will be given a database/CSV schema.

Your task: Produce {count} SHORT, CLEAR and DIVERSE natural-language questions that a
user could ask about this data in a "Talk to Data" chat.

Rules:
- Questions must be in English.
- Each question must be SQL-answerable, concrete and data-driven (not generic/abstract).
- Base questions on tables/columns that ACTUALLY exist in the schema; never invent column names.
- Diversity: try to cover different analysis types such as count/total, grouping,
  ranking (top N), time trend, ratio/comparison (when the schema allows).
- Each question is a single sentence, at most ~12 words.
- Return ONLY a JSON array of {count} strings. No other text, explanation, or markdown.

Example output format:
["Question one?", "Question two?", "Question three?", "Question four?", "Question five?"]
"""


def _parse_suggestions(text: str) -> list[str]:
    """LLM çıktısından string listesini çıkarır (markdown fence / fazla metne dayanıklı)."""
    if not text:
        return []
    # Markdown kod bloğunu temizle
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # İlk JSON dizisini yakala
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip()]
            return out[:SUGGESTION_COUNT]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: satır satır ayıkla
    lines = [re.sub(r'^[\s\-\*\d\.\)"]+|["\,]+$', "", ln).strip() for ln in cleaned.splitlines()]
    return [ln for ln in lines if len(ln) > 3][:SUGGESTION_COUNT]


async def generate_ttd_suggestions(group_id: int, db: Session, user_id: int) -> list[str]:
    """
    Data group'un şemasından {SUGGESTION_COUNT} örnek TTD sorusu üretir.

    Hata durumunda boş liste döner (çağıran taraf bunu 'üretilemedi' olarak ele alır).
    """
    # Geç import: circular dependency (api.schemas ↔ api.routes) ve ağır import maliyetini önle.
    from api.schemas.data_group import get_data_group_schema_logic

    try:
        schema_result = await get_data_group_schema_logic(group_id, db, user_id)
        raw_schema = (schema_result or {}).get("schema", "") if isinstance(schema_result, dict) else ""
    except Exception as e:
        logger.info(f"[ttd_suggest] Şema çekilemedi (group {group_id}): {str(e)[:150]}")
        return []

    if not raw_schema or not raw_schema.strip():
        logger.info(f"[ttd_suggest] Boş şema (group {group_id}); öneri üretilmedi.")
        return []

    schema_text = raw_schema[:_MAX_SCHEMA_CHARS]
    llm = get_llm(
        model_name=settings.DEFAULT_LLM_MODEL,
        temperature=0.4,  # biraz çeşitlilik için
        provider=settings.DEFAULT_LLM_PROVIDER,
    )
    system = _SYSTEM_PROMPT.format(count=SUGGESTION_COUNT)
    try:
        resp = await llm.ainvoke([
            ("system", system),
            ("human", f"Schema:\n{schema_text}\n\nPlease produce {SUGGESTION_COUNT} example questions as a JSON array."),
        ])
        content = resp.content if hasattr(resp, "content") else str(resp)
        suggestions = _parse_suggestions(content if isinstance(content, str) else str(content))
        logger.info(f"[ttd_suggest] group {group_id}: {len(suggestions)} öneri üretildi.")
        return suggestions
    except Exception as e:
        logger.info(f"[ttd_suggest] LLM üretimi başarısız (group {group_id}): {str(e)[:150]}")
        return []
