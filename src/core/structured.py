"""Shared structured-output helpers with a text-parser fallback (C1 / H4).

`with_structured_output(...)` raises on weaker models that don't emit clean tool
calls / valid JSON — on the main pipeline that surfaced as a hard node failure,
while the talk-to-data graph already degraded gracefully via a regex JSON parser.
This module lifts that parser out of `graphs/talktodata/utils.py` so **both**
graphs share one implementation:

    result = invoke_structured(llm, prompt_value, MySchema)

``invoke_structured`` tries native structured output first, then falls back to
invoking the model plainly and parsing JSON out of the text (stripping code
fences, locating the first JSON object). ``fallback_field`` lets a plain-text
answer be wrapped as ``{fallback_field: text}`` for single-field schemas.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.output_parsers import BaseOutputParser

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class StructuredJsonParser(BaseOutputParser):
    """Robust parser for LLMs that return raw text, fenced JSON, or wrong shapes.

    - strips ```json fences
    - locates the first ``{...}`` object and validates it against the schema
    - if no JSON and ``fallback_field`` is set, wraps the plain text as that field
    """

    pydantic_object: type
    fallback_field: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def parse(self, text: str):
        text = (text or "").strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self.pydantic_object.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass  # fall through to plain-text handling
        if self.fallback_field:
            return self.pydantic_object.model_validate({self.fallback_field: text})
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")

    def get_format_instructions(self) -> str:
        from langchain_core.output_parsers import PydanticOutputParser
        return PydanticOutputParser(pydantic_object=self.pydantic_object).get_format_instructions()


def invoke_structured(llm, prompt_value, pydantic_cls, fallback_field: Optional[str] = None):
    """Try ``with_structured_output`` then fall back to :class:`StructuredJsonParser`.

    ``prompt_value`` is anything ``llm.invoke`` accepts directly — a
    ``ChatPromptValue`` (e.g. ``prompt.invoke({...})``), a message list, or a
    string. Returns an instance of ``pydantic_cls``.
    """
    try:
        return llm.with_structured_output(pydantic_cls).invoke(prompt_value)
    except Exception as e1:
        logger.warning(
            "with_structured_output failed for %s (%s) — trying text parser",
            getattr(pydantic_cls, "__name__", pydantic_cls), type(e1).__name__,
        )
        raw_text = llm.invoke(prompt_value).content
        if isinstance(raw_text, list):  # some providers return content parts
            raw_text = " ".join(str(p) for p in raw_text)
        return StructuredJsonParser(
            pydantic_object=pydantic_cls,
            fallback_field=fallback_field,
        ).parse(raw_text)
