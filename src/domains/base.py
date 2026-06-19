"""Core abstractions for the Domain (Sector) Pack framework.

Every hook on :class:`DomainPack` is **optional and pure** — it returns a
context string (or a small structured object) that the calling node appends to
its own prompt. Hooks never mutate pipeline state and never raise on the happy
path; callers additionally wrap them in try/except so a misbehaving pack can
never bring the pipeline down.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


def trim_doc(text: Optional[str], max_chars: int = 4000) -> str:
    """Clamp a (possibly large) domain document to keep prompt token cost bounded."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n… (kısaltıldı)"


@dataclass
class Expert:
    """A single domain persona (e.g. the Fraud Expert)."""

    key: str                       # stable identifier, e.g. "fraud"
    name: str                      # display name, e.g. "Fraud Expert"
    summary: str                   # one-line persona description
    signals: list[str] = field(default_factory=list)       # data signals it cares about
    intent_bias: list[str] = field(default_factory=list)    # pipeline intents it favors
    query_hint: str = ""           # SQL / query-generation guidance
    feature_hint: str = ""         # feature-engineering guidance
    interpret_lens: str = ""       # interpretation lens (how to read the results)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "summary": self.summary,
            "signals": self.signals,
            "intent_bias": self.intent_bias,
        }


@dataclass
class MerchantContext:
    """Result of expert selection — serialized into pipeline state.

    ``council`` always includes ``primary_expert`` as its first element. When
    ``is_council`` is True more than one expert contributes to every touch point
    and the interpretation step asks the LLM to synthesize their views.
    """

    pack_name: str
    primary_expert: str
    council: list[str] = field(default_factory=list)
    sub_intent: str = ""
    domain_framing: str = ""
    is_council: bool = False

    def to_dict(self) -> dict:
        return {
            "pack_name": self.pack_name,
            "primary_expert": self.primary_expert,
            "council": self.council,
            "sub_intent": self.sub_intent,
            "domain_framing": self.domain_framing,
            "is_council": self.is_council,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["MerchantContext"]:
        if not data:
            return None
        return cls(
            pack_name=data.get("pack_name", ""),
            primary_expert=data.get("primary_expert", ""),
            council=list(data.get("council") or []),
            sub_intent=data.get("sub_intent", ""),
            domain_framing=data.get("domain_framing", ""),
            is_council=bool(data.get("is_council", False)),
        )


@dataclass
class InterpretLens:
    """Domain lens appended to an interpretation/report system prompt."""

    system_suffix: str = ""        # text appended to the system prompt
    is_council: bool = False       # whether multiple experts contribute

    def __bool__(self) -> bool:
        return bool(self.system_suffix.strip())


class FeatureRecipe:
    """A deterministic, executable feature transform contributed by a pack.

    Unlike the text hooks (which only *nudge* the LLM via prompt text), a recipe
    **guarantees** the feature is materialized whenever its inputs are present —
    no LLM in the loop. Recipes run at model grain inside the feature builder,
    at the same point as the plan's ``math_interactions``.

    Contract (duck-typed; ``df`` is a pandas DataFrame, ``meta`` is optional and
    pack-defined — kept untyped here so ``base`` stays import-light):

    * :meth:`applies` — cheap guard: do the required input columns exist?
    * :meth:`build`   — return ``{new_column_name: pandas.Series}`` to add.

    Both must be pure and must never raise on the happy path; the builder also
    wraps every recipe in try/except so a bad recipe can't break the pipeline.
    """

    name: str = "recipe"
    description: str = ""
    leakage_safe: bool = True   # False → builder may skip it under leakage pressure

    def applies(self, df, meta=None) -> bool:  # noqa: ANN001
        return True

    def build(self, df, meta=None) -> dict:  # noqa: ANN001
        return {}


class DomainPack(ABC):
    """Base class for all sector packs.

    Concrete packs must set ``name`` / ``experts`` and implement
    :meth:`select_experts`. All other hooks have safe no-op defaults so a pack
    can opt into only the touch points it cares about.
    """

    name: str = "base"
    label: str = "Base"          # human-readable name for UI dropdowns
    experts: list[Expert] = []

    # ── Expert selection (router + optional council) ────────────────────────
    @abstractmethod
    def select_experts(
        self,
        user_query: str,
        db_intelligence: Optional[dict] = None,
        domain_doc: str = "",
    ) -> MerchantContext:
        """Pick the primary expert (and, if needed, a council) for a query.

        ``domain_doc`` is optional per-dataset context (markdown) that can refine
        routing; packs may ignore it.
        """

    def get_expert(self, key: str) -> Optional[Expert]:
        for e in self.experts:
            if e.key == key:
                return e
        return None

    def active_experts(self, ctx: MerchantContext) -> list[Expert]:
        keys = ctx.council if (ctx.is_council and ctx.council) else [ctx.primary_expert]
        return [e for k in keys if (e := self.get_expert(k)) is not None]

    # ── Agentic analysis (tool-using expert agents) ──────────────────────────
    async def analyze(
        self,
        *,
        user_query: str,
        db_intelligence: Optional[dict],
        domain_doc: str,
        data_group_id: Optional[int],
        user_id: Optional[int],
        source_type: str = "",
    ):
        """Run the pack's expert agents and return a ``MerchantAnalysis`` (or None).

        Default: no agentic layer. Packs that support it (e.g. financial) override
        this. Returning None makes the hooks fall back to their static text.
        """
        return None

    # ── Deterministic feature recipes (executable, not prompt text) ──────────
    def feature_recipes(
        self,
        ctx: MerchantContext,
        db_intelligence: Optional[dict] = None,
        analysis=None,
    ) -> list["FeatureRecipe"]:
        """Return deterministic feature transforms to materialize for this query.

        Default: none. Packs that override this *guarantee* their features get
        built (when inputs exist), as opposed to :meth:`feature_hints` which only
        suggests features to the LLM.
        """
        return []

    # ── Touch point 1: intent discovery ─────────────────────────────────────
    def enrich_intent_prompt(self, ctx: MerchantContext, domain_doc: str = "") -> str:
        return ""

    # ── Touch point 2: query / feature-plan generation ──────────────────────
    def query_hints(self, ctx: MerchantContext, db_intelligence: Optional[dict] = None, analysis=None) -> str:
        return ""

    # ── Touch point 3: feature engineering ──────────────────────────────────
    def feature_hints(self, ctx: MerchantContext, db_intelligence: Optional[dict] = None, analysis=None) -> str:
        return ""

    # ── Touch point 4: output interpretation ────────────────────────────────
    def interpret_lens(self, ctx: MerchantContext, intent: str = "", analysis=None) -> InterpretLens:
        return InterpretLens()
