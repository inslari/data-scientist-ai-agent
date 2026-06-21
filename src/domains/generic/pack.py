"""``GenericDomainPack`` — a fully data-driven :class:`DomainPack`.

A concrete sector pack subclasses this and sets a handful of class attributes
(experts, keywords, framing labels). All four prompt hooks, expert routing and the
agentic ``analyze`` come for free. The text the hooks emit is identical in shape to
the original hand-written financial pack, with the sector-specific labels
substituted in — so behavior is preserved while new sectors cost ~one small file.
"""

from __future__ import annotations

from typing import Optional

from src.domains.base import DomainPack, FeatureRecipe, InterpretLens, MerchantContext, trim_doc
from src.domains.schemas import MerchantAnalysis
from . import agents as _agents
from . import router as _router


class GenericDomainPack(DomainPack):
    # ── Sector config (override per pack) ────────────────────────────────────
    name: str = "generic"
    label: str = "Generic"
    experts: list = []
    expert_keywords: dict[str, list[str]] = {}
    default_expert: str = ""                 # keyword-fallback default

    # Framing labels woven into prompts / agent personas
    engine_label: str = "Domain Intelligence Engine"   # e.g. "Merchant Intelligence Engine"
    lens_label: str = "DOMAIN INTELLIGENCE"            # agentic-guidance + lens header
    domain_label: str = "DOMAIN"                       # static "DOMAIN ..." headers
    domain_descriptor: str = "this sector's data"      # "... for {descriptor}"
    domain_adjective: str = "domain"                   # "{adjective} expert lens(es)"
    default_analyst_label: str = "Analyst"             # fallback when no expert resolves

    # Which ANALYSIS_MODULES group the deep-analysis path should add on top of
    # "general" (see src/services/analysis_engine.py). "" / "general" → none extra.
    analysis_type: str = "general"

    # ── Expert selection ─────────────────────────────────────────────────────
    def select_experts(
        self,
        user_query: str,
        db_intelligence: Optional[dict] = None,
        domain_doc: str = "",
    ) -> MerchantContext:
        return _router.route(
            user_query,
            pack_name=self.name,
            experts=self.experts,
            expert_keywords=self.expert_keywords,
            default_expert=self.default_expert or (self.experts[0].key if self.experts else ""),
            engine_label=self.engine_label,
            domain_descriptor=self.domain_descriptor,
            db_intelligence=db_intelligence,
            domain_doc=domain_doc,
        )

    # ── Agentic analysis: run the selected experts as tool-using agents ──────
    async def analyze(
        self,
        *,
        user_query: str,
        db_intelligence: Optional[dict],
        domain_doc: str,
        data_group_id: Optional[int],
        user_id: Optional[int],
        source_type: str = "",
    ) -> Optional[MerchantAnalysis]:
        if not data_group_id or user_id is None:
            return None
        ctx = self.select_experts(user_query, db_intelligence, domain_doc)
        expert_keys = ctx.council if (ctx.is_council and ctx.council) else [ctx.primary_expert]
        expert_keys = [k for k in expert_keys if k]
        if not expert_keys:
            return None

        experts_by_key = {e.key: e for e in self.experts}
        engine_label, descriptor = self.engine_label, self.domain_descriptor

        async def _one(key, uq, db_context, doc, run_sql_tool):
            expert = experts_by_key.get(key)
            if expert is None:
                return None
            return await _agents.run_one_expert(
                expert, uq, db_context, doc, run_sql_tool,
                engine_label=engine_label, domain_descriptor=descriptor,
            )

        async def _syn(findings, uq):
            return await _agents.synthesize(findings, uq, engine_label=engine_label)

        return await _agents.run_council(
            pack_name=self.name,
            expert_keys=expert_keys,
            is_council=ctx.is_council,
            primary_expert=ctx.primary_expert,
            user_query=user_query,
            db_intelligence=db_intelligence or {},
            domain_doc=domain_doc or "",
            data_group_id=data_group_id,
            user_id=user_id,
            source_type=source_type,
            run_one_expert_fn=_one,
            synthesize_fn=_syn,
        )

    # ── Deterministic feature recipes (override per pack if any) ─────────────
    def feature_recipes(self, ctx, db_intelligence=None, analysis=None) -> list[FeatureRecipe]:
        return []

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _expert_names(self, ctx: MerchantContext) -> str:
        experts = self.active_experts(ctx)
        return ", ".join(e.name for e in experts) if experts else self.default_analyst_label

    # ── Touch point 1: intent discovery ──────────────────────────────────────
    def enrich_intent_prompt(self, ctx: MerchantContext, domain_doc: str = "") -> str:
        experts = self.active_experts(ctx)
        if not experts:
            return ""
        biases = sorted({b for e in experts for b in e.intent_bias})
        lines = [
            f"--- {self.domain_label} DOMAIN CONTEXT ({self.engine_label}) ---",
            f"Active expert(s): {self._expert_names(ctx)}.",
        ]
        if ctx.domain_framing:
            lines.append(f"Framing: {ctx.domain_framing}")
        if ctx.sub_intent:
            lines.append(f"Analytical goal: {ctx.sub_intent}")
        lines.append(
            f"This is {self.domain_descriptor}. Typical intents for such requests: "
            + ", ".join(biases) + "."
        )
        doc = trim_doc(domain_doc, max_chars=1500)
        if doc:
            lines.append("\nDataset description (provided by the user):\n" + doc)
        lines.append(
            "Use this only as domain context — still classify the intent strictly per the "
            "rules above based on what the user actually asks."
        )
        return "\n".join(lines)

    # ── Touch point 2: query / feature-plan generation ───────────────────────
    def query_hints(self, ctx: MerchantContext, db_intelligence: Optional[dict] = None, analysis=None) -> str:
        if analysis and analysis.findings:
            blocks = [f"--- {self.lens_label} QUERY GUIDANCE (from data investigation) ---"]
            for f in analysis.findings:
                if f.key_signals:
                    blocks.append(f"[{f.expert}] observed: " + "; ".join(f.key_signals))
            blocks.append("Use these data-grounded signals; reference only existing columns.")
            return "\n".join(blocks)
        experts = self.active_experts(ctx)
        if not experts:
            return ""
        blocks = [f"--- {self.domain_label} DOMAIN QUERY GUIDANCE ---"]
        for e in experts:
            blocks.append(f"[{e.name}] {e.query_hint}")
            if e.signals:
                blocks.append("  Key signals: " + "; ".join(e.signals))
        blocks.append(
            "Apply this guidance only where matching columns actually exist in the schema; "
            "do not invent columns."
        )
        return "\n".join(blocks)

    # ── Touch point 3: feature engineering ────────────────────────────────────
    def feature_hints(self, ctx: MerchantContext, db_intelligence: Optional[dict] = None, analysis=None) -> str:
        if analysis and analysis.findings:
            blocks = ["", f"--- {self.lens_label} FEATURE GUIDANCE (from data investigation) ---"]
            targets = [f.recommended_target for f in analysis.findings if f.recommended_target]
            if targets:
                blocks.append("Recommended target(s): " + ", ".join(dict.fromkeys(targets)))
            feats = [feat for f in analysis.findings for feat in f.recommended_features]
            if feats:
                blocks.append("Recommended features:")
                blocks += [f"  - {x}" for x in dict.fromkeys(feats)]
            risks = [r for f in analysis.findings for r in f.risks]
            if risks:
                blocks.append("Risks / leakage to avoid: " + "; ".join(dict.fromkeys(risks)))
            blocks.append("Only propose features derivable from existing columns.")
            return "\n".join(blocks)
        experts = self.active_experts(ctx)
        if not experts:
            return ""
        blocks = ["", f"--- {self.domain_label} DOMAIN FEATURE GUIDANCE ---"]
        for e in experts:
            blocks.append(f"[{e.name}] {e.feature_hint}")
        blocks.append(
            "Only propose features derivable from the available columns; respect any leakage "
            "warnings above."
        )
        return "\n".join(blocks)

    # ── Touch point 4: output interpretation ──────────────────────────────────
    def interpret_lens(self, ctx: MerchantContext, intent: str = "", analysis=None) -> InterpretLens:
        if analysis and analysis.findings:
            blocks = ["", f"--- {self.lens_label} LENS (agentic analysis) ---"]
            for f in analysis.findings:
                line = f"[{f.expert}] {f.interpretation_notes or f.summary}"
                blocks.append(line.strip())
            if analysis.synthesis:
                blocks.append("\nCouncil synthesis:\n" + analysis.synthesis)
            blocks.append(
                "Ground every statement in the actual numbers; turn the above findings into "
                "concrete, actionable recommendations."
            )
            return InterpretLens(system_suffix="\n".join(blocks), is_council=analysis.is_council)
        experts = self.active_experts(ctx)
        if not experts:
            return InterpretLens()
        blocks = [
            "",
            f"--- {self.lens_label} LENS ---",
            f"Interpret the results through the following {self.domain_adjective} expert lens(es):",
        ]
        for e in experts:
            blocks.append(f"[{e.name}] {e.interpret_lens}")
        if ctx.is_council and len(experts) > 1:
            blocks.append(
                "COUNCIL SYNTHESIS: present each expert's perspective briefly, then synthesize "
                "a single coherent recommendation that reconciles them. Avoid repeating the same "
                "point across experts."
            )
        blocks.append(
            f"Keep the {self.domain_adjective} framing concrete and actionable; ground every "
            "statement in the actual numbers provided."
        )
        return InterpretLens(system_suffix="\n".join(blocks), is_council=ctx.is_council)
