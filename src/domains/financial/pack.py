"""The Financial / Merchant Intelligence pack.

Composes the five experts' domain knowledge into the four pipeline hooks. All
hooks are pure and return plain text that the calling node appends to its own
prompt; an empty/blank return means "no change".
"""

from __future__ import annotations

from typing import Optional

from src.domains.base import DomainPack, InterpretLens, MerchantContext, trim_doc
from src.domains.schemas import MerchantAnalysis
from . import router as _router
from .experts import ALL_EXPERTS


class FinancialPack(DomainPack):
    name = "financial"
    label = "Finansal / Merchant Intelligence"
    experts = ALL_EXPERTS

    # ── Expert selection ────────────────────────────────────────────────────
    def select_experts(
        self,
        user_query: str,
        db_intelligence: Optional[dict] = None,
        domain_doc: str = "",
    ) -> MerchantContext:
        return _router.route(user_query, db_intelligence, domain_doc)

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
        from .agents import run_analysis
        return await run_analysis(
            expert_keys=expert_keys,
            is_council=ctx.is_council,
            primary_expert=ctx.primary_expert,
            user_query=user_query,
            db_intelligence=db_intelligence or {},
            domain_doc=domain_doc or "",
            data_group_id=data_group_id,
            user_id=user_id,
            source_type=source_type,
        )

    def _expert_names(self, ctx: MerchantContext) -> str:
        experts = self.active_experts(ctx)
        return ", ".join(e.name for e in experts) if experts else "Financial Analyst"

    # ── Touch point 1: intent discovery ──────────────────────────────────────
    def enrich_intent_prompt(self, ctx: MerchantContext, domain_doc: str = "") -> str:
        experts = self.active_experts(ctx)
        if not experts:
            return ""
        biases = sorted({b for e in experts for b in e.intent_bias})
        lines = [
            "--- FINANCIAL DOMAIN CONTEXT (Merchant Intelligence Engine) ---",
            f"Active expert(s): {self._expert_names(ctx)}.",
        ]
        if ctx.domain_framing:
            lines.append(f"Framing: {ctx.domain_framing}")
        if ctx.sub_intent:
            lines.append(f"Analytical goal: {ctx.sub_intent}")
        lines.append(
            "This is financial/payments data. Typical intents for such requests: "
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
            blocks = ["--- MERCHANT INTELLIGENCE QUERY GUIDANCE (from data investigation) ---"]
            for f in analysis.findings:
                if f.key_signals:
                    blocks.append(f"[{f.expert}] observed: " + "; ".join(f.key_signals))
            blocks.append("Use these data-grounded signals; reference only existing columns.")
            return "\n".join(blocks)
        # Fallback: static expert hints
        experts = self.active_experts(ctx)
        if not experts:
            return ""
        blocks = ["--- FINANCIAL DOMAIN QUERY GUIDANCE ---"]
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
            blocks = ["", "--- MERCHANT INTELLIGENCE FEATURE GUIDANCE (from data investigation) ---"]
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
        # Fallback: static expert hints
        experts = self.active_experts(ctx)
        if not experts:
            return ""
        blocks = ["", "--- FINANCIAL DOMAIN FEATURE GUIDANCE ---"]
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
            blocks = ["", "--- MERCHANT INTELLIGENCE LENS (agentic analysis) ---"]
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
        # Fallback: static expert lenses
        experts = self.active_experts(ctx)
        if not experts:
            return InterpretLens()
        blocks = [
            "",
            "--- MERCHANT INTELLIGENCE LENS ---",
            "Interpret the results through the following financial expert lens(es):",
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
            "Keep the financial framing concrete and actionable; ground every statement in the "
            "actual numbers provided."
        )
        return InterpretLens(system_suffix="\n".join(blocks), is_council=ctx.is_council)
