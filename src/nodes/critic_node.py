from __future__ import annotations
from src.core.nodes import node

from src.state import AgentState
from src.services.quality_gate import evaluate_quality_gate
from src.services.critic_advisor import advise

from src.core.logging_config import get_logger
logger = get_logger(__name__)


@node("critic_node")
def critic_node(state: AgentState) -> AgentState:
    logger.info(f"[critic_node] BAŞLIYOR — Teşhis tabanlı self-correction")

    evaluation_report = state.get("evaluation_report", {})
    intent = state.get("intent", "classification")

    # Sadece supervised ML'i critique et
    if intent not in ("classification", "regression"):
        logger.warning("[critic_node] Unsupervised/rapor görevi. Critique atlanıyor.")
        return {"current_step": "critic_node"}

    if not evaluation_report:
        existing_error = state.get("error")
        return {
            "current_step": "critic_node",
            "error": existing_error or "Evaluation report bulunamadı",
        }

    # Quality gate (her zaman)
    quality_gate = evaluate_quality_gate(state)


    # Geçici debug — per-class verisinin nerede olduğunu bul
    logger.info(f"[DEBUG] evaluation_report keys: {list(evaluation_report.keys())}")
    logger.info(f"[DEBUG] per_class_metrics: {evaluation_report.get('per_class_metrics')}")

    # Teşhis + retry kararı
    decision = advise(state)

    # Bu turun özeti — reducer (operator.add) sayesinde sadece delta dönüyoruz,
    # LangGraph mevcut retry_history'ye ekler.
    this_turn = {
        "score": decision["current_score"],
        "primary_type": decision["primary_type"],
    }

    if decision["should_retry"]:
        primary = decision["primary_diagnosis"]
        logger.info(f"[critic_node] RETRY → {decision['next_node']} "
              f"(teşhis: {primary['type']}, severity: {primary['severity']})")
        for fb in decision["feedback"]:
            logger.info(f"[critic_node]   {fb}")

        result = {
            "retry_count": state.get("retry_count", 0) + 1,
            "critic_feedback": decision["feedback"],   # reducer ekler (delta)
            "retry_history": [this_turn],              # reducer ekler (delta)
            "next_retry_node": decision["next_node"],
            "quality_gate": quality_gate,
            "current_step": "critic_node",
        }

        # Leakage şüpheli kolonları → feature_builder'ın retry'da DETERMINISTIK
        # düşürmesi için state'e biriktir (LLM'in drop_features'ına bağlı kalmadan).
        forced = decision.get("forced_drops") or []
        if forced:
            existing = state.get("forced_leakage_drops") or []
            merged = list(existing)
            for c in forced:
                if c not in merged:
                    merged.append(c)
            result["forced_leakage_drops"] = merged
            logger.info(f"[critic_node] Forced leakage drop (retry'da uygulanacak): {merged}")

        return result

    # Retry yok — dur
    logger.info(f"[critic_node] DURDU: {decision['stop_reason']}")
    return {
        "retry_history": [this_turn],                  # reducer ekler (delta)
        "next_retry_node": "report_node",
        "quality_gate": quality_gate,
        "current_step": "critic_node",
    }