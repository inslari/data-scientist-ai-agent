from __future__ import annotations

from src.state import AgentState
from src.services.quality_gate import evaluate_quality_gate
from src.services.critic_advisor import advise


def critic_node(state: AgentState) -> AgentState:
    print("\n========================================\n[NODE START] critic_node başlıyor...\n========================================")
    print(f"[critic_node] BAŞLIYOR — Teşhis tabanlı self-correction")

    evaluation_report = state.get("evaluation_report", {})
    intent = state.get("intent", "classification")
    retry_history = state.get("retry_history") or []
    current_feedback = state.get("critic_feedback", [])

    # Sadece supervised ML'i critique et
    if intent not in ("classification", "regression"):
        print("[critic_node] Unsupervised/rapor görevi. Critique atlanıyor.")
        return {**state, "current_step": "critic_node"}

    if not evaluation_report:
        existing_error = state.get("error")
        return {
            **state,
            "current_step": "critic_node",
            "error": existing_error or "Evaluation report bulunamadı",
        }

    # Quality gate (her zaman)
    quality_gate = evaluate_quality_gate(state)


    # Geçici debug — per-class verisinin nerede olduğunu bul
    print(f"[DEBUG] evaluation_report keys: {list(evaluation_report.keys())}")
    print(f"[DEBUG] per_class_metrics: {evaluation_report.get('per_class_metrics')}")

    # Teşhis + retry kararı
    decision = advise(state)

    # retry_history'ye bu turu ekle
    new_history = retry_history + [{
        "score": decision["current_score"],
        "primary_type": decision["primary_type"],
    }]

    if decision["should_retry"]:
        primary = decision["primary_diagnosis"]
        print(f"[critic_node] RETRY → {decision['next_node']} "
              f"(teşhis: {primary['type']}, severity: {primary['severity']})")
        for fb in decision["feedback"]:
            print(f"[critic_node]   {fb}")
        return {
            **state,
            "retry_count": state.get("retry_count", 0) + 1,
            "critic_feedback": current_feedback + decision["feedback"],
            "retry_history": new_history,
            "next_retry_node": decision["next_node"],
            "quality_gate": quality_gate,
            "current_step": "critic_node",
        }

    # Retry yok — dur
    print(f"[critic_node] DURDU: {decision['stop_reason']}")
    return {
        **state,
        "retry_history": new_history,
        "next_retry_node": "report_node",
        "quality_gate": quality_gate,
        "current_step": "critic_node",
    }