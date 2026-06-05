from __future__ import annotations

import json
from src.state import AgentState

# Retry gap
RETRY_OVERFITTING_GAP = 0.15

def critic_node(state: AgentState) -> AgentState:
    """
    Evaluates the final model performance and checks for critical issues.
    If issues are found and retries are available, it sends feedback back to the pipeline.
    """
    print("\n========================================\n[NODE START] critic_node başlıyor...\n========================================")
    print(f"\n{'='*60}")
    print(f"[critic_node] BAŞLIYOR — Model Değerlendirmesi ve Self-Correction")
    print(f"{'='*60}")

    evaluation_report = state.get("evaluation_report", {})
    intent = state.get("intent", "classification")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    current_feedback = state.get("critic_feedback", [])
    
    # We only critique supervised ML tasks.
    if intent not in ("classification", "regression"):
        print("[critic_node] Unsupervised veya rapor görevi. Critique atlanıyor.")
        return {**state, "current_step": "critic_node"}
        
    if not evaluation_report:
        existing_error = state.get("error")
        error_msg = existing_error if existing_error else "Evaluation report bulunamadı"
        return {**state, "current_step": "critic_node", "error": error_msg}

    task = evaluation_report.get("task", "")
    test_metrics = evaluation_report.get("test_metrics", {})
    overfitting = evaluation_report.get("overfitting", {})
    
    issues = []
    action_needed = None
    
    # Overfitting Check
    ovf_status = overfitting.get("status", "unknown")
    gap = overfitting.get("gap", 0)
    if ovf_status == "overfitting_detected":
        if gap >= RETRY_OVERFITTING_GAP:
            issues.append(f"Ciddi Overfitting tespit edildi (fark: {gap:.4f} ≥ {RETRY_OVERFITTING_GAP}). Model daha basit olmalı, hiperparametreler ile 'Strong Regularization' uygulanacak.")
            action_needed = "automl_node"  # Hyperparameter tuning
        else:
            print(f"[critic_node] Hafif overfitting (fark: {gap:.4f} < {RETRY_OVERFITTING_GAP}). Retry gereksiz, devam ediliyor.")


    # Performance Metric Check
    if "classification" in task:
        auc = test_metrics.get("roc_auc", test_metrics.get("roc_auc_ovr", test_metrics.get("accuracy", 0)))
        if auc < 0.65:
            issues.append(f"Model performansı çok düşük (AUC/Accuracy: {auc}). Daha fazla veya daha iyi feature'lar gerekiyor.")
            action_needed = "feature_plan_node"
    elif "regression" in task:
        r2 = test_metrics.get("r2", 0)
        if r2 < 0.40:
            issues.append(f"Model performansı çok düşük (R²: {r2}). Veri hedefe açıklayıcı değil veya daha iyi feature'lar gerekiyor.")
            action_needed = "feature_plan_node"

    # Data Leakage Check
    # This will be populated by feature_engineering_node
    leakage_warnings = state.get("feature_report", {}).get("leakage_warnings", [])
    if leakage_warnings:
        issues.append(f"Data Leakage riski tespit edildi: {', '.join(leakage_warnings)}")
        action_needed = "feature_plan_node"

    if issues:
        print(f"[critic_node] Sorunlar bulundu: {issues}")
        if retry_count < max_retries:
            print(f"[critic_node] RETRY TETİKLENDİ ({retry_count + 1}/{max_retries}). Yönlendirme: {action_needed}")
            new_feedback = current_feedback + issues
            return {
                **state,
                "retry_count": retry_count + 1,
                "critic_feedback": new_feedback,
                "current_step": "critic_node",
                # We use a special field for routing
                "next_retry_node": action_needed
            }
        else:
            print(f"[critic_node] Max retries ({max_retries}) aşıldı. Mevcut modelle devam ediliyor.")
            
    print(f"[critic_node] Model kabul edildi veya retry hakkı kalmadı.")
    print(f"[DEBUG FE OUT] domain_leakage_report var mi: {state.get('domain_leakage_report') is not None}")
    return {
        **state,
        "current_step": "critic_node",
        "next_retry_node": "report_node"
    }
