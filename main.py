"""
Data Scientist AI Agent
Entry point for the application. Initializes and runs the LangGraph pipeline.

Usage:
    uv run python main.py
"""

import sys

# Fix UTF-8 encoding issues on Windows console
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from src.graph import build_graph
from src.state import AgentState


def run_pipeline(query: str):
    print("=== Data Scientist Agent Başlatılıyor ===\n")
    print(f"Soru: '{query}'\n")
    
    app = build_graph()
    
    initial_state: AgentState = {
        "user_query": query,
        "messages": [],
    }

    try:
        result = app.invoke(initial_state)
        
        # Display results
        print("\n" + "=" * 60)
        print("PIPELINE ÖZET")
        print("=" * 60)

        print(f"\nIntent     : {result.get('intent')} (confidence: {result.get('intent_confidence')})")
        print(f"Tables     : {result.get('relevant_tables')}")

        if result.get("analysis_plan"):
            print("\nAnalysis Plan:")
            for step in result.get("analysis_plan"):
                print(f"  {step.get('order')}. [{step.get('step')}] {step.get('description')}")

        if result.get("profiling_report"):
            prof = result.get("profiling_report")
            print("\nExecution:")
            print(f"  SQL sonucu  : {len(result.get('sql_results', []))} satır çekildi")
            print(f"  Profiling   : {prof.get('shape', {}).get('rows')} satır x {prof.get('shape', {}).get('columns')} kolon")
            if prof.get("recommendations"):
                print("  Öneriler    :")
                for rec in prof.get("recommendations"):
                    print(f"    - {rec}")

        if result.get("feature_report"):
            fr = result.get("feature_report")
            print("\nFeature Engineering:")
            print(f"  Hedef değişken       : {fr.get('target_column')}")
            print(f"  Final feature sayısı : {fr.get('n_features')}")
            print(f"  Eğitim seti          : {fr.get('n_train')} satır")
            print(f"  Test seti            : {fr.get('n_test')} satır")
            print(f"  Hedef dağılımı       : {fr.get('target_distribution_train')}")

        metrics = result.get("metrics", {})
        if metrics:
            print(f"\nModel Sonuçları ({result.get('model_name')}):")
            for k, v in metrics.items():
                if k not in ("per_class_report", "feature_importance", "task", "all_model_scores"):
                    print(f"  {k}: {v}")

            fi = metrics.get("feature_importance", {})
            if fi:
                print("\n  Top 5 Feature Importance:")
                for feat, score in list(fi.items())[:5]:
                    print(f"    {feat}: {score:.4f}")

            all_scores = metrics.get("all_model_scores", {})
            if all_scores:
                print("\n  Tüm Model Skorları:")
                for m, s in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    if isinstance(s, float):
                        print(f"    {m:<25}: {s:.4f}")
                    else:
                        print(f"    {m:<25}: {s}")

        eval_report = result.get("evaluation_report", {})
        if eval_report:
            reliability = eval_report.get("reliability_score", {})
            overfitting = eval_report.get("overfitting", {})
            test_m = eval_report.get("test_metrics", {})
            print(f"\nEvaluation ({eval_report.get('model_name')}):")
            print(f"  Task      : {eval_report.get('task')}")
            for k, v in test_m.items():
                print(f"  {k}: {v}")
            print(f"  Overfitting: {overfitting.get('status')} (gap: {overfitting.get('gap')})")
            print(f"  Güvenilirlik: {reliability.get('score')}/100 — Grade: {reliability.get('grade')}")

        report = result.get("report", "")
        if report:
            print("\n" + "=" * 60)
            print("FINAL RAPOR")
            print("=" * 60)
            print(report)
            
    except Exception as e:
        print(f"\n❌ Pipeline execution error: {e}")


if __name__ == "__main__":
    # Default fallback query
    default_query = "Bu veri setindeki kişilerin gelir düzeyini tahmin et. Hangi faktörler gelirin 50K üzerinde olmasını etkiliyor?"
    
    # Parse CLI arguments
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = default_query

    run_pipeline(user_query)
