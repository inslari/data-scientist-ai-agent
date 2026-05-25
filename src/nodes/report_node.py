from __future__ import annotations

"""
Report Node
-----------
The final node of the pipeline. Synthesizes outputs from all previous nodes
to generate a comprehensive, human-readable Markdown report via LLM.

Utilized state fields:
    user_query          → The user's original question
    intent              → Analysis type
    relevant_tables     → Used database tables
    analysis_plan       → Steps defined by the planner
    profiling_report    → Data quality and statistics
    feature_plan        → LLM's feature engineering plan
    feature_report      → Summary of applied transformations
    model_name          → Selected model
    metrics             → AutoML metrics
    evaluation_report   → Comprehensive evaluation results
    optimal_threshold   → Optimal decision threshold found via Youden's J

Output:
    report              → Final report in Markdown format
    current_step        → "report_node"
"""

import os
from datetime import datetime
from typing import Any

from src.core.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from src.core.config import settings
from src.state import AgentState


# LLM Setup

def _get_llm():
    from dotenv import load_dotenv
    load_dotenv()
    
    
    return get_llm(model_name=settings.DEFAULT_LLM_MODEL, temperature=0.3)


# Context Builder

def _build_context(state: AgentState) -> str:
    """Formats all state outputs into a structured text block for the LLM."""
    lines: list[str] = []
    intent = state.get("intent", "unknown")

    # ── Kullanıcı Sorusu ─────────────────────────────────────────────
    lines.append(f"## Kullanıcı Sorusu\n{state.get('user_query', 'Belirtilmemiş')}\n")

    # ── Intent ──────────────────────────────────────────────
    confidence = state.get("intent_confidence", 0)
    tables = state.get("relevant_tables", [])
    lines.append(
        f"## Analiz Tipi\n"
        f"- Intent: {intent} (güven: {confidence})\n"
        f"- Tablolar: {', '.join(tables) if tables else 'Yok'}\n"
    )

    # ── Analysis Plan ─────────────────────────────────────────────
    plan = state.get("analysis_plan", [])
    if plan:
        lines.append("## Analiz Planı")
        for step in plan:
            lines.append(f"  {step.get('order', '?')}. [{step.get('step')}] {step.get('description', '')}")
        lines.append("")

    # ── Veri Profili ─────────────────────────────────────────────
    profiling = state.get("profiling_report", {})
    if profiling:
        shape = profiling.get("shape", {})
        recs = profiling.get("recommendations", [])
        lines.append(
            f"## Veri Profili\n"
            f"- Boyut: {shape.get('rows')} satır × {shape.get('columns')} kolon\n"
            f"- Öneri sayısı: {len(recs)}"
        )
        if recs:
            for r in recs[:5]:
                lines.append(f"  - {r}")
        lines.append("")

    # ── Feature Plan ─────────────────────────────────────────────
    fp = state.get("feature_plan", {})
    if fp:
        lines.append(
            f"## Feature Engineering Planı\n"
            f"- Hedef: {fp.get('target_column', '?')} — {fp.get('target_definition', '')}\n"
            f"- Analiz özeti: {fp.get('analysis_summary', '')}\n"
            f"- Feature grupları: {', '.join(fp.get('feature_groups', []))}\n"
        )

    # ── Feature Engineering Sonucu ─────────────────────────────────
    fr = state.get("feature_report", {})
    if fr:
        lines.append(
            f"## Feature Engineering Sonucu\n"
            f"- Final feature sayısı: {fr.get('n_features')}\n"
            f"- Eğitim seti: {fr.get('n_train')} satır\n"
            f"- Test seti: {fr.get('n_test')} satır\n"
        )

    # ── Task-specific Evaluation Context ────────────────────────────
    ev = state.get("evaluation_report", {})
    metrics = state.get("metrics", {})
    model_name = state.get("model_name", "unknown")

    if intent == "clustering":
        # ── Clustering Specific ──────────────────────────────────────
        cluster_dist = ev.get("cluster_distribution", {})
        algo_scores = ev.get("all_algorithm_scores", {})
        lines.append("## Kümeleme Sonuçları")
        lines.append(f"- Seçilen algoritma: {ev.get('best_algorithm', model_name).upper()}")
        lines.append(f"- Küme sayısı: {ev.get('n_clusters', '?')}")
        sil = ev.get('silhouette_score')
        if sil is not None:
            qual = 'Mükemmel' if sil > 0.7 else 'İyi' if sil > 0.5 else 'Orta' if sil > 0.25 else 'Zayıf'
            lines.append(f"- Silhouette Score: {sil:.4f} ({qual} — -1 ile 1 arasında, 1'e yakın = mükemmel ayrışma)")
        db = ev.get("davies_bouldin")
        ch = ev.get("calinski_harabasz")
        if db is not None:
            lines.append(f"- Davies-Bouldin Index: {db:.4f} (0'a yakın = daha iyi küme ayrışması)")
        if ch is not None:
            lines.append(f"- Calinski-Harabasz Score: {ch:.0f} (yüksek = daha kompakt ve ayrı kümeler)")
        lines.append("")
        if cluster_dist:
            lines.append("### Küme Dağılımı (Profil Verileri)")
            for cname, cinfo in cluster_dist.items():
                if cname != "noise":
                    cnt = cinfo.get('count', '?')
                    pct = cinfo.get('pct', 0)
                    lines.append(f"  - {cname}: {cnt:,} örnek (%{pct:.1f} oranında)")
            lines.append("")
        if algo_scores:
            lines.append("### Algoritma Karşılaştırması (Silhouette)")
            for algo, sc in sorted(algo_scores.items(), key=lambda x: x[1], reverse=True):
                marker = " ← SEÇİLEN" if algo == ev.get("best_algorithm") else ""
                lines.append(f"  - {algo.upper()}: {sc:.4f}{marker}")
            lines.append("")

    elif intent == "anomaly_detection":
        # ── Anomaly Detection Specific ──────────────────────────────
        cont_scan = ev.get("contamination_scan", {})
        fi_anom = ev.get("feature_importance", {})
        n_anom = ev.get("n_anomalies", 0)
        n_norm = ev.get("n_normal", 0)
        total = n_anom + n_norm
        lines.append("## Anomaly Detection Sonuçları")
        lines.append(f"- Algoritma: IsolationForest")
        lines.append(f"- Tespit edilen anomali: {n_anom:,} kayıt (%{ev.get('anomaly_pct', 0):.2f} oranında)")
        lines.append(f"- Normal kayıt: {n_norm:,}")
        lines.append(f"- Toplam: {total:,} kayıt")
        best_cont = ev.get("best_contamination")
        score_thr = ev.get("score_threshold")
        if best_cont:
            lines.append(f"- Seçilen contamination parametresi: {best_cont} (toplam verinin tahmini anomali oranı)")
        if score_thr:
            lines.append(f"- Anomali karar eşiği (score): {score_thr:.4f}")
        lines.append("")
        if cont_scan:
            lines.append("### Contamination Taraması (Farklı eşikler)")
            for cont, info in cont_scan.items():
                marker = " ← SEÇİLEN" if str(cont) == str(best_cont) else ""
                lines.append(f"  - contamination={cont}: {info.get('n_anomalies')} anomali (%{info.get('anomaly_pct', 0):.1f}){marker}")
            lines.append("")
        if fi_anom:
            lines.append("### En Etkili Özellikler (Anomali Sürücleri)")
            total_imp = sum(fi_anom.values()) or 1
            for feat, imp in list(fi_anom.items())[:8]:
                lines.append(f"  - {feat}: %{imp/total_imp*100:.1f}")
            lines.append("")

    else:
        # ── Supervised (classification / regression) ──────────────────
        if metrics:
            lines.append(f"## AutoML Sonuçları\n- Seçilen model: {model_name}")
            for k, v in metrics.items():
                if k not in ("per_class_report", "feature_importance", "task", "all_model_scores"):
                    lines.append(f"  - {k}: {v}")
            all_scores = metrics.get("all_model_scores", {})
            if all_scores:
                lines.append("- Tüm model skorları:")
                for m, s in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"    {m}: {s:.4f}")
            lines.append("")

        if ev:
            test_m = ev.get("test_metrics", {})
            ovf = ev.get("overfitting", {})
            cal = ev.get("calibration", {})
            ci = ev.get("confidence_intervals", {})
            rel = ev.get("reliability_score", {})
            shap = ev.get("shap_importance", {})
            err = ev.get("error_analysis", {})
            fi_sum = ev.get("feature_importance_summary", {})
            opt_thresh = ev.get("optimal_threshold")

            lines.append("## Evaluation Sonuçları\n### Test Metrikleri")
            for k, v in test_m.items():
                lines.append(f"  - {k}: {v}")

            if ci:
                lines.append("\n### Bootstrap Güven Aralıkları (%95)")
                for metric, vals in ci.items():
                    lines.append(f"  - {metric}: {vals['mean']} [{vals['lower']} — {vals['upper']}]")

            lines.append(
                f"\n### Overfitting\n"
                f"  - Train: {ovf.get('train_score')} | Test: {ovf.get('test_score')}\n"
                f"  - Fark: {ovf.get('gap')} | Durum: {ovf.get('status')}\n"
                f"  - {ovf.get('description')}"
            )

            if cal:
                lines.append(
                    f"\n### Kalibrasyon\n"
                    f"  - Brier Score: {cal.get('brier_score')}\n"
                    f"  - İyi kalibre: {cal.get('is_well_calibrated')}"
                )

            if err:
                conf = err.get("confidence_analysis", {})
                lines.append(
                    f"\n### Hata Analizi\n"
                    f"  - Hatalı tahmin: {err.get('n_errors')}/{err.get('n_total')} ({err.get('error_rate', 0)*100:.2f}%)\n"
                    f"  - Yüksek güven hatası: {conf.get('high_confidence_errors', 0)} adet ({conf.get('high_confidence_errors_pct', 0)}%)"
                )
                feat_diff = err.get("feature_distribution_diff", {})
                if feat_diff:
                    lines.append("  - En farklı feature'lar (Cohen's d):")
                    for feat, info in list(feat_diff.items())[:3]:
                        lines.append(f"      {feat}: d={info['cohens_d']} ({info['effect_magnitude']})")

            if shap:
                lines.append("\n### SHAP (Açıklanabilir AI)")
                for feat, pct in list(shap.get("top5_shap_features", {}).items())[:5]:
                    lines.append(f"  - {feat}: %{pct}")

            if fi_sum:
                lines.append(
                    f"\n### Feature Importance\n"
                    f"  - Top 5 toplam katkı: %{fi_sum.get('top5_total_contribution_pct')}\n"
                    f"  - Önemsiz feature: {fi_sum.get('insignificant_features_count', 0)} adet"
                )

            bias = ev.get("bias_analysis", {})
            if bias:
                lines.append("\n### Bias & Fairness Analizi")
                for base_col, cols in bias.items():
                    lines.append(f"  - Grup: {base_col}")
                    for col, metrics in cols.items():
                        lines.append(f"    - {col}: FPR={metrics.get('fpr', 0):.3f}, FNR={metrics.get('fnr', 0):.3f}")

            if opt_thresh:
                lines.append(f"\n- Optimal Threshold (Youden's J): {opt_thresh}")

            lines.append(
                f"\n### Güvenilirlik\n"
                f"  - Skor: {rel.get('score')}/100 | Grade: {rel.get('grade')}\n"
                f"  - Faktörler: {', '.join(rel.get('factors', []))}"
            )

    return "\n".join(lines)


# System Prompts — task-specific

SYSTEM_PROMPT = """You are an expert Data Scientist. You will be provided with the complete outputs of an ML pipeline.
Your task is to analyze these outputs and write a comprehensive, clear, and professional analysis report.

CRITICAL: The final output MUST be written entirely in fluent, professional Turkish.

Write the report strictly using the following Markdown structure:

# 📊 Analiz Raporu — [Title]

## 1. Yönetici Özeti (Executive Summary)
Provide a 3-4 sentence summary that a non-technical stakeholder can understand.
What problem was solved, how good is the model, and what is the key finding?

## 2. Veri Kalitesi (Data Quality)
- Evaluate the overall state of the dataset.
- Summarize the cleaning operations applied.
- Highlight any notable data quality issues or outliers.

## 3. Model Performansı (Model Performance)
- Interpret the most important metrics (do not just list numbers; explain what they mean).
- Evaluate the reliability of the performance using the Bootstrap Confidence Intervals.
- Compare the chosen model against other evaluated models.

## 4. Güçlü Yönler (Strengths)
- Areas where the model excels.
- Evaluate the model's calibration (e.g., Brier Score).
- Key insights derived from SHAP values.

## 5. Zayıf Yönler ve Riskler (Weaknesses & Risks)
- Interpret the overfitting status (Train vs Test gap).
- Evaluate "High Confidence Errors" (cases where the model was certain but wrong).
- Draw actionable conclusions from the Error Analysis section.

## 6. Adalet ve Yanlılık Analizi (Fairness & Bias)
- Analyze the Bias & Fairness metrics (e.g., FPR, FNR differences across demographic groups) if they exist.
- Identify if the model systematically disadvantages certain groups.
- If biases are found, explain the real-world implications (e.g., "The model rejects loan applications from Group X more frequently").
- If no bias data is present, state that it was not evaluated.

## 7. En Etkili Özellikler (Most Impactful Features)
- Compare and interpret SHAP values and Feature Importance.
- Explain the impact of each top feature on the target variable using **real-world domain knowledge**, not just mathematical impact.
- For example, deduce logical causalities like: "Capital gain strongly impacts the model because it directly represents wealth and investment income."
- DO NOT use circular, generic, or repetitive statements like "This feature is important because it had the highest contribution score."

## 8. Operasyonel Öneriler (Operational Recommendations)
- Discuss the Optimal Threshold (if provided) and when it should be used instead of the default 0.5.
- What needs to be done before putting this model into production?
- Suggest improvements like regularization or simpler models if applicable.

## 9. Sonraki Adımlar (Next Steps)
Propose 3-5 concrete, prioritized actionable steps.

---
RULES:
- OUTPUT LANGUAGE MUST BE TURKISH.
- THINK DEEPLY. Extract hidden meanings, logical causalities, and business implications from the numbers. Do not just state the numbers. Be a highly experienced senior consultant.
- Use technical terms but briefly explain them.
- Put numbers into context (e.g., "0.96 AUC is excellent because...").
- Do not repeat yourself.
- Keep the tone professional, authoritative, and fluent.
- Preserve the exact Markdown formatting (headers, bullet points, bold text).
"""

SYSTEM_PROMPT_CLUSTERING = """You are an expert Data Scientist specializing in unsupervised learning and customer segmentation.

CRITICAL: The final output MUST be written entirely in fluent, professional Turkish.

Write the report strictly using the following Markdown structure:

# 📊 Kümeleme Analizi Raporu — [Başlık]

## 1. Yönetici Özeti (Executive Summary)
Provide a 3-4 sentence summary: how many segments were found, how well-separated they are (Silhouette Score), and what it means for the business.

## 2. Veri Kalitesi (Data Quality)
- Evaluate the overall state of the dataset.
- Summarize any cleaning operations applied and their impact.
- Highlight outliers or quality issues that could affect cluster quality.

## 3. Küme Kalitesi (Cluster Quality)
- Interpret the Silhouette Score in plain language (what does the number mean, is it good?).
- Discuss Davies-Bouldin Index and Calinski-Harabasz Score if available.
- Compare KMeans vs DBSCAN performance from the algorithm comparison table.
- Explain WHY the selected algorithm outperformed the alternatives.

## 4. Küme Profilleri (Cluster Profiles)
- For each cluster, describe its relative size and infer likely characteristics.
- Use domain knowledge to interpret what each segment likely represents.
- Example: "Küme 0, toplam verinin %X'ini oluşturmaktadır — muhtemelen..."

## 5. Güçlü Yönler (Strengths)
- Where the segmentation model excels.
- Explain why the Silhouette score indicates good/excellent cluster separation.

## 6. Zayıf Yönler ve Riskler (Weaknesses & Risks)
- Potential issues: do the clusters make business sense?
- Risks of over-interpreting cluster boundaries without domain validation.
- Limitations of the chosen algorithm (e.g., KMeans assumes spherical clusters).

## 7. Operasyonel Öneriler (Operational Recommendations)
- How to use these segments in real business decisions.
- What domain experts should validate before acting on results.
- Whether additional features or data collection could improve segmentation.

## 8. Sonraki Adımlar (Next Steps)
Propose 3-5 concrete, prioritized actionable steps.

---
RULES:
- OUTPUT LANGUAGE MUST BE TURKISH.
- NEVER mention: Brier Score, ROC-AUC, confusion matrix, F1, precision, recall, SHAP, accuracy, overfitting, high confidence errors.
  These concepts do NOT apply to unsupervised clustering.
- Focus on cluster interpretability, business value, and actionable insights.
- THINK DEEPLY. Extract hidden meanings, logical causalities, and business implications from the numbers. Do not just state the numbers. Be a highly experienced senior consultant.
- Use technical terms but briefly explain them.
- Do not repeat yourself.
- Keep the tone professional, authoritative, and fluent.
"""

SYSTEM_PROMPT_ANOMALY = """You are an expert Data Scientist specializing in anomaly detection and fraud/outlier analysis.

CRITICAL: The final output MUST be written entirely in fluent, professional Turkish.

Write the report strictly using the following Markdown structure:

# 📊 Anomali Tespiti Raporu — [Başlık]

## 1. Yönetici Özeti (Executive Summary)
Provide a 3-4 sentence summary: how many anomalies were found, at what rate, and what business risk this signals.

## 2. Veri Kalitesi (Data Quality)
- Evaluate the overall state of the dataset.
- Summarize any cleaning operations and their impact on detection quality.

## 3. Anomali Tespiti Performansı (Detection Performance)
- Interpret the contamination parameter and why it was chosen.
- Explain the anomaly rate in business terms (is X% normal for this domain?).
- Compare the contamination scan results to justify the selected threshold.

## 4. Anomali Profili (Anomaly Profile)
- Interpret the feature importance: which features most drive anomaly scores?
- Use domain knowledge to explain WHY these features signal anomalies.
- Example: "Yüksek 'capital.gain' değerleri anomali skorunu artırıyor — bu, ortalama dışı finansal hareketlere işaret edebilir."

## 5. Güçlü Yönler (Strengths)
- Where the model excels at detecting outliers.
- Why IsolationForest is appropriate for this dataset.

## 6. Zayıf Yönler ve Riskler (Weaknesses & Risks)
- False positive risks (normal records flagged as anomalies).
- False negative risks (real anomalies missed) — especially critical for fraud detection.
- No ground-truth labels available — results need human validation.
- Limitations of IsolationForest.

## 7. Operasyonel Öneriler (Operational Recommendations)
- How to operationalize anomaly flags in a business workflow.
- Manual review strategy for flagged records.
- When to adjust the contamination parameter.

## 8. Sonraki Adımlar (Next Steps)
Propose 3-5 concrete, prioritized actionable steps.

---
RULES:
- OUTPUT LANGUAGE MUST BE TURKISH.
- NEVER mention: ROC-AUC, confusion matrix, F1, precision, recall, Brier Score, SHAP, accuracy, calibration.
  These require ground-truth labels which anomaly detection does NOT have.
- Focus on anomaly interpretation, business risk, feature drivers, and recommended actions.
- THINK DEEPLY. Extract hidden meanings, logical causalities, and business implications from the numbers. Do not just state the numbers. Be a highly experienced senior consultant.
- Use technical terms but briefly explain them.
- Do not repeat yourself.
- Keep the tone professional, authoritative, and fluent.
"""

# Main Node
def report_node(state: AgentState) -> AgentState:
    """
    Node 9: Pipeline'ın tüm çıktılarını LLM ile yorumlayarak
    kapsamlı bir Markdown raporu üretir.

    State girdileri (tüm önceki node'lardan):
        user_query, intent, relevant_tables, analysis_plan,
        profiling_report, feature_plan, feature_report,
        model_name, metrics, evaluation_report, optimal_threshold

    State çıktıları:
        report       : Markdown formatında final rapor
        current_step : "report_node"
    """
    print("\n[report] Başlıyor...")

    # control evaluation report
    ev = state.get("evaluation_report", {})
    if not ev:
        error_msg = "Evaluation raporu bulunamadı, report_node atlanıyor."
        print(f"[report] UYARI: {error_msg}")
        return {
            **state,
            "report": f"# Rapor Üretilemedi\n\n{error_msg}",
            "current_step": "report_node",
        }

    # build context
    context = _build_context(state)
    print(f"[report] Context hazırlandı ({len(context)} karakter)")

    # select system prompt based on intent
    intent = state.get("intent", "classification")
    if intent == "clustering":
        system_prompt = SYSTEM_PROMPT_CLUSTERING
    elif intent == "anomaly_detection":
        system_prompt = SYSTEM_PROMPT_ANOMALY
    else:
        system_prompt = SYSTEM_PROMPT

    # llm
    llm = _get_llm()
    user_query = state.get("user_query", "")
    model_name = state.get("model_name", "unknown")

    prompt = (
        f"Kullanıcı sorusu: '{user_query}'\n\n"
        f"Aşağıda pipeline'ın tüm çıktıları bulunmaktadır. "
        f"Bu verileri kullanarak kapsamlı bir analiz raporu yaz.\n\n"
        f"{context}"
    )

    print("[report] LLM'den rapor üretiliyor...")
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        report_text = response.content
        if isinstance(report_text, list):
            report_text = "".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in report_text])
        elif not isinstance(report_text, str):
            report_text = str(report_text)

        # create header note with model name and date 
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
        header_note = (
            f"\n> **Model:** {model_name} | "
            f"**Intent:** {intent} | "
            f"**Tarih:** {timestamp}\n"
        )
        # add note under first header
        if "# " in report_text:
            first_newline = report_text.find("\n")
            if first_newline != -1:
                report_text = (
                    report_text[:first_newline]
                    + header_note
                    + report_text[first_newline:]
                )

    except Exception as e:
        print(f"[report] LLM hatası: {e}")
        # create fallback report if llm fails 
        report_text = _generate_fallback_report(state)

    print(f"[report] Rapor tamamlandı ({len(report_text)} karakter)")
    print("\n" + "=" * 60)
    print("FINAL RAPOR (İlk 500 karakter):")
    print("=" * 60)
    print(report_text[:500] + "..." if len(report_text) > 500 else report_text)

    return {
        **state,
        "report": report_text,
        "current_step": "report_node",
    }


# fallback report (llm failes)
def _generate_fallback_report(state: AgentState) -> str:
    """LLM çağrısı başarısız olursa state verilerinden doğrudan rapor üretir."""
    ev = state.get("evaluation_report", {})
    test_m = ev.get("test_metrics", {})
    ovf = ev.get("overfitting", {})
    rel = ev.get("reliability_score", {})
    shap = ev.get("shap_importance", {})
    ci = ev.get("confidence_intervals", {})
    err = ev.get("error_analysis", {})
    cal = ev.get("calibration", {})
    model_name = state.get("model_name", "unknown")
    intent = state.get("intent", "unknown")
    tables = state.get("relevant_tables", [])
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")

    roc = test_m.get("roc_auc", test_m.get("roc_auc_ovr", "N/A"))
    acc = test_m.get("accuracy", "N/A")
    f1 = test_m.get("f1_weighted", "N/A")
    mcc = test_m.get("mcc", "N/A")
    grade = rel.get("grade", "?")
    score = rel.get("score", 0)

    lines = [
        f"# 📊 Analiz Raporu",
        f"> **Model:** {model_name} | **Intent:** {intent} | **Tarih:** {timestamp}",
        "",
        "## 1. Yönetici Özeti",
        f"- Analiz tipi: **{intent}**",
        f"- Kullanılan tablolar: {', '.join(tables)}",
        f"- En iyi model: **{model_name}**",
        f"- Genel güvenilirlik: **{score}/100 (Grade: {grade})**",
        "",
        "## 2. Model Performansı",
        f"| Metrik | Değer |",
        f"|--------|-------|",
    ]

    for k, v in test_m.items():
        lines.append(f"| {k} | {v} |")

    if ci:
        lines += ["", "### Bootstrap Güven Aralıkları (%95)"]
        for metric, vals in ci.items():
            lines.append(f"- **{metric}**: {vals['mean']} [{vals['lower']} — {vals['upper']}]")

    lines += [
        "",
        "## 3. Overfitting Analizi",
        f"- Train: {ovf.get('train_score')} | Test: {ovf.get('test_score')}",
        f"- Fark: {ovf.get('gap')} | Durum: **{ovf.get('status')}**",
        f"- {ovf.get('description', '')}",
    ]

    if cal:
        lines += [
            "",
            "## 4. Kalibrasyon",
            f"- Brier Score: {cal.get('brier_score')}",
            f"- İyi kalibre: {'✅' if cal.get('is_well_calibrated') else '❌'}",
        ]

    if shap:
        lines += ["", "## 5. En Etkili Özellikler (SHAP)"]
        for feat, pct in shap.get("top5_shap_features", {}).items():
            lines.append(f"- **{feat}**: %{pct}")

    if err:
        conf = err.get("confidence_analysis", {})
        lines += [
            "",
            "## 6. Hata Analizi",
            f"- Toplam hata: {err.get('n_errors')}/{err.get('n_total')} ({err.get('error_rate', 0)*100:.2f}%)",
            f"- Yüksek güven hatası: {conf.get('high_confidence_errors', 0)} adet",
        ]

    bias = ev.get("bias_analysis", {})
    if bias:
        lines += ["", "## 7. Bias & Fairness Analizi"]
        for base_col, cols in bias.items():
            lines.append(f"- **Grup: {base_col}**")
            for col, metrics in cols.items():
                lines.append(f"  - {col}: FPR={metrics.get('fpr', 0):.3f}, FNR={metrics.get('fnr', 0):.3f}")

    opt_thresh = ev.get("optimal_threshold")
    if opt_thresh:
        lines += [
            "",
            "## 8. Operasyonel Öneri",
            f"- Varsayılan 0.5 yerine **{opt_thresh}** threshold'u kullanmak recall'u artırır.",
        ]

    lines += [
        "",
        "## 9. Güvenilirlik",
        f"- Skor: **{score}/100** | Grade: **{grade}**",
    ]
    for factor in rel.get("factors", []):
        lines.append(f"  - ✓ {factor}")

    return "\n".join(lines)
