from __future__ import annotations

"""
Report Node
-----------
Pipeline'ın son node'u. Tüm önceki node'ların çıktılarını alarak
LLM aracılığıyla kapsamlı, insan okunabilir bir Markdown raporu üretir.

Kullandığı state alanları:
    user_query          → Kullanıcının orijinal sorusu
    intent              → Analiz tipi
    relevant_tables     → Kullanılan tablolar
    analysis_plan       → Planner'ın belirlediği adımlar
    profiling_report    → Veri kalitesi ve istatistikleri
    feature_plan        → LLM'in feature engineering planı
    feature_report      → Uygulanan dönüşümlerin özeti
    model_name          → Seçilen model
    metrics             → AutoML metrikleri
    evaluation_report   → Kapsamlı evaluation sonuçları
    optimal_threshold   → Youden's J ile bulunan optimal karar eşiği

Çıktı:
    report              → Markdown formatında final rapor
    current_step        → "report_node"
"""

import os
from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState


# ─────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    from dotenv import load_dotenv
    load_dotenv()
    
    # Projenin geri kalanında OpenAI kullanıldığı için burayı da uyumlu yaptık
    return ChatOpenAI(
        model="gpt-4o-mini", # Raporlama için gpt-4o-mini yeterince iyi ve çok daha ucuz
        temperature=0.3
    )


# ─────────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────

def _build_context(state: AgentState) -> str:
    """
    State'teki tüm node çıktılarını LLM'e göndermek için
    yapılandırılmış bir metin bloğuna dönüştürür.
    """
    lines: list[str] = []

    # ── Kullanıcı Sorusu ─────────────────────────────────────────
    lines.append(f"## Kullanıcı Sorusu\n{state.get('user_query', 'Belirtilmemiş')}\n")

    # ── Intent ───────────────────────────────────────────────────
    intent = state.get("intent", "unknown")
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

    # ── Feature Engineering Sonucu ────────────────────────────────
    fr = state.get("feature_report", {})
    if fr:
        lines.append(
            f"## Feature Engineering Sonucu\n"
            f"- Final feature sayısı: {fr.get('n_features')}\n"
            f"- Eğitim seti: {fr.get('n_train')} satır\n"
            f"- Test seti: {fr.get('n_test')} satır\n"
            f"- Hedef dağılımı (train): {fr.get('target_distribution_train')}\n"
        )

    # ── AutoML Sonuçları ──────────────────────────────────────────
    metrics = state.get("metrics", {})
    model_name = state.get("model_name", "unknown")
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

    # ── Evaluation Raporu ─────────────────────────────────────────
    ev = state.get("evaluation_report", {})
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

        lines.append(f"## Evaluation Sonuçları\n### Test Metrikleri")
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

        if opt_thresh:
            lines.append(f"\n- Optimal Threshold (Youden's J): {opt_thresh}")

        lines.append(
            f"\n### Güvenilirlik\n"
            f"  - Skor: {rel.get('score')}/100 | Grade: {rel.get('grade')}\n"
            f"  - Faktörler: {', '.join(rel.get('factors', []))}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────

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

## 6. En Etkili Özellikler (Most Impactful Features)
- Compare and interpret SHAP values and Feature Importance.
- Explain the impact of each top feature on the target variable using **real-world domain knowledge**, not just mathematical impact.
- For example, deduce logical causalities like: "Capital gain strongly impacts the model because it directly represents wealth and investment income."
- DO NOT use circular, generic, or repetitive statements like "This feature is important because it had the highest contribution score."

## 7. Operasyonel Öneriler (Operational Recommendations)
- Discuss the Optimal Threshold (if provided) and when it should be used instead of the default 0.5.
- What needs to be done before putting this model into production?
- Suggest improvements like regularization or simpler models if applicable.

## 8. Sonraki Adımlar (Next Steps)
Propose 3-5 concrete, prioritized actionable steps.

---
RULES:
- OUTPUT LANGUAGE MUST BE TURKISH.
- Use technical terms but briefly explain them.
- Put numbers into context (e.g., "0.96 AUC is excellent because...").
- Do not repeat yourself.
- Keep the tone professional and fluent.
- Preserve the exact Markdown formatting (headers, bullet points, bold text).
"""


# ─────────────────────────────────────────────────────────────────
# MAIN NODE
# ─────────────────────────────────────────────────────────────────

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

    # Hata kontrolü — evaluation gelmediyse rapor üretemeyiz
    ev = state.get("evaluation_report", {})
    if not ev:
        error_msg = "Evaluation raporu bulunamadı, report_node atlanıyor."
        print(f"[report] UYARI: {error_msg}")
        return {
            **state,
            "report": f"# Rapor Üretilemedi\n\n{error_msg}",
            "current_step": "report_node",
        }

    # Context oluştur
    context = _build_context(state)
    print(f"[report] Context hazırlandı ({len(context)} karakter)")

    # LLM'e gönder
    llm = _get_llm()
    user_query = state.get("user_query", "")
    intent = state.get("intent", "classification")
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
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        report_text = response.content

        # Başlık satırını güzelleştir — modeli ve tarihi ekle
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
        header_note = (
            f"\n> **Model:** {model_name} | "
            f"**Intent:** {intent} | "
            f"**Tarih:** {timestamp}\n"
        )
        # İlk başlığın hemen altına ekle
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
        # LLM başarısız olursa yapılandırılmış fallback rapor üret
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


# ─────────────────────────────────────────────────────────────────
# FALLBACK RAPOR (LLM başarısız olursa)
# ─────────────────────────────────────────────────────────────────

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

    opt_thresh = ev.get("optimal_threshold")
    if opt_thresh:
        lines += [
            "",
            "## 7. Operasyonel Öneri",
            f"- Varsayılan 0.5 yerine **{opt_thresh}** threshold'u kullanmak recall'u artırır.",
        ]

    lines += [
        "",
        "## 8. Güvenilirlik",
        f"- Skor: **{score}/100** | Grade: **{grade}**",
    ]
    for factor in rel.get("factors", []):
        lines.append(f"  - ✓ {factor}")

    return "\n".join(lines)
