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
from src.core.prompt_loader import load_prompt
from src.core.feature_naming import humanize_feature
from src.core.report_i18n import (
    normalize_lang,
    LANGUAGE_NAME,
    R,
    freq_label,
    split_label,
)
from src.domains import get_active_pack
from src.domains.base import MerchantContext
from src.domains.schemas import MerchantAnalysis
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# LLM Setup

def _get_llm():
    from dotenv import load_dotenv
    load_dotenv()
    
    
    return get_llm(model_name=settings.DEFAULT_LLM_MODEL, temperature=0.3)


# Context Builder

def _overfitting_scores(ovf: dict) -> tuple[Any, Any, str]:
    """Overfitting 1. skor/test skorunu şemadan bağımsız okur.
    CV varsa display_score (CV 5-fold) + display_label döner → grafikle TUTARLI metin
    (banka raporu hem grafiği hem metni okur; ikisi de CV-bazlı kalmalı). Eski raporlarda
    display_score yoksa train_score/train_r2'ye düşer (fallback — çökmesin)."""
    if ovf.get("display_score") is not None:
        test_s = ovf.get("test_score", ovf.get("test_r2"))
        return ovf.get("display_score"), test_s, ovf.get("display_label", "skor")
    if ovf.get("train_score") is not None:
        return ovf.get("train_score"), ovf.get("test_score"), "Train"
    if ovf.get("train_r2") is not None:
        return ovf.get("train_r2"), ovf.get("test_r2"), "Train"
    return None, None, ""


def _plainify_rec(rec: str) -> str:
    """Profiling önerisindeki iç işleme jargonunu (RobustScaler, IQR capping,
    one-hot/label/target encoding, log/sqrt transform) genel kullanıcı için sade
    ETKİYE çevirir. Metod adı LLM'e hiç verilmez → rapora sızamaz (deterministik).
    Sol taraf (kolon + sorun) korunur; '→ metod' kuyruğu sade etkiyle değiştirilir."""
    if "→" not in rec:
        return rec
    left, right = rec.split("→", 1)
    r = right.lower()
    if "encoding" in r or "one-hot" in r or "one hot" in r:
        eff = "kategoriler sayıya çevrildi"
    elif "transform" in r or "log" in r or "sqrt" in r:
        eff = "çarpık dağılım dengelendi"
    elif "scaler" in r or "capping" in r or "outlier" in r or "iqr" in r:
        eff = "uç değerler dengelendi"
    elif "drop" in r or "çıkar" in r:
        eff = "kolon modelden çıkarıldı"
    else:
        return left.strip()
    return f"{left.strip()} → {eff}"


def _build_context(state: AgentState, lang: str = "tr") -> str:
    """Formats all state outputs into a structured text block for the LLM.

    Çıktıya VERBATIM kopyalanan parçalar (status rozeti, Karar, FAIL/WARN) ``lang``
    diline lokalize edilir; geri kalan etiketler LLM girdisidir (yeniden yazılır).
    """
    lines: list[str] = []
    intent = state.get("intent", "unknown")

    # quality gate
    qg = state.get("quality_gate") or {}
    if qg:
        summary = qg.get("summary", "")

        # Durum rozeti — H1 DEĞİL; deterministik başlığın altında küçük durum satırı.
        # Kategorik "güvenilir/üretime alınmamalı" iddiası yok, sadece durum işareti.
        badge = {
            "pass": R("badge_pass", lang),
            "warn": R("badge_warn", lang),
            "fail": R("badge_fail", lang),
        }.get(qg.get("status"), "")

        if badge:
            lines.append(f"**{badge}**\n")
        lines.append(f"**{R('decision', lang)}:** {summary}\n")

        failures = qg.get("failures", [])
        if failures:
            lines.append(f"## {R('fail_header', lang)}")
            for f in failures:
                lines.append(f"  - {f['message']}")
            lines.append("")

        warnings = qg.get("warnings", [])
        if warnings:
            lines.append(f"## {R('warn_header', lang)}")
            for w in warnings:
                lines.append(f"  - {w['message']}")
            lines.append("")

    # user query
    lines.append(f"## Kullanıcı Sorusu\n{state.get('user_query', 'Belirtilmemiş')}\n")

    # intent
    confidence = state.get("intent_confidence", 0)
    tables = state.get("relevant_tables", [])
    lines.append(
        f"## Analiz Tipi\n"
        f"- Intent: {intent} (güven: {confidence})\n"
        f"- Tablolar: {', '.join(tables) if tables else 'Yok'}\n"
    )

    # analysis plan
    plan = state.get("analysis_plan", [])
    if plan:
        lines.append("## Analiz Planı")
        for step in plan:
            lines.append(f"  {step.get('order', '?')}. [{step.get('step')}] {step.get('description', '')}")
        lines.append("")

    # data profiling
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
                lines.append(f"  - {_plainify_rec(r)}")
        lines.append("")

    # feature plan
    fp = state.get("feature_plan", {})
    if fp:
        lines.append(
            f"## Feature Engineering Planı\n"
            f"- Hedef: {fp.get('target_column', '?')} — {fp.get('target_definition', '')}\n"
            f"- Analiz özeti: {fp.get('analysis_summary', '')}\n"
            f"- Feature grupları: {', '.join(fp.get('feature_groups', []))}\n"
        )

    # feat engineering result
    fr = state.get("feature_report", {})
    if fr:
        lines.append(
            f"## Feature Engineering Sonucu\n"
            f"- Final feature sayısı: {fr.get('n_features')}\n"
            f"- Eğitim seti: {fr.get('n_train')} satır\n"
            f"- Test seti: {fr.get('n_test')} satır\n"
        )
        # Zaman serisi feature'larını context'e taşı: LLM lag/rolling'i ve kronolojik
        # split'i doğru yorumlasın. Yalnızca gerçek timeseries_lag değerleri (anti-fabrication).
        ts = fr.get("timeseries_lag") or {}
        if ts and (state.get("modality") or "").strip().lower() == "timeseries":
            lines.append(
                f"- Zaman serisi: frekans={ts.get('frequency') or '?'}, "
                f"split={fr.get('split_strategy') or '?'}, "
                f"lag pencereleri={ts.get('windows') or []}, "
                f"rolling pencereleri={ts.get('roll_windows') or []}, "
                f"düşen lead satır={ts.get('dropped_lead_rows', 0)}"
                + (f", entity_col={ts.get('entity_col')}" if ts.get('entity_col') else "")
            )
            lines.append("")

    # domain leakage
    domain_leakage = state.get("domain_leakage_report", {})
    if domain_leakage:
        concerns = domain_leakage.get("concerns", [])
        assessment = domain_leakage.get("overall_assessment", "")
        
        if concerns:
            lines.append("\n## Leakage Suphesi (Domain Analizi)")
            lines.append(f"Genel degerlendirme: {assessment}\n")
            for c in concerns:
                severity_marker = {
                    "high": "[YUKSEK SUPHE]",
                    "medium": "[ORTA SUPHE]",
                    "low": "[DUSUK SUPHE]",
                }.get(c.get("confidence", "low"), "[SUPHE]")
                lines.append(
                    f"  {severity_marker} '{c.get('column_name')}' "
                    f"({c.get('relationship_type')}): {c.get('reasoning')}"
                )
            lines.append("")
            
            # high confidence drop info
            high_dropped = [c for c in concerns if c.get("confidence") == "high"]
            if high_dropped:
                cols = [c['column_name'] for c in high_dropped]
                lines.append(f"**OTOMATIK DROP EDILDI (high confidence leakage):** {', '.join(cols)}")
                lines.append("Bu feature'lar otomatik olarak modelden cikarildi. Asagidaki metrikler bu feature'lar olmadan egitilen modeli yansitir — yani gercek performans, leakage'li sahte yuksek skor degil.\n")
                
        elif assessment:
            lines.append(f"\n## Leakage Kontrolu\n{assessment}\n")
    
    # task-specific evaluation
    ev = state.get("evaluation_report", {})
    metrics = state.get("metrics", {})
    model_name = state.get("model_name", "unknown")

    if intent == "clustering":
        # clustering
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
        # anomaly detection specific
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
                lines.append(f"  - {humanize_feature(feat, lang)}: %{imp/total_imp*100:.1f}")
            lines.append("")

    else:
        # supervised (clustering - regression)
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
                lines.append(
                    "\n### Güven Aralığı (%95)\n"
                    "  (skorun tekrarlı örneklemeyle ne kadar oynadığı — dar aralık = sonuç güvenilir/kararlı)"
                )
                for metric, vals in ci.items():
                    lines.append(f"  - {metric}: {vals['mean']} [{vals['lower']} — {vals['upper']}]")
            else:
                lines.append("\n*Bootstrap güven aralığı bu çalışmada hesaplanmadı (yetersiz veri veya zaman aşımı).*")

            disp_s, test_s, ovf_label = _overfitting_scores(ovf)
            if ovf and disp_s is not None:
                _is_cv = str(ovf_label).upper().startswith("CV")
                ovf_lines = (
                    f"\n### Genelleme (CV çapraz-doğrulama vs Test)\n"
                    f"  - {ovf_label}: {disp_s} | Test: {test_s}\n"
                    f"  - Fark ({ovf_label}−Test): {ovf.get('gap')} | Durum: {ovf.get('status')}\n"
                    f"  - {ovf.get('description')}"
                )
                if _is_cv:
                    ovf_lines += (
                        f"\n  - OVERFIT DEĞERLENDİRMESİ: SADECE bu {ovf_label}−Test farkına bak. "
                        f"Fark küçükse model iyi genelleşiyordur; 'overfit var' YAZMA."
                    )
                # Train (resubstitution) skoru varsa: OVERFIT GÖSTERGESİ DEĞİL — LLM'i uyar.
                _tr = ovf.get("train_score", ovf.get("train_r2"))
                if _tr is not None:
                    ovf_lines += (
                        f"\n  - ÖNEMLİ: Train skoru {_tr} yapısal olarak yüksektir "
                        f"(CV-tabanlı ensemble modelinde ~%99 normaldir) ve OVERFIT GÖSTERGESİ DEĞİLDİR. "
                        f"Train ile Test'i KARŞILAŞTIRMA; overfit kararını yalnızca CV−Test farkına göre ver."
                    )
                lines.append(ovf_lines)
            else:
                lines.append("\n*Genelleme (overfit) analizi bu çalışmada uygulanmadı.*")

            if cal and isinstance(cal, dict) and cal.get('brier_score') is not None:
                lines.append(
                    f"\n### Olasılık Güvenilirliği (kalibrasyon)\n"
                    f"  - Brier Score: {cal.get('brier_score')} "
                    f"(olasılık tahminlerinin ne kadar gerçekçi olduğu; 0'a yakın = güvenilir)\n"
                    f"  - İyi kalibre: {cal.get('is_well_calibrated')}"
                )
            else:
                lines.append("\n*Kalibrasyon analizi bu çalışmada uygulanmadı (regresyon veya çok sınıflı modelde geçerli değil).*")

            if err:
                conf = err.get("confidence_analysis", {})
                lines.append(
                    f"\n### Hata Analizi\n"
                    f"  - Hatalı tahmin: {err.get('n_errors')}/{err.get('n_total')} ({err.get('error_rate', 0)*100:.2f}%)\n"
                    f"  - Yüksek güven hatası: {conf.get('high_confidence_errors', 0)} adet ({conf.get('high_confidence_errors_pct', 0)}%)"
                )
                feat_diff = err.get("feature_distribution_diff", {})
                if feat_diff:
                    lines.append(
                        "  - Hatalı vakaların doğru vakalardan en çok ayrıştığı özellikler "
                        "(fark büyüklüğü — Cohen's d: 0.2 küçük, 0.5 orta, 0.8 büyük fark):"
                    )
                    for feat, info in list(feat_diff.items())[:3]:
                        lines.append(f"      {humanize_feature(feat, lang)}: fark={info['cohens_d']} ({info['effect_magnitude']})")

            if shap and shap.get("top5_shap_features"):
                lines.append(
                    "\n### Modelin Kararında En Etkili Özellikler\n"
                    "  (her özelliğin tahmine katkı payı — SHAP yöntemiyle hesaplandı)"
                )
                for feat, pct in list(shap.get("top5_shap_features", {}).items())[:5]:
                    lines.append(f"  - {humanize_feature(feat, lang)}: %{pct}")
            else:
                lines.append("\n*SHAP açıklanabilirlik analizi bu çalışmada hesaplanmadı.*")

            if fi_sum:
                lines.append(
                    f"\n### Feature Importance\n"
                    f"  - Top 5 toplam katki: %{fi_sum.get('top5_total_contribution_pct')}\n"
                    f"  - Onemsiz feature: {fi_sum.get('insignificant_features_count', 0)} adet"
                )
                
                # Importance-based leakage uyarisi
                suspects = fi_sum.get("leakage_suspects", [])
                if suspects:
                    lines.append("\n### Leakage Suphesi (Feature Importance)")
                    for s in suspects:
                        severity_emoji = "[YUKSEK]" if s["severity"] == "high" else "[ORTA]"
                        lines.append(f"  {severity_emoji} {s['column']}: {s['reason']}")

            bias = ev.get("bias_analysis", {})
            if bias:
                lines.append("\n### Bias & Fairness Analizi")
                for base_col, cols in bias.items():
                    lines.append(f"  - Grup: {base_col}")
                    for col, metrics in cols.items():
                        lines.append(f"    - {col}: FPR={metrics.get('fpr', 0):.3f}, FNR={metrics.get('fnr', 0):.3f}")
            else:
                lines.append("\n*Adalet/yanlılık analizi bu veri setinde uygulanmadı (demografik etiket yok).*")

            if opt_thresh:
                lines.append(
                    f"\n- En dengeli karar eşiği: {opt_thresh} "
                    f"(varsayılan 0.5 yerine; doğru yakalama ile yanlış alarm arasında en iyi dengeyi kurar)"
                )

            if rel and rel.get('score') is not None:
                lines.append(
                    f"\n### Güvenilirlik\n"
                    f"  - Skor: {rel.get('score')}/100 | Grade: {rel.get('grade')}\n"
                    f"  - Faktörler: {', '.join(rel.get('factors', []))}"
                )
            else:
                lines.append("\n*Güvenilirlik skoru bu çalışmada hesaplanmadı.*")
    
    final_context = "\n".join(lines)
    return final_context


# System Prompts — task-specific (prompts/*.md dosyalarından yüklenir)

SYSTEM_PROMPT            = load_prompt("report_system_classification.md")
SYSTEM_PROMPT_REGRESSION = load_prompt("report_system_regression.md")
SYSTEM_PROMPT_CLUSTERING = load_prompt("report_system_clustering.md")
SYSTEM_PROMPT_ANOMALY    = load_prompt("report_system_anomaly.md")


# Deterministik başlık üretimi (LLM'e bırakılmaz)

# Hedef adı yalnızca bu intent'lerde meta satırına yazılır (clustering/anomaly hedefsiz)
_SUPERVISED_INTENTS = ("classification", "regression")


def _report_title_for(intent: str, lang: str = "tr") -> str:
    """Intent'e göre sabit, generik rapor başlığı. Hedef adı başlıkta GEÇMEZ."""
    if intent == "clustering":
        return R("title_clustering", lang)
    if intent == "anomaly_detection":
        return R("title_anomaly", lang)
    return R("title_model", lang)


def _resolve_target_name(state: AgentState) -> str:
    """
    Hedef sütun adını önceliğe göre çözer: target_col → target_definition.column_name
    → feature_plan.target_column. Ham ad korunur (çeviri/güzelleştirme yok).
    """
    return (
        state.get("target_col")
        or (state.get("target_definition") or {}).get("column_name")
        or (state.get("feature_plan") or {}).get("target_column")
        or ""
    )


def _build_report_header(state: AgentState, lang: str = "tr") -> str:
    """
    Deterministik H1 başlık + meta satırı üretir:
        # <generik başlık>
        > hedef: <ham ad> · görev: <intent> · model: <model> · <tarih>
    Hedef adı yalnızca supervised intent'lerde ve doluysa eklenir.
    """
    intent = state.get("intent", "classification")
    model_name = state.get("model_name", "unknown")
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")

    title = _report_title_for(intent, lang)

    parts: list[str] = []
    if intent in _SUPERVISED_INTENTS:
        target_name = _resolve_target_name(state)
        if target_name:
            parts.append(f"{R('meta_target', lang)}: {target_name}")
    parts.append(f"{R('meta_task', lang)}: {intent}")
    parts.append(f"{R('meta_model', lang)}: {model_name}")
    parts.append(timestamp)

    return f"# {title}\n> {' · '.join(parts)}\n"


# Zaman serisi (timeseries) deterministik özet bölümü — LLM'e bırakılmaz.
# frequency / split_strategy ham değerleri report_i18n'deki dil haritalarıyla çevrilir.


def _build_timeseries_section(state: AgentState, lang: str = "tr") -> str:
    """modality=timeseries ise zaman serisine özel deterministik bir özet bölümü döndürür.

    Yalnızca feature_report.timeseries_lag içinde GERÇEKTEN bulunan değerleri yazar
    (anti-fabrication: hiçbir sayı uydurulmaz). Tabular/boş modality veya ts bilgisi
    yoksa "" döner — bölüm hiç görünmez, boş başlık bırakılmaz (boş bölüm kuralı).
    """
    if (state.get("modality") or "").strip().lower() != "timeseries":
        return ""
    fr = state.get("feature_report") or {}
    ts = fr.get("timeseries_lag") or {}
    if not ts:
        return ""

    lines: list[str] = [
        R("ts_title", lang),
        "",
        R("ts_intro", lang),
    ]

    # (b) tespit edilen frekans
    freq = ts.get("frequency")
    if freq:
        lines.append(R("ts_freq", lang, f=freq_label(freq, lang)))

    # (c) kullanılan split (entity-aware ise ayrıca belirtilir)
    split = fr.get("split_strategy")
    split_txt = split_label(split, lang)
    if split_txt:
        lines.append(R("ts_split", lang, s=split_txt))

    # (d) üretilen feature türleri — lag / rolling pencereleri
    lag_w = ts.get("windows") or []
    if lag_w:
        n_lag = len(ts.get("lag_columns") or [])
        lines.append(
            R("ts_lag", lang, w=", ".join(str(w) for w in lag_w))
            + (R("ts_lag_n", lang, n=n_lag) if n_lag else "")
        )
    roll_w = ts.get("roll_windows") or []
    if roll_w:
        n_roll = len(ts.get("rolling_columns") or [])
        lines.append(
            R("ts_roll", lang, w=", ".join(str(w) for w in roll_w))
            + (R("ts_roll_n", lang, n=n_roll) if n_roll else "")
        )
    dropped_lead = ts.get("dropped_lead_rows")
    if dropped_lead:
        lines.append(R("ts_dropped_lead", lang, n=dropped_lead))

    # (e) multi-entity (panel) bilgisi — yalnızca entity_col doluysa
    entity_col = ts.get("entity_col")
    if entity_col:
        seg = R("ts_entity", lang, c=entity_col)
        dropped_ent = ts.get("dropped_entities")
        if dropped_ent:
            seg += R("ts_dropped_ent", lang, n=dropped_ent)
        lines.append(seg)

    lines.append("")
    return "\n".join(lines) + "\n"


# Main Node
def generate_report(state: AgentState) -> dict:
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
    logger.info("\n[report] Başlıyor...")

    # Rapor çıktı dili (UI'dan gelir; boş → "tr"). Tüm deterministik/verbatim
    # dizeler ve LLM çıktısı bu dile göre üretilir.
    lang = normalize_lang(state.get("report_language"))

    # Check for pipeline errors first
    if state.get("error"):
        error_msg = state.get("error")
        logger.warning(f"[report] UYARI: Pipeline hatası algılandı: {error_msg}")
        return {
            **state,
            "report": (
                f"{R('err_incomplete_title', lang)}\n\n"
                + R("err_incomplete_body", lang, e=error_msg)
            ),
            "current_step": "report_node",
        }

    # control evaluation report
    ev = state.get("evaluation_report", {})
    if not ev:
        logger.warning(f"[report] UYARI: {R('err_noeval', 'tr')}")
        return {
            **state,
            "report": (
                f"{R('err_noreport_title', lang)}\n\n"
                f"{R('err_noeval', lang)}\n\n"
                f"{R('err_noreport_body', lang)}"
            ),
            "current_step": "report_node",
        }

    # build context (verbatim parçalar lang'e lokalize)
    context = _build_context(state, lang)
    logger.info(f"[report] Context hazırlandı ({len(context)} karakter)")

    # select system prompt based on intent
    intent = state.get("intent", "classification")
    if intent == "clustering":
        system_prompt = SYSTEM_PROMPT_CLUSTERING
    elif intent == "anomaly_detection":
        system_prompt = SYSTEM_PROMPT_ANOMALY
    elif intent == "regression":
        system_prompt = SYSTEM_PROMPT_REGRESSION
    else:
        system_prompt = SYSTEM_PROMPT

    # Domain pack (varsa) yorum merceğini system prompt'a ekle
    try:
        pack = get_active_pack(state.get("domain_pack"))
        mc = MerchantContext.from_dict(state.get("merchant_context"))
        analysis = MerchantAnalysis.from_dict(state.get("merchant_analysis"))
        if pack.name != "none" and (mc is not None or analysis is not None):
            lens = pack.interpret_lens(mc, intent, analysis)
            if lens:
                system_prompt = system_prompt + "\n\n" + lens.system_suffix
                src = "agentic" if analysis else "statik"
                logger.info(f"[report] Domain pack '{pack.name}' yorum merceği eklendi ({src}).")
    except Exception as e:  # pragma: no cover - defensive
        logger.info(f"[report] Domain pack yorum merceği başarısız ({e}); atlanıyor.")

    # Çıktı dili direktifi — en yüksek öncelik; .md'lerdeki sabit dil cümlesi kaldırıldı,
    # dil buradan dinamik geliyor. LLM her başlık/etiket/cümleyi bu dilde yazar.
    system_prompt = system_prompt + (
        f"\n\n## OUTPUT LANGUAGE (HIGHEST PRIORITY)\n"
        f"Write the ENTIRE report in {LANGUAGE_NAME[lang]} — every section heading, "
        f"label, bullet and sentence must be in {LANGUAGE_NAME[lang]}. Translate the "
        f"section headings too. Do NOT mix languages. "
        f"(The verbatim status/decision/FAIL/WARN lines from the context are already "
        f"in {LANGUAGE_NAME[lang]}; copy them as-is.)"
    )

    # llm
    llm = _get_llm()
    user_query = state.get("user_query", "")
    model_name = state.get("model_name", "unknown")

    prompt = R("human_intro", lang, q=user_query, ctx=context)

    logger.info("[report] LLM'den rapor üretiliyor...")
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

        # Deterministik başlık + meta satırını en üste ekle (LLM başlık üretmiyor).
        # Zaman serisi özeti yalnızca modality=timeseries iken araya girer (tabular'da "").
        report_text = (
            _build_report_header(state, lang) + "\n"
            + _build_timeseries_section(state, lang)
            + report_text.lstrip()
        )

    except Exception as e:
        logger.error(f"[report] LLM hatası: {e}")
        # create fallback report if llm fails
        report_text = _generate_fallback_report(state, lang)

    logger.info(f"[report] Rapor tamamlandı ({len(report_text)} karakter)")

    return {
        **state,
        "report": report_text,
        "current_step": "report_node",
    }


# fallback report (llm failes)
def _generate_fallback_report(state: AgentState, lang: str = "tr") -> str:
    """LLM çağrısı başarısız olursa state verilerinden doğrudan rapor üretir.

    Tüm yapısal başlık/etiketler ``lang`` diline lokalizedir (report_i18n).
    """
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

    grade = rel.get("grade", "?")
    score = rel.get("score", 0)

    ts_section = _build_timeseries_section(state, lang)
    lines = [
        _build_report_header(state, lang).rstrip("\n"),
        "",
    ]
    if ts_section:
        lines += [ts_section.rstrip("\n"), ""]
    lines += [
        R("fb_h1", lang),
        f"- {R('fb_analysis_type', lang)}: **{intent}**",
        f"- {R('fb_tables', lang)}: {', '.join(tables)}",
        f"- {R('fb_best_model', lang)}: **{model_name}**",
        f"- {R('fb_overall_rel', lang)}: **{score}/100 (Grade: {grade})**",
        "",
        R("fb_h2", lang),
        f"| {R('fb_metric', lang)} | {R('fb_value', lang)} |",
        f"|--------|-------|",
    ]

    for k, v in test_m.items():
        lines.append(f"| {k} | {v} |")

    if ci:
        lines += ["", R("fb_ci", lang)]
        for metric, vals in ci.items():
            lines.append(f"- **{metric}**: {vals['mean']} [{vals['lower']} — {vals['upper']}]")
    else:
        lines.append(f"\n{R('fb_no_ci', lang)}")

    disp_s, test_s, ovf_label = _overfitting_scores(ovf)
    if ovf and disp_s is not None:
        lines += [
            "",
            R("fb_h3", lang),
            f"- {ovf_label}: {disp_s} | Test: {test_s}",
            f"- {R('fb_gap', lang)} ({ovf_label}−Test): {ovf.get('gap')} | {R('fb_status', lang)}: **{ovf.get('status')}**",
            f"- {ovf.get('description', '')}",
        ]
        _tr = ovf.get("train_score", ovf.get("train_r2"))
        if ovf_label == "CV (5-fold)" and _tr is not None:
            lines.append(f"- {R('fb_resub_note', lang, tr=_tr)}")
    else:
        lines.append(f"\n{R('fb_no_ovf', lang)}")

    if cal and isinstance(cal, dict) and cal.get('brier_score') is not None:
        lines += [
            "",
            R("fb_h4", lang),
            f"- {R('fb_brier', lang)}: {cal.get('brier_score')}",
            f"- {R('fb_well_cal', lang)}: {'✅' if cal.get('is_well_calibrated') else '❌'}",
        ]
    else:
        lines.append(f"\n{R('fb_no_cal', lang)}")

    if shap and shap.get("top5_shap_features"):
        lines += ["", R("fb_h5", lang)]
        for feat, pct in shap.get("top5_shap_features", {}).items():
            lines.append(f"- **{feat}**: %{pct}")
    else:
        lines.append(f"\n{R('fb_no_shap', lang)}")

    if err:
        conf = err.get("confidence_analysis", {})
        hce = conf.get("high_confidence_errors", 0)
        suffix = R("fb_count_suffix", lang)
        lines += [
            "",
            R("fb_h6", lang),
            f"- {R('fb_total_err', lang)}: {err.get('n_errors')}/{err.get('n_total')} ({err.get('error_rate', 0)*100:.2f}%)",
            f"- {R('fb_high_conf_err', lang)}: {hce}" + (f" {suffix}" if suffix else ""),
        ]

    bias = ev.get("bias_analysis", {})
    if bias:
        lines += ["", R("fb_h7", lang)]
        for base_col, cols in bias.items():
            lines.append(f"- **{R('fb_group', lang)}: {base_col}**")
            for col, metrics in cols.items():
                lines.append(f"  - {col}: FPR={metrics.get('fpr', 0):.3f}, FNR={metrics.get('fnr', 0):.3f}")
    else:
        lines.append(f"\n{R('fb_no_bias', lang)}")

    opt_thresh = ev.get("optimal_threshold")
    if opt_thresh:
        lines += [
            "",
            R("fb_h8", lang),
            R("fb_threshold", lang, t=opt_thresh),
        ]

    lines += [
        "",
        R("fb_h9", lang),
        f"- {R('fb_score', lang)}: **{score}/100** | Grade: **{grade}**",
    ]
    for factor in rel.get("factors", []):
        lines.append(f"  - ✓ {factor}")

    return "\n".join(lines)
