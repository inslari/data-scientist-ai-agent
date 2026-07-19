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
from src.core.report_schema import (
    ORDER_HEADER,
    ORDER_LEGACY_PROSE,
    ORDER_STATUS,
    ORDER_TIMESERIES,
    ReportSection,
    render_report,
    sections_to_dicts,
)
from src.services.report_sections import (
    build_assessment_section,
    build_data_quality_section,
    build_error_analysis_section,
    build_fairness_section,
    build_features_section,
    build_generalization_section,
    build_leakage_section,
    build_performance_section,
    build_recommendations_section,
    build_relations_section,
    build_section_prose,
    build_summary_section,
    build_synthesis_prose,
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


def _build_status_section(state: AgentState, lang: str = "tr") -> str:
    """quality_gate durum rozeti + Karar + FAIL/WARN — DETERMİNİSTİK render.

    Eskiden bu blok context'e konup LLM'den "VERBATIM kopyala" diye istenirdi.
    İki şekilde birden kırılıyordu:
      1. LLM satırın önüne boş bir '## ' başlığı koyuyordu (talimat "H1 yapma"
         diyordu, o da H2 yapıyordu) → her raporda görünür artık.
      2. quality_gate HİÇ YOKKEN rozeti UYDURUYORDU — ölçüldü: "✅ Durum: Model
         güvenilir" yazdı; hem uydurma hem prompt'un açıkça yasakladığı
         ("güvenilir" deme) iddia.
    Talimatı sertleştirmek işe yaramadı (denendi). Rozet zaten deterministik bir
    veriden geliyor → LLM'e hiç sorulmaz, kod basar. Fairness'taki ders: kuralı
    yasak olarak değil, YAPI olarak kur.

    quality_gate yoksa "" döner → bölüm hiç görünmez (uydurulamaz da).
    """
    qg = state.get("quality_gate") or {}
    if not qg:
        return ""

    lines: list[str] = []
    badge = {
        "pass": R("badge_pass", lang),
        "warn": R("badge_warn", lang),
        "fail": R("badge_fail", lang),
    }.get(qg.get("status"), "")

    if badge:
        lines.append(f"**{badge}**")
        lines.append("")
    summary = qg.get("summary", "")
    if summary:
        lines.append(f"**{R('decision', lang)}:** {summary}")
        lines.append("")

    for key, header in (("failures", "fail_header"), ("warnings", "warn_header")):
        items = qg.get(key) or []
        if items:
            lines.append(f"## {R(header, lang)}")
            for it in items:
                lines.append(f"- {it['message']}")
            lines.append("")

    return "\n".join(lines)


def _build_context(state: AgentState, lang: str = "tr") -> str:
    """Formats all state outputs into a structured text block for the LLM.

    Bu blok tamamen LLM GİRDİSİDİR — hiçbir parçası çıktıya birebir kopyalanmaz.
    Durum rozeti/Karar/FAIL/WARN artık _build_status_section ile deterministik
    basılıyor; buraya yalnızca LLM özetinde bağlam olarak kullansın diye girer.
    """
    lines: list[str] = []
    intent = state.get("intent", "unknown")

    # quality gate — YALNIZCA bağlam. Çıktıya kopyalanması İSTENMİYOR (prompt bunu
    # açıkça yasaklar); rozetin kendisi montajda deterministik basılıyor.
    qg = state.get("quality_gate") or {}
    if qg:
        lines.append(
            f"## Kalite Kapısı (BAĞLAM — bu satırları rapora KOPYALAMA, sistem basıyor)\n"
            f"- Durum: {qg.get('status', '?')}\n"
            f"- Karar: {qg.get('summary', '')}"
        )
        for key, label in (("failures", "Kritik sorun"), ("warnings", "Uyarı")):
            for it in qg.get(key) or []:
                lines.append(f"  - [{label}] {it['message']}")
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

    # domain leakage ARTIK bu context'te DEĞİL: katmanlı Sızıntı & Riskler bölümü
    # (build_leakage_section) üç dedektörü — domain / target_lock / importance —
    # kalem bazında ayrıştırıp otomatik-drop bilgisiyle birlikte sunuyor. Burada
    # ayrıca basmak aynı bulguyu iki farklı çerçeveyle tekrarlardı. LLM'e leakage
    # yazMAMASI söylenir (prompt: bölüm sistem tarafından eklenir).

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
            # all_model_scores ARTIK burada değil: katmanlı Model Performansı
            # bölümünün "Denenen diğer modeller" alt bölümünde, her skorun neye
            # göre (CV) ölçüldüğü etiketli hâlde veriliyor.
            lines.append("")

        if ev:
            # NOT: confidence_intervals, error_analysis, shap_importance ve
            # feature_importance_summary BİLEREK okunmuyor — hepsi artık katmanlı
            # bölümlerin (performans / hata / etkili özellikler / sızıntı) kanıt
            # katmanında, kendi provenance etiketleriyle.
            test_m = ev.get("test_metrics", {})
            ovf = ev.get("overfitting", {})
            cal = ev.get("calibration", {})
            rel = ev.get("reliability_score", {})
            opt_thresh = ev.get("optimal_threshold")

            # Test metrikleri BAĞLAM olarak kalır (Özet "model ne kadar iyi"
            # sorusunu, Zayıf Yönler ise CV−Test farkını bunlar olmadan yanıtlayamaz)
            # ama PERFORMANS BÖLÜMÜ artık LLM'in değil: katmanlı bölüm olarak
            # (build_performance_section) kanıt/aralık/confusion ile birlikte
            # üretiliyor. Status rozeti deseni: sayıyı göster, yazma işini alma.
            lines.append(
                "## Evaluation Sonuçları\n### Test Metrikleri\n"
                "  (BAĞLAM — bu satırları rapora KOPYALAMA ve bir 'Model Performansı'\n"
                "   bölümü YAZMA; sistem o bölümü kanıtlarıyla birlikte kendisi ekliyor.\n"
                "   Burada yalnızca Özet'in ve genelleme değerlendirmesinin tutarlı\n"
                "   olabilmesi için gösteriliyor.)"
            )
            for k, v in test_m.items():
                lines.append(f"  - {k}: {v}")

            # Güven aralıkları ARTIK bu context'te değil: katmanlı Model Performansı
            # bölümünde her metriğin YANINDA basılıyor. Burada bırakmak aynı aralığın
            # iki yerde, biri kanıtsız iki kez anlatılmasına yol açardı.

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

            # Hata analizi ARTIK bu context'te değil: katmanlı bölüm olarak ayrı
            # üretiliyor (build_error_analysis_section). Burada bırakmak aynı bulgunun
            # iki yerde iki farklı çerçeveyle anlatılmasına yol açardı — üstelik
            # buradaki ölçekli değerler ("fark=0.37") kanıt katmanının ham değerleriyle
            # ("hatalı vakalarda 42 saat") çelişirdi.

            # Etkili Özellikler (SHAP + feature_importance) ve Sızıntı & Riskler
            # (leakage_suspects + target_lock + domain) ARTIK bu context'te DEĞİL:
            # katmanlı bölümler olarak ayrı üretiliyor (build_features_section /
            # build_leakage_section). SHAP'ı test_set, importance'ı model_internal
            # provenance'ıyla AYRI sunuyorlar; üç leakage dedektörü kalem bazında
            # ayrıştırılıyor. Burada bırakmak ikisini de iki farklı yerde iki farklı
            # çerçeveyle anlatırdı.

            # Fairness ARTIK bu context'te değil: katmanlı bölüm olarak ayrı üretilip
            # (build_fairness_section) rapora enjekte ediliyor. Buraya bir özet koymak
            # aynı bulgunun iki farklı yerde iki farklı çerçeveyle anlatılmasına yol
            # açardı. LLM'e fairness'ı YAZMAMASI söylenir (prompt: bölüm sistem
            # tarafından eklenir).

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

    # Veri Kalitesi (Katman 2) hard-fail → aksiyona dönük özel rapor.
    # Generic 'error'/'0 satır' yolundan ÖNCE: kullanıcı NE bozuldu / KAÇ satır
    # atlandı görsün (TR/EN). Bkz. src/services/data_quality.py.
    if state.get("data_quality_status") == "failed":
        dqr = state.get("data_quality_report", {}) or {}
        reason = (dqr.get("decision", {}) or {}).get("reason", "")
        q = dqr.get("quarantine", {}) or {}
        th = dqr.get("target_health", {}) or {}
        struct = dqr.get("structural", {}) or {}
        _tables = struct.get("tables", []) or []
        cols = _tables[0].get("expected_cols", "?") if _tables else struct.get("expected_cols", "?")

        parts = [R("dq_failed_title", lang), "", R("dq_failed_intro", lang), ""]
        if reason == "target_role_content_conflict":
            parts.append(R("dq_target_conflict", lang,
                           col=th.get("target", "?"), nunique=th.get("nunique", "?")))
        elif reason == "structural_mismatch_high":
            parts.append(R("dq_structural_high", lang,
                           n=q.get("rows", 0), total=q.get("total", 0),
                           rate=f"%{q.get('mismatch_rate', 0) * 100:.1f}", cols=cols))
        elif reason == "structural_mismatch_target":
            parts.append(R("dq_structural", lang,
                           n=q.get("rows", 0), total=q.get("total", 0), cols=cols))
        parts.append(R("dq_action_hint", lang))
        logger.warning(f"[report] Veri kalitesi hard-fail: reason='{reason}'")
        return {
            **state,
            "report": "\n".join(parts),
            "current_step": "report_node",
        }

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

    # ── Kanıt-dayalı katmanlı bölümler (her biri kendi verisinden) ──────────
    # Bunlar önce kurulur; sentez bölümleri (Özet/Güçlü-Zayıf/Öneriler) bunların
    # DIGEST'ini girdi alır — üstte olmayan bir sayı taşıyamasınlar diye.
    evidence_sections = [
        build_section_prose(build_data_quality_section(state, lang), lang),
        build_section_prose(build_relations_section(state, lang), lang),
        build_section_prose(build_performance_section(state, lang), lang),
        build_section_prose(build_generalization_section(state, lang), lang),
        build_section_prose(build_error_analysis_section(state, lang), lang),
        build_section_prose(build_features_section(state, lang), lang),
        build_section_prose(build_leakage_section(state, lang), lang),
        build_section_prose(build_fairness_section(state, lang), lang),
    ]

    is_supervised = intent in _SUPERVISED_INTENTS

    # ── Sentez bölümleri (supervised) VEYA blob (clustering/anomaly) ─────────
    # Supervised'da tüm sentez bölümleri katmanlı yapıya taşındı → blob'a gerek yok
    # (legacy_prose = ""). Clustering/anomaly'de o bölümler henüz taşınmadığı için
    # blob hâlâ birincil anlatı: LLM çağrısı orada kalır.
    synthesis_sections: list[ReportSection] = []
    legacy_prose = ""
    if is_supervised:
        synthesis_sections = [
            build_synthesis_prose(build_summary_section(state), evidence_sections, state, lang),
            build_synthesis_prose(build_assessment_section(state), evidence_sections, state, lang),
            build_synthesis_prose(build_recommendations_section(state), evidence_sections, state, lang),
        ]
    else:
        logger.info("[report] LLM'den rapor üretiliyor (blob — clustering/anomaly)...")
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ])
            prose = response.content
            if isinstance(prose, list):
                prose = "".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in prose])
            elif not isinstance(prose, str):
                prose = str(prose)
            legacy_prose = prose.lstrip()
        except Exception as e:
            logger.error(f"[report] LLM hatası: {e}")
            legacy_prose = _generate_fallback_report(state, lang)

    # ── Montaj ───────────────────────────────────────────────────────────────
    # Rapor ORDER'a göre dizilen bölümlerden kurulur (bkz. report_schema.ORDER_*).
    # legacy_prose YALNIZCA clustering/anomaly'de doludur; supervised'da boş string
    # (geçerli ham taşıyıcı → hiçbir şey basmaz). Blob supervised için tamamen boşaldı.
    layered_sections = [*evidence_sections, *synthesis_sections]
    assembly = [
        ReportSection(id="header", title="", order=ORDER_HEADER,
                      raw_markdown=_build_report_header(state, lang)),
        ReportSection(id="timeseries", title="", order=ORDER_TIMESERIES,
                      raw_markdown=_build_timeseries_section(state, lang)),
        ReportSection(id="status", title="", order=ORDER_STATUS,
                      raw_markdown=_build_status_section(state, lang)),
        ReportSection(id="legacy_prose", title="", order=ORDER_LEGACY_PROSE,
                      raw_markdown=legacy_prose),
        *layered_sections,
    ]
    report_text = render_report(assembly, lang)

    # NOT: 'degraded' karantina uyarısı ve örnekleme notu ARTIK burada append
    # EDİLMİYOR — ikisi de katmanlı Veri Kalitesi bölümünün (build_data_quality_section)
    # kanıt/disclaimer katmanında. Burada tekrar basmak çift gösterirdi.

    # Büyük-N model guard (roster kısıldı) → görünür not (sessiz değil).
    _guard = state.get("heavy_model_guard", {}) or {}
    if _guard.get("applied"):
        _parts = []
        if _guard.get("dropped"):
            _parts.append(R("heavy_guard_dropped", lang, models=", ".join(_guard["dropped"])))
        if _guard.get("reduced"):
            _changes = "; ".join(f"{k} {v}" for k, v in _guard["reduced"].items())
            _parts.append(R("heavy_guard_reduced", lang,
                            models=", ".join(_guard["reduced"].keys()), changes=_changes))
        if _parts:
            _detail = (" ve " if lang == "tr" else " and ").join(_parts)
            report_text += R("heavy_model_guard_note", lang, n=_guard.get("n_train", 0),
                             thr=_guard.get("threshold", 0), detail=_detail)

    logger.info(f"[report] Rapor tamamlandı ({len(report_text)} karakter)")

    return {
        **state,
        "report": report_text,
        "report_sections": sections_to_dicts(layered_sections),
        "current_step": "report_node",
    }


# fallback report (llm failes)
def _generate_fallback_report(state: AgentState, lang: str = "tr") -> str:
    """LLM çağrısı başarısız olursa state verilerinden rapor GÖVDESİNİ üretir.

    Yalnızca gövde döner: H1 başlık, zaman serisi özeti ve katmanlı bölümler
    (fairness vb.) MONTAJ tarafından eklenir (bkz. generate_report). Bu fonksiyon
    blob'un yerini aldığı için onunla aynı kapsamda kalmalı — yoksa fallback
    yolunda başlık/bölüm İKİ KEZ basılır.

    Tüm yapısal başlık/etiketler ``lang`` diline lokalizedir (report_i18n).
    """
    # NOT: test_metrics / confidence_intervals / error_analysis / shap_importance
    # BİLEREK okunmuyor — hepsi katmanlı bölümlerin kanıt katmanında ve montajla
    # eklenir (LLM'den bağımsız çalıştıkları için fallback yolunda da basılırlar).
    ev = state.get("evaluation_report", {})
    ovf = ev.get("overfitting", {})
    rel = ev.get("reliability_score", {})
    cal = ev.get("calibration", {})
    model_name = state.get("model_name", "unknown")
    intent = state.get("intent", "unknown")
    tables = state.get("relevant_tables", [])

    grade = rel.get("grade", "?")
    score = rel.get("score", 0)

    # Performans metrikleri / güven aralıkları / eksik-metrik uyarısı ARTIK burada
    # DEĞİL: katmanlı Model Performansı bölümü LLM'den bağımsız çalışır (kanıtı kod
    # kurar), dolayısıyla fallback yolunda da montaj tarafından eklenir. Burada
    # bırakmak aynı metrikleri iki kez, biri kanıtsız bastırırdı.
    lines = [
        R("fb_h1", lang),
        f"- {R('fb_analysis_type', lang)}: **{intent}**",
        f"- {R('fb_tables', lang)}: {', '.join(tables)}",
        f"- {R('fb_best_model', lang)}: **{model_name}**",
        f"- {R('fb_overall_rel', lang)}: **{score}/100 (Grade: {grade})**",
    ]

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

    # Etkili Özellikler (SHAP) ve Sızıntı & Riskler burada YOK: katmanlı bölüm
    # olarak montajda ekleniyorlar. İkisi de LLM'den bağımsız çalışır (kanıtı kod
    # kurar), yani fallback yolunda da basılırlar — burada bırakmak çift olurdu.

    # Hata analizi / Fairness / Veri İlişkileri de aynı şekilde burada YOK:
    # katmanlı bölüm olarak montajda ekleniyorlar (LLM çökse de kanıt katmanı
    # prose'suz render edilir). Buraya kopya koymak bölümleri iki kez basardı.

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
