from __future__ import annotations

"""
Report i18n
-----------
Rapor ÇIKTISINA giren tüm deterministik / verbatim dizeleri tek yerden yönetir.
``report_generator`` ve ``quality_gate`` ortak kullanır.

Kapsam: yalnızca çıktıya birebir/deterministik giren metinler buradadır —
LLM'in yeniden yazdığı ``_build_context`` genel etiketleri (## Veri Profili,
"Boyut:" vb.) burada YOKTUR; onlar LLM'e girdidir, çıktıya geçmez.

Kullanım:
    from src.core.report_i18n import normalize_lang, LANGUAGE_NAME, R
    lang = normalize_lang(state.get("report_language"))
    R("badge_pass", lang)
    R("low_auc", lang, a=0.55)                     # .format ile parametre
"""

SUPPORTED = ("tr", "en")

LANGUAGE_NAME = {"tr": "Turkish", "en": "English"}


def normalize_lang(lang: str | None) -> str:
    """UI'dan gelen dili tr|en'e indirger. Varsayılan: tr (mevcut müşteriler)."""
    l = (lang or "").strip().lower()
    return "en" if l.startswith("en") else "tr"


# ── Frekans / split haritaları (timeseries) ──────────────────────────────────
FREQ = {
    "tr": {"daily": "günlük", "weekly": "haftalık", "monthly": "aylık"},
    "en": {"daily": "daily", "weekly": "weekly", "monthly": "monthly"},
}

SPLIT = {
    "tr": {
        "global_chronological": "kronolojik holdout — geçmişle eğitildi, gelecekte test edildi",
        "entity_aware": "entity-bazlı kronolojik holdout — her entity kendi içinde geçmişle eğitilip gelecekte test edildi",
        "random": "rastgele bölme",
        "none": "ayrı bir bölme uygulanmadı",
    },
    "en": {
        "global_chronological": "chronological holdout — trained on the past, tested on the future",
        "entity_aware": "entity-aware chronological holdout — each entity trained on its own past and tested on its future",
        "random": "random split",
        "none": "no separate split was applied",
    },
}


# ── Tüm deterministik/verbatim dizeler ───────────────────────────────────────
# {param} alanları .format(**params) ile doldurulur.
_STRINGS: dict[str, dict[str, str]] = {
    # ---- report_generator: başlık & meta ----
    "title_model": {"tr": "Model Değerlendirme Raporu", "en": "Model Evaluation Report"},
    "title_clustering": {"tr": "Kümeleme Analizi Raporu", "en": "Clustering Analysis Report"},
    "title_anomaly": {"tr": "Anomali Tespiti Raporu", "en": "Anomaly Detection Report"},
    "meta_target": {"tr": "hedef", "en": "target"},
    "meta_task": {"tr": "görev", "en": "task"},
    "meta_model": {"tr": "model", "en": "model"},

    # ---- report_generator: human prompt (LLM'e talimat; çıktıda değil ama dile uysun) ----
    "human_intro": {
        "tr": (
            "Kullanıcı sorusu: '{q}'\n\n"
            "Aşağıda pipeline'ın tüm çıktıları bulunmaktadır. "
            "Bu verileri kullanarak kapsamlı bir analiz raporu yaz.\n\n{ctx}"
        ),
        "en": (
            "User question: '{q}'\n\n"
            "Below are all outputs of the pipeline. "
            "Using this data, write a comprehensive analysis report.\n\n{ctx}"
        ),
    },

    # ---- report_generator: status & karar (verbatim çıktı) ----
    "badge_pass": {"tr": "✅ Durum: güçlü performans", "en": "✅ Status: strong performance"},
    "badge_warn": {"tr": "⚠️ Durum: dikkat gerektiren noktalar var", "en": "⚠️ Status: points that require attention"},
    "badge_fail": {"tr": "🛑 Durum: yetersiz, gözden geçirme gerekli", "en": "🛑 Status: insufficient, review required"},
    "decision": {"tr": "Karar", "en": "Decision"},
    "fail_header": {"tr": "Kritik Sorunlar (FAIL)", "en": "Critical Issues (FAIL)"},
    "warn_header": {"tr": "Uyarılar (WARN)", "en": "Warnings (WARN)"},

    # ---- report_generator: timeseries bölümü (verbatim çıktı) ----
    "ts_title": {"tr": "## Zaman Serisi Analizi", "en": "## Time Series Analysis"},
    "ts_intro": {
        "tr": "- Bu bir **zaman serisi** analizidir; satırlar zaman eksenine göre sıralanıp işlenmiştir.",
        "en": "- This is a **time series** analysis; rows were sorted and processed along the time axis.",
    },
    "ts_freq": {"tr": "- Tespit edilen frekans: **{f}**", "en": "- Detected frequency: **{f}**"},
    "ts_split": {"tr": "- Eğitim/test ayrımı: **{s}**", "en": "- Train/test split: **{s}**"},
    "ts_lag": {"tr": "- Lag (gecikme) pencereleri: {w}", "en": "- Lag windows: {w}"},
    "ts_lag_n": {"tr": " ({n} lag özelliği)", "en": " ({n} lag features)"},
    "ts_roll": {"tr": "- Rolling (hareketli ortalama/std) pencereleri: {w}", "en": "- Rolling (moving avg/std) windows: {w}"},
    "ts_roll_n": {"tr": " ({n} rolling özelliği)", "en": " ({n} rolling features)"},
    "ts_dropped_lead": {
        "tr": "- Geçmişi yetersiz ilk {n} satır eğitim öncesi düşürüldü.",
        "en": "- The first {n} rows with insufficient history were dropped before training.",
    },
    "ts_entity": {
        "tr": "- Çok-entity (panel) veri: **{c}** sütununa göre gruplandı; lag/rolling/frekans her entity'nin kendi içinde hesaplandı",
        "en": "- Multi-entity (panel) data: grouped by **{c}**; lag/rolling/frequency were computed within each entity",
    },
    "ts_dropped_ent": {
        "tr": ". Yetersiz geçmişe sahip {n} küçük entity düşürüldü",
        "en": ". {n} small entities with insufficient history were dropped",
    },

    # ---- report_generator: fallback rapor (LLM hata verirse — verbatim çıktı) ----
    "fb_h1": {"tr": "## 1. Yönetici Özeti", "en": "## 1. Executive Summary"},
    "fb_analysis_type": {"tr": "Analiz tipi", "en": "Analysis type"},
    "fb_tables": {"tr": "Kullanılan tablolar", "en": "Tables used"},
    "fb_best_model": {"tr": "En iyi model", "en": "Best model"},
    "fb_overall_rel": {"tr": "Genel güvenilirlik", "en": "Overall reliability"},
    "fb_h2": {"tr": "## 2. Model Performansı", "en": "## 2. Model Performance"},
    "fb_metric": {"tr": "Metrik", "en": "Metric"},
    "fb_value": {"tr": "Değer", "en": "Value"},
    "fb_ci": {"tr": "### Bootstrap Güven Aralıkları (%95)", "en": "### Bootstrap Confidence Intervals (95%)"},
    "fb_no_ci": {"tr": "*Bootstrap güven aralığı bu çalışmada hesaplanmadı.*", "en": "*Bootstrap confidence intervals were not computed in this run.*"},
    "fb_h3": {"tr": "## 3. Overfitting Analizi", "en": "## 3. Overfitting Analysis"},
    "fb_gap": {"tr": "Fark", "en": "Gap"},
    "fb_status": {"tr": "Durum", "en": "Status"},
    "fb_resub_note": {
        "tr": "Not: Train (resubstitution) {tr} — ezber skoru, genelleme göstergesi değil.",
        "en": "Note: Train (resubstitution) {tr} — a memorization score, not a generalization indicator.",
    },
    "fb_no_ovf": {"tr": "*Overfitting analizi bu çalışmada uygulanmadı.*", "en": "*Overfitting analysis was not performed in this run.*"},
    "fb_h4": {"tr": "## 4. Kalibrasyon", "en": "## 4. Calibration"},
    "fb_brier": {"tr": "Brier Score", "en": "Brier Score"},
    "fb_well_cal": {"tr": "İyi kalibre", "en": "Well-calibrated"},
    "fb_no_cal": {"tr": "*Kalibrasyon analizi bu çalışmada uygulanmadı.*", "en": "*Calibration analysis was not performed in this run.*"},
    "fb_h5": {"tr": "## 5. En Etkili Özellikler (SHAP)", "en": "## 5. Most Influential Features (SHAP)"},
    "fb_no_shap": {"tr": "*SHAP açıklanabilirlik analizi bu çalışmada hesaplanmadı.*", "en": "*SHAP explainability analysis was not computed in this run.*"},
    "fb_h6": {"tr": "## 6. Hata Analizi", "en": "## 6. Error Analysis"},
    "fb_total_err": {"tr": "Toplam hata", "en": "Total errors"},
    "fb_high_conf_err": {"tr": "Yüksek güven hatası", "en": "High-confidence errors"},
    "fb_count_suffix": {"tr": "adet", "en": ""},
    "fb_h7": {"tr": "## 7. Bias & Fairness Analizi", "en": "## 7. Bias & Fairness Analysis"},
    "fb_group": {"tr": "Grup", "en": "Group"},
    "fb_no_bias": {
        "tr": "*Adalet/yanlılık analizi bu veri setinde uygulanmadı (demografik etiket yok).*",
        "en": "*Fairness/bias analysis was not performed on this dataset (no demographic labels).*",
    },
    "fb_h8": {"tr": "## 8. Operasyonel Öneri", "en": "## 8. Operational Recommendation"},
    "fb_threshold": {
        "tr": "- Varsayılan 0.5 yerine **{t}** threshold'u kullanmak recall'u artırır.",
        "en": "- Using a threshold of **{t}** instead of the default 0.5 improves recall.",
    },
    "fb_h9": {"tr": "## 9. Güvenilirlik", "en": "## 9. Reliability"},
    "fb_score": {"tr": "Skor", "en": "Score"},

    # ---- report_generator: pipeline hata raporları ----
    "err_incomplete_title": {"tr": "# ❌ Analiz Tamamlanamadı", "en": "# ❌ Analysis Could Not Be Completed"},
    "err_incomplete_body": {
        "tr": "Veri işleme sürecinde kritik bir hata meydana geldi ve analiz durduruldu:\n\n**Hata Nedeni:** {e}\n\n*Lütfen daha büyük veya daha tutarlı bir veri seti ile tekrar deneyin.*",
        "en": "A critical error occurred during data processing and the analysis was halted:\n\n**Error cause:** {e}\n\n*Please try again with a larger or more consistent dataset.*",
    },
    # ---- data_quality (Katman 2): aksiyona dönük hata raporları ----
    # Generic "0 satır" yerine kullanıcı NE bozuldu / KAÇ satır atlandı görsün.
    "dq_failed_title": {
        "tr": "# ❌ Veri Kalitesi Kontrolü: Analiz Durduruldu",
        "en": "# ❌ Data Quality Check: Analysis Halted",
    },
    "dq_failed_intro": {
        "tr": "Modelleme öncesi otomatik veri kalitesi kontrolü verinin güvenle modellenemeyeceğini tespit etti; yanıltıcı bir sonuç üretmemek için analiz durduruldu.",
        "en": "The pre-modeling data quality check found the data cannot be safely modeled; analysis was halted to avoid producing a misleading result.",
    },
    # Target rol-içerik çelişkisi (fraud vakası): ad ikili ima ediyor ama sürekli.
    "dq_target_conflict": {
        "tr": "**Hedef değişken `{col}`** ikili (0/1) bir etiket bekleniyordu ama **{nunique} farklı sürekli değer** içeriyor. Bu genellikle kolon hizasının kayması (ör. içinde virgül olan bir alanın satırı bölmesi) sonucu hedef sütuna yanlış verinin düşmesinden kaynaklanır. Model eğitilmedi.",
        "en": "**Target variable `{col}`** was expected to be a binary (0/1) label but contains **{nunique} distinct continuous values**. This usually means the columns are misaligned (e.g. a field with an embedded comma split the row), so the wrong data landed in the target column. No model was trained.",
    },
    # Yapısal kayma: kaç satır neden atlandı.
    "dq_structural": {
        "tr": "**{n}/{total} satır** hatalı format nedeniyle atlandı: alan sayısı beklenen **{cols} kolon** ile uyuşmuyor (kolon kayması imzası). Kalan satırların sütun hizası — özellikle hedef etiketi — güvenilir sayılamadığı için analiz durduruldu.",
        "en": "**{n}/{total} rows** were skipped due to malformed format: their field count does not match the expected **{cols} columns** (column-shift signature). Because the remaining rows' column alignment — especially the target label — cannot be trusted, analysis was halted.",
    },
    "dq_structural_high": {
        "tr": "**{n}/{total} satır** ({rate}) hatalı format nedeniyle atlandı — beklenen **{cols} kolon** ile alan sayısı uyuşmuyor. Bozulma oranı çok yüksek; veri güvenle modellenemez.",
        "en": "**{n}/{total} rows** ({rate}) were skipped due to malformed format — field count does not match the expected **{cols} columns**. The corruption rate is too high to model the data safely.",
    },
    "dq_action_hint": {
        "tr": "\n\n*Öneri: Kaynak dosyayı kontrol edin — içinde ayraç (virgül) barındıran alanların tırnaklandığından ve her satırın aynı sayıda sütuna sahip olduğundan emin olun, sonra yeniden yükleyin.*",
        "en": "\n\n*Suggestion: Check the source file — ensure fields containing the delimiter (comma) are quoted and every row has the same number of columns, then re-upload.*",
    },
    # Training cap: büyük veri stratified örneklendi (sessiz değil, rapora yansır).
    "sampling_note_stratified": {
        "tr": "\n\n> ℹ️ **Örnekleme:** {orig:,} satırdan {n:,} satıra **stratified** örneklendi; sınıf oranı korundu (azınlık %{before} → %{after}). Sınır: `MAX_TRAINING_ROWS={cap:,}` (env ile artırılabilir).",
        "en": "\n\n> ℹ️ **Sampling:** stratified-sampled {orig:,} → {n:,} rows; class ratio preserved (minority {before}% → {after}%). Limit: `MAX_TRAINING_ROWS={cap:,}` (raise via env).",
    },
    "sampling_note_random": {
        "tr": "\n\n> ℹ️ **Örnekleme:** {orig:,} satırdan {n:,} satıra temsili (random) örneklendi. Sınır: `MAX_TRAINING_ROWS={cap:,}` (env ile artırılabilir).",
        "en": "\n\n> ℹ️ **Sampling:** representatively (random) sampled {orig:,} → {n:,} rows. Limit: `MAX_TRAINING_ROWS={cap:,}` (raise via env).",
    },

    # Büyük-N model guard: roster neden değişti (sessiz değil, rapora yansır).
    "heavy_model_guard_note": {
        "tr": "\n\n> ⚙️ **Model seçimi:** {n:,} eğitim satırı > {thr:,} eşiği — {detail}. (`HEAVY_MODEL_MAX_ROWS` ile ayarlanır.)",
        "en": "\n\n> ⚙️ **Model selection:** {n:,} training rows > {thr:,} threshold — {detail}. (tune via `HEAVY_MODEL_MAX_ROWS`.)",
    },
    "heavy_guard_dropped": {
        "tr": "{models} hız/verim nedeniyle atlandı",
        "en": "{models} skipped for speed/performance",
    },
    "heavy_guard_reduced": {
        "tr": "{models} ağaç sayısı düşürüldü ({changes})",
        "en": "{models} tree count reduced ({changes})",
    },

    # degraded: normal rapora eklenen veri-kalitesi notu (durdurmaz).
    "dq_degraded_note": {
        "tr": "\n\n> ⚠️ **Veri kalitesi notu:** {n} satır hatalı format nedeniyle atlandı; analiz kalan temiz satırlarla yapıldı.",
        "en": "\n\n> ⚠️ **Data quality note:** {n} rows were skipped due to malformed format; analysis used the remaining clean rows.",
    },

    "err_noreport_title": {"tr": "# ❌ Rapor Üretilemedi", "en": "# ❌ Report Could Not Be Generated"},
    "err_noeval": {
        "tr": "Evaluation raporu bulunamadı, report_node atlanıyor.",
        "en": "Evaluation report not found; skipping report_node.",
    },
    "err_noreport_body": {
        "tr": "Bu durum genellikle modelin kurulamamasından kaynaklanır.",
        "en": "This usually results from the model failing to be built.",
    },

    # ---- quality_gate: mesajlar (verbatim çıktı) ----
    "low_auc": {
        "tr": "AUC çok düşük ({a:.3f}). Model rastgele tahminden zar zor iyi.",
        "en": "AUC is very low ({a:.3f}). The model is barely better than random guessing.",
    },
    "low_r2": {
        "tr": "R² çok düşük ({r:.3f}). Model hedefi açıklayamıyor.",
        "en": "R² is very low ({r:.3f}). The model cannot explain the target.",
    },
    "severe_overfitting": {
        "tr": "Ciddi overfitting (train-test farkı: {g:.3f}). Model eğitim verisini ezberlemiş.",
        "en": "Severe overfitting (train-test gap: {g:.3f}). The model memorized the training data.",
    },
    "mild_overfitting": {
        "tr": "Hafif overfitting (fark: {g:.3f}). Modelin genelleme kalitesini kontrol edin.",
        "en": "Mild overfitting (gap: {g:.3f}). Check the model's generalization quality.",
    },
    "domain_leakage_high": {
        "tr": "Yüksek şüpheli leakage: {cols}. Tahminler güvenilir olmayabilir.",
        "en": "High suspected leakage: {cols}. Predictions may be unreliable.",
    },
    "importance_dominance": {
        "tr": "{n} feature anormal yüksek importance'a sahip.",
        "en": "{n} feature(s) have abnormally high importance.",
    },
    "target_lock_high_reg": {
        "tr": (
            "Olası veri sızıntısı: {f}. "
            "Bu özellikler belirli aralıklarda hedefi dar bir bantta neredeyse tam belirliyor "
            "(hedefin varyansı çöküyor) — yani model 'tahmin' etmek yerine cevabı doğrudan "
            "görüyor olabilir. Gerçek kullanımda bu bilgi elde olmayabilir; bu özelliklerin "
            "çıkarılması önerilir."
        ),
        "en": (
            "Possible data leakage: {f}. "
            "In certain ranges these features almost fully determine the target within a narrow band "
            "(its variance collapses) — i.e. the model may be seeing the answer directly instead of "
            "predicting it. This information may not be available in real use; removing these features "
            "is recommended."
        ),
    },
    "target_lock_high_clf": {
        "tr": (
            "Olası veri sızıntısı: {f}. "
            "Bu özellikler belirli aralıklarda sonucu neredeyse kesin belirliyor — "
            "yani model 'tahmin' etmek yerine cevabı doğrudan görüyor olabilir. "
            "Gerçek kullanımda bu bilgi elde olmayabilir; bu özelliklerin çıkarılması önerilir."
        ),
        "en": (
            "Possible data leakage: {f}. "
            "In certain ranges these features almost certainly determine the outcome — "
            "i.e. the model may be seeing the answer directly instead of predicting it. "
            "This information may not be available in real use; removing these features is recommended."
        ),
    },
    "target_lock_low_reg": {
        "tr": (
            "Güçlü belirleyici özellik: {f}. "
            "Hedefle güçlü ama tutarlı bir ilişkisi var (belirli aralıklarda varyansını "
            "kademeli azaltıyor) — bu genelde normaldir, sızıntı değil gerçek bir sinyaldir. "
            "Yine de gerçek kullanımda bu bilginin elde olacağından emin olun."
        ),
        "en": (
            "Strong predictive feature: {f}. "
            "It has a strong but consistent relationship with the target (gradually reducing its "
            "variance over certain ranges) — this is usually normal, a real signal rather than leakage. "
            "Still, make sure this information will be available in real use."
        ),
    },
    "target_lock_low_clf": {
        "tr": (
            "Güçlü belirleyici özellik: {f}. "
            "Sonuçla güçlü ama tutarlı bir ilişkisi var — bu genelde normaldir, "
            "sızıntı değil gerçek bir sinyaldir. Yine de gerçek kullanımda bu bilginin "
            "elde olacağından emin olun."
        ),
        "en": (
            "Strong predictive feature: {f}. "
            "It has a strong but consistent relationship with the outcome — this is usually normal, "
            "a real signal rather than leakage. Still, make sure this information will be available "
            "in real use."
        ),
    },
    "low_reliability": {
        "tr": "Güvenilirlik skoru düşük ({s}/100).",
        "en": "Reliability score is low ({s}/100).",
    },
    "high_mape": {
        "tr": "MAPE yüksek (%{m:.1f}). Tahmin hataları büyük olabilir.",
        "en": "MAPE is high ({m:.1f}%). Prediction errors may be large.",
    },
    "target_zero_variance": {
        "tr": "Seçilen hedef '{c}' tek bir değerden oluşuyor (sıfır varyans) — modellenecek bir değişkenlik yok.",
        "en": "Selected target '{c}' consists of a single value (zero variance) — there is no variability to model.",
    },
    "target_id_like": {
        "tr": "Seçilen hedef '{c}' neredeyse tüm satırlarda benzersiz (%{p:.0f}) — bir kimlik (ID) sütunu olabilir, modelleme anlamlı olmayabilir.",
        "en": "Selected target '{c}' is unique in nearly all rows ({p:.0f}%) — it may be an identifier (ID) column; modeling may be meaningless.",
    },
    "target_high_null": {
        "tr": "Seçilen hedef '{c}' satırların %{p:.0f}'inde boş — yüksek eksik oranı modellemeyi güvenilmez kılabilir.",
        "en": "Selected target '{c}' is null in {p:.0f}% of rows — a high missing rate can make modeling unreliable.",
    },

    # ---- quality_gate: summary (verbatim çıktı) ----
    "summary_fail": {
        "tr": "Model yetersiz — {n} kritik sorun tespit edildi. Sonuçlar bu haliyle güvenilir değil; sorunlar giderilmeden kullanılmamalı.",
        "en": "The model is inadequate — {n} critical issue(s) detected. Results are not reliable as-is and should not be used until the issues are fixed.",
    },
    "summary_warn": {
        "tr": "Model çalışıyor ancak {n} dikkat edilmesi gereken durum var. Sonuçları kullanmadan önce kontrol edin.",
        "en": "The model works, but there are {n} point(s) that require attention. Review the results before using them.",
    },
    "summary_pass": {
        "tr": "Model güçlü performans gösteriyor. Dağıtımdan önce bağımsız doğrulama ve sürekli izleme önerilir.",
        "en": "The model shows strong performance. Independent validation and continuous monitoring are recommended before deployment.",
    },
}


def R(key: str, lang: str, **params) -> str:
    """Anahtar + dil için lokalize dizeyi döndürür; {param} alanlarını .format ile doldurur.

    Bilinmeyen anahtar → anahtarın kendisi (sessiz bozulma yerine görünür iz).
    """
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get("tr") or key
    if params:
        try:
            return text.format(**params)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def freq_label(freq: str | None, lang: str) -> str:
    if not freq:
        return ""
    return FREQ.get(lang, FREQ["tr"]).get(freq, freq)


def split_label(split: str | None, lang: str) -> str | None:
    if not split:
        return None
    return SPLIT.get(lang, SPLIT["tr"]).get(split)
