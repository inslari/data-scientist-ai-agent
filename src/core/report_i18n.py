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
    "fb_h1": {"tr": "## Yönetici Özeti", "en": "## Executive Summary"},
    "fb_analysis_type": {"tr": "Analiz tipi", "en": "Analysis type"},
    "fb_tables": {"tr": "Kullanılan tablolar", "en": "Tables used"},
    "fb_best_model": {"tr": "En iyi model", "en": "Best model"},
    "fb_overall_rel": {"tr": "Genel güvenilirlik", "en": "Overall reliability"},
    # NOT: fb_h2/fb_metric/fb_value/fb_ci/fb_no_ci KALDIRILDI. Fallback artık
    # performans metriklerini ve güven aralıklarını basmıyor — katmanlı Model
    # Performansı bölümü LLM'den bağımsız çalıştığı için fallback yolunda da
    # montajla eklenir. Buraya "## Model Performansı" başlığını geri koymak
    # raporda İKİ performans bölümü demektir.
    "fb_h3": {"tr": "## Overfitting Analizi", "en": "## Overfitting Analysis"},
    "fb_gap": {"tr": "Fark", "en": "Gap"},
    "fb_status": {"tr": "Durum", "en": "Status"},
    "fb_resub_note": {
        "tr": "Not: Train (resubstitution) {tr} — ezber skoru, genelleme göstergesi değil.",
        "en": "Note: Train (resubstitution) {tr} — a memorization score, not a generalization indicator.",
    },
    "fb_no_ovf": {"tr": "*Overfitting analizi bu çalışmada uygulanmadı.*", "en": "*Overfitting analysis was not performed in this run.*"},
    "fb_h4": {"tr": "## Kalibrasyon", "en": "## Calibration"},
    "fb_brier": {"tr": "Brier Score", "en": "Brier Score"},
    "fb_well_cal": {"tr": "İyi kalibre", "en": "Well-calibrated"},
    "fb_no_cal": {"tr": "*Kalibrasyon analizi bu çalışmada uygulanmadı.*", "en": "*Calibration analysis was not performed in this run.*"},
    # NOT: fb_h5 / fb_no_shap KALDIRILDI. Etkili Özellikler artık katmanlı bölüm
    # (build_features_section); SHAP'ı test_set, iç önemi model_internal etiketiyle
    # kendisi basıyor. "## En Etkili Özellikler (SHAP)" başlığını burada tutmak
    # fallback yolunda İKİNCİ bir özellik bölümüne zemin olurdu.
    "fb_h6": {"tr": "## Hata Analizi", "en": "## Error Analysis"},
    "fb_total_err": {"tr": "Toplam hata", "en": "Total errors"},
    "fb_high_conf_err": {"tr": "Yüksek güven hatası", "en": "High-confidence errors"},
    "fb_count_suffix": {"tr": "adet", "en": ""},
    # fb_h7 / fb_group / fb_no_bias KALDIRILDI: fairness artık katmanlı bölüm olarak
    # (src/services/report_sections.py) üretiliyor ve kendi başlığını/metnini taşıyor.
    # Eski "demografik etiket yok" cümlesi de yanıltıcıydı — atlamanın SEBEBİ artık
    # ölçüm anında belirlenip taşınıyor (tek değerli kolon, ince örneklem, yüksek
    # kardinalite, hizalama hatası...), tek bir sabit cümleye indirgenmiyor.
    "fb_h8": {"tr": "## Operasyonel Öneri", "en": "## Operational Recommendation"},
    "fb_threshold": {
        "tr": "- Varsayılan 0.5 yerine **{t}** threshold'u kullanmak recall'u artırır.",
        "en": "- Using a threshold of **{t}** instead of the default 0.5 improves recall.",
    },
    "fb_h9": {"tr": "## Güvenilirlik", "en": "## Reliability"},
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
    # Kaçış notu: kural doğru (kolon kayması en olası sebep) ama kullanıcı çıkışsız kalmasın —
    # `is_risk_score` gibi MEŞRU sürekli bir hedefte isimlendirme kasıtlıysa eşik env ile açılır.
    # Kod davranışı değişmez; yalnız mesaj yolu gösterir.
    "dq_target_conflict": {
        "tr": "**Hedef değişken `{col}`** ikili (0/1) bir etiket bekleniyordu ama **{nunique} farklı sürekli değer** içeriyor. Bu genellikle kolon hizasının kayması (ör. içinde virgül olan bir alanın satırı bölmesi) sonucu hedef sütuna yanlış verinin düşmesinden kaynaklanır. Model eğitilmedi.\n\n_Hedefiniz gerçekten sürekli bir skorsa (ör. `is_risk_score` = 0–100) ve bu isimlendirme kasıtlıysa, izin verilen benzersiz değer sınırını `DQ_TARGET_MAX_CARD` ortam değişkeniyle yükseltebilirsiniz._",
        "en": "**Target variable `{col}`** was expected to be a binary (0/1) label but contains **{nunique} distinct continuous values**. This usually means the columns are misaligned (e.g. a field with an embedded comma split the row), so the wrong data landed in the target column. No model was trained.\n\n_If your target genuinely is a continuous score (e.g. `is_risk_score` = 0–100) and the naming is intentional, you can raise the allowed distinct-value limit via the `DQ_TARGET_MAX_CARD` environment variable._",
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
    # Sessiz başarısızlık yasağı: olasılık üretilemediyse kullanıcı BOŞ KART değil, NEDEN görsün.
    "eval_no_probabilities": {
        "tr": "Model bu veri için olasılık tahmini (`predict_proba`) üretemedi; bu yüzden ROC-AUC, PR-AUC, kalibrasyon, ROC eğrisi ve eşik (threshold) analizi **hesaplanamadı**. Aşağıdaki performans yorumu yalnızca sınıf tahminlerine dayanan metriklere (accuracy, F1, MCC) dayanmaktadır.",
        "en": "The model could not produce probability estimates (`predict_proba`) for this data, so ROC-AUC, PR-AUC, calibration, the ROC curve and threshold analysis **could not be computed**. The performance discussion below rests only on label-based metrics (accuracy, F1, MCC).",
    },
    "auc_unavailable": {
        "tr": "ROC-AUC hesaplanamadı (olasılık tahmini üretilemedi), bu yüzden ayrım gücü doğrulanamadı. Model accuracy/F1 üzerinden değerlendirildi.",
        "en": "ROC-AUC could not be computed (no probability estimates), so discriminative power is unverified. The model was assessed on accuracy/F1 instead.",
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

    # ---- SERVİS KATMANI SEBEPLERİ (kod → R_REASON) ─────────────────────────
    # Bu dizeler EVALUATION ANINDA üretilir (fairness/segment_carrier/evaluator),
    # rapor dilinden ÖNCE. Servis serbest-metin Türkçe basmasın diye artık dil-nötr
    # bir KOD ({"code","params"}) taşır; rapor katmanı render anında R_REASON ile
    # doğru dile çevirir. "Deterministik string tek yerden" ilkesi (bkz. modül başlığı).
    # ── fairness: grup metriği tanımsızlık nedenleri ──
    "fr_fpr_undefined": {
        "tr": "bu grupta gerçek negatif vaka yok",
        "en": "no actual negative cases in this group",
    },
    "fr_fnr_undefined": {
        "tr": "bu grupta gerçek pozitif vaka yok",
        "en": "no actual positive cases in this group",
    },
    "fr_diff_undefined": {
        "tr": "karşılaştırma için gerekli vaka türü gruplardan birinde yok",
        "en": "one of the groups lacks the case type needed for the comparison",
    },
    "fr_di_undef_zero": {
        "tr": "referans grupta hiç pozitif tahmin yok — oran tanımsız (payda 0)",
        "en": "the reference group has no positive predictions — the ratio is undefined (denominator 0)",
    },
    "fr_di_undef_other": {
        "tr": "seçim oranı hesaplanamadı",
        "en": "selection rate could not be computed",
    },
    "fr_brr_undef_zero": {
        "tr": "referans grupta hiç gerçek pozitif yok — oran tanımsız (payda 0)",
        "en": "the reference group has no actual positives — the ratio is undefined (denominator 0)",
    },
    "fr_brr_undef_other": {
        "tr": "taban oranı hesaplanamadı",
        "en": "base rate could not be computed",
    },
    # ── fairness: analiz/segment/grup atlama nedenleri ──
    "fr_no_carrier": {
        "tr": "segment taşıyıcısı bulunamadı",
        "en": "no segment carrier was found",
    },
    "fr_no_rows": {
        "tr": "test setinde satır yok",
        "en": "no rows in the test set",
    },
    "fr_misaligned": {
        "tr": "segment taşıyıcısı ({s} satır) test seti ({t} satır) ile hizalanmıyor — yanlış gruba metrik atfetmemek için analiz yapılmadı",
        "en": "the segment carrier ({s} rows) does not align with the test set ({t} rows) — analysis was skipped to avoid attributing metrics to the wrong group",
    },
    "fr_multiclass": {
        "tr": "çok sınıflı hedef ({n} sınıf) — seçim oranı ve yanlış alarm/kaçırma oranları ikili hedef için tanımlıdır",
        "en": "multi-class target ({n} classes) — selection rate and false-alarm/miss rates are defined for a binary target",
    },
    "fr_thin_group": {
        "tr": "örneklem çok ince (n={n} < {min}) — bu grup için ölçüm yapılsaydı sayı gürültüden ayırt edilemezdi",
        "en": "sample too thin (n={n} < {min}) — a measurement for this group would be indistinguishable from noise",
    },
    "fr_no_groups": {
        "tr": "hiçbir grup ölçüm için yeterli örnekleme sahip değil",
        "en": "no group has a large enough sample to be measured",
    },
    # ── feature_builder: unsupervised dalda fairness taşıyıcısı tanımsız ──
    "seg_unsupervised": {
        "tr": ("denetimsiz analizde (kümeleme/anomali) ayrı bir test seti ve gerçek "
               "etiket yok — grup başına performans karşılaştırması tanımsız"),
        "en": ("in unsupervised analysis (clustering/anomaly) there is no separate test "
               "set with true labels — group-wise performance comparison is undefined"),
    },
    # ── segment_carrier: aday kolon eleme + kapsam boş nedenleri ──
    "seg_skip_card_unreadable": {
        "tr": "kardinalite okunamadı ({e})",
        "en": "cardinality could not be read ({e})",
    },
    "seg_skip_single_value": {
        "tr": "tek değerli — karşılaştırılacak grup yok",
        "en": "single-valued — no group to compare",
    },
    "seg_skip_high_card": {
        "tr": "çok fazla farklı değer ({n} > {max}) — grup başına örneklem anlamlı bir karşılaştırma için fazla incelirdi",
        "en": "too many distinct values ({n} > {max}) — the per-group sample would be too thin for a meaningful comparison",
    },
    "seg_empty_partition": {
        "tr": "test partisyonu boş",
        "en": "the test partition is empty",
    },
    "seg_no_demo_narrow": {
        "tr": "veri setinde demografik/segment kolonu bulunamadı",
        "en": "no demographic/segment column was found in the dataset",
    },
    "seg_no_demo_broad": {
        "tr": "veri setinde düşük kardinaliteli kategorik kolon bulunamadı",
        "en": "no low-cardinality categorical column was found in the dataset",
    },
    # ── evaluator: hata analizinde ham karşılık yok ──
    "raw_unavailable_derived": {
        "tr": "bu özellik modelleme sırasında türetildi (ham karşılığı yok) — değerler modelin çalıştığı ölçekte",
        "en": "this feature was derived during modeling (no raw equivalent) — values are on the scale the model operates in",
    },
    # ── feature_builder: segment taşıyıcısı diske yazılamadı ──
    "seg_write_failed": {
        "tr": "ham test taşıyıcısı diske yazılamadı",
        "en": "the raw test carrier could not be written to disk",
    },
    # ── relations: ilişki hesaplanamadı nedenleri ──
    "rel_train_empty": {
        "tr": "eğitim partisyonu boş — ilişki hesaplanamadı",
        "en": "the training partition is empty — relationships could not be computed",
    },
    "rel_target_failed": {
        "tr": "hedefle ilişkiler hesaplanamadı ({e})",
        "en": "relationships with the target could not be computed ({e})",
    },
    "rel_no_target": {
        "tr": "hedef sütun yok — yalnızca sütunlar arası ilişkiler hesaplandı",
        "en": "no target column — only column-to-column relationships were computed",
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


def R_REASON(reason, lang: str) -> str:
    """Bir SEBEP taşıyıcısını lokalize metne çevirir. Boş sebep → boş string.

    Servis katmanı (fairness/segment_carrier/evaluator) sebebi rapor dilinden ÖNCE
    üretir; bu yüzden serbest Türkçe metin yerine dil-nötr bir KOD taşır:
        {"code": "fr_multiclass", "params": {"n": 3}}
    Rapor katmanı render anında (dil bilinir) bunu R() ile çevirir. Truthiness
    korunur: boş sebep "" (falsy), dolu sebep dolu dict (truthy) — mevcut
    ``if reason:`` kontrolleri değişmeden çalışır.

    Geriye dönük: düz string gelirse aynen döndürülür (henüz kodlanmamış sebepler
    sessizce bozulmasın; migrasyon tamamlandığında yalnız "" ve dict kalır).
    """
    if not reason:
        return ""
    if isinstance(reason, dict) and "code" in reason:
        return R(reason["code"], lang, **(reason.get("params") or {}))
    return str(reason)


# Metrik sözlüğü (glossary) girdileri — düz dize değil, alanlı sözlük (label/description/
# range/caveat). Mevcut `metric_descriptions` sözlüğü evaluator içinde TR-only hard-code;
# yeni girdiler buradan iki dilli gelir. (Eski girdilerin tamamını buraya taşımak ayrı bir
# temizlik işi — bilinçli olarak kapsam dışı.)
_METRIC_DESCRIPTIONS: dict[str, dict[str, dict]] = {
    "pr_auc": {
        "tr": {
            "label": "PR-AUC (Precision-Recall AUC)",
            "description": (
                "Precision-Recall eğrisinin altında kalan alan. Dengesiz veride ASIL "
                "performans metriğidir: 'yakaladığım pozitiflerin ne kadarı gerçekten "
                "pozitif' sorusuna odaklanır ve devasa negatif kütleden etkilenmez."
            ),
            "range": "Taban = pozitif oranı (rastgele model), 1 = mükemmel",
            "caveat": (
                "ROC-AUC'un aksine rastgele tabanı 0.5 DEĞİLDİR; taban veri setinin "
                "pozitif oranıdır. Mutlaka `pr_auc_baseline` ile birlikte okunmalıdır."
            ),
        },
        "en": {
            "label": "PR-AUC (Precision-Recall AUC)",
            "description": (
                "Area under the precision-recall curve. The metric that actually matters "
                "on imbalanced data: it asks 'of the positives I flagged, how many are "
                "real', and is not flattered by a huge negative mass."
            ),
            "range": "Baseline = positive rate (random model), 1 = perfect",
            "caveat": (
                "Unlike ROC-AUC, the random baseline is NOT 0.5 — it is the dataset's "
                "positive rate. Always read it together with `pr_auc_baseline`."
            ),
        },
    },
    "pr_auc_baseline": {
        "tr": {
            "label": "PR-AUC Tabanı (Rastgele Model)",
            "description": (
                "Test setindeki pozitif oranı — rastgele tahmin eden bir modelin alacağı "
                "PR-AUC. PR-AUC'u yorumlamak için referans noktası."
            ),
            "range": "0–1 (veri setinin pozitif oranı)",
            "caveat": (
                "Örnek: %0.17 fraud oranında rastgele model 0.0017 alır; 0.71'lik bir "
                "PR-AUC vasat değil, tabanın ~400 katıdır."
            ),
        },
        "en": {
            "label": "PR-AUC Baseline (Random Model)",
            "description": (
                "The positive rate in the test set — the PR-AUC a random model would get. "
                "The reference point for interpreting PR-AUC."
            ),
            "range": "0–1 (the dataset's positive rate)",
            "caveat": (
                "Example: at a 0.17% fraud rate a random model scores 0.0017; a PR-AUC of "
                "0.71 is not mediocre, it is ~400x the baseline."
            ),
        },
    },
}


def R_METRIC(key: str, lang: str) -> dict:
    """Metrik sözlüğü girdisini (label/description/range/caveat) dile göre döndürür."""
    entry = _METRIC_DESCRIPTIONS.get(key)
    if entry is None:
        return {"label": key, "description": ""}
    return entry.get(lang) or entry.get("tr") or {"label": key, "description": ""}


def freq_label(freq: str | None, lang: str) -> str:
    if not freq:
        return ""
    return FREQ.get(lang, FREQ["tr"]).get(freq, freq)


def split_label(split: str | None, lang: str) -> str | None:
    if not split:
        return None
    return SPLIT.get(lang, SPLIT["tr"]).get(split)
