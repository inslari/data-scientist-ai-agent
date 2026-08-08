from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey
from sqlalchemy.orm import deferred
from api.core.database import Base


class MonitoringSummaryModel(Base):
    """Katman 2 — pencereli monitoring özeti (bir (proje, artifact, gün) grubu = bir satır).

    Aggregator (`src/services/monitoring/aggregator.py`, `run_maintenance` alt-görevi) UTC günlük
    tumbling pencerelerde YALNIZ kapanmış günleri işler; `predictions`'ı `(project_id, artifact_id,
    gün)` ile gruplar. Champion gün ortasında değişmişse öncesi/sonrası farklı `artifact_id` →
    doğal İKİ satır (drift belirli sürüme atfedilir; sürüm karışımı = sahte drift).

    PK = (project_id, artifact_id, window_start) → idempotanslık DB düzeyinde: aynı pencere iki kez
    işlenirse UPSERT (çökme-sonrası kurtarma güvenli). `computed_at` yeniden-yazımda güncellenir;
    dirty-kontrol (`max(coalesce(label_backfilled_at, created_at)) > computed_at`) yalnız veri
    değişen pencereleri yeniden hesaplar (geç `actual_label` backfill'i eski pencereyi kirletir).

    Parça 2 İSKELET: çekirdek alanlar (n_total/n_sampled/status/window_*/computed_at) dolu;
    metrik alanları (psi_*/cbpe_*/quality_*) NULL → Parça 3/4 yalnız alan doldurur, şema DEĞİŞMEZ.

    ``n`` İKİ ANLAM (örnekleme yüzünden ayrışır, farklı metrikler farklısını kullanır):
      * ``n_total``   = gruptaki TÜM prediction satırları → CBPE + veri kalitesi bunu kullanır.
      * ``n_sampled`` = ``input_row`` DOLU satırlar (alan-bazlı örneklem) → PSI bunu kullanır.

    ``status`` ÜÇ değer (Faz 1 #2 dersi "temiz ≠ ölçülmedi" → burada "yetersiz n ≠ referans yok"):
      * ``ok``             — yeterli n, metrik hesaplanabilir.
      * ``insufficient_n`` — pencere min-n altında (gürültü → sayı üretilmez).
      * ``no_reference``   — artifact silinmiş (retention GC) VEYA ``training_reference.per_feature``
        boş (eski/yarım artifact) → PSI referansı yok. İSKELET set ETMEZ; Parça 3 PSI eklerken doldurur.
    """

    __tablename__ = "monitoring_summary"

    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    artifact_id = Column(String, primary_key=True)           # serve eden champion (drift atfı)
    window_start = Column(Integer, primary_key=True)         # UTC gün başı epoch (tumbling)

    window_end = Column(Integer, nullable=False)             # window_start + 86400
    window_kind = Column(String, nullable=False, default="daily")   # ileride "weekly" vb.
    # FLOAT epoch (int değil) → dirty-kontrolün 1-saniye sınır yarışı kapanır: label_backfilled_at da
    # float; run-başında yakalanan computed_at, sonradan commit'lenen etiketten daima küçük → kaçan etiket yok.
    computed_at = Column(Float, nullable=False)              # son hesap epoch (dirty-kontrol çapası)

    n_total = Column(Integer, nullable=False)                # gruptaki ÜRETİM satırları (CBPE/kalite) — ui_manual HARİÇ
    n_sampled = Column(Integer, nullable=False)              # input_row dolu ÜRETİM satırları (PSI) — ui_manual HARİÇ
    status = Column(String, nullable=False)                  # ok | insufficient_n | no_reference
    # KAYNAK AYRIMI (Katman 2): dashboard Scoring playground'dan gelen MANUEL denemeler (predictions.source=
    # "ui_manual") drift/perf'ten DIŞLANIR (n_total onları saymaz) ama SESSİZCE YUTULMAZ → bu sayaç ile
    # first-class görünür (0 olsa da alan hep var). "Bu pencerede N manuel deneme metrikten çıkarıldı" satırı.
    excluded_ui_count = Column(Integer, nullable=False, default=0)

    # ── Parça 3 (PSI, input drift) ──
    psi_overall = Column(Float, nullable=True)          # kolonlar arası ortalama (3b: importance-ağırlıklı)
    psi_max = Column(Float, nullable=True)              # EN KÖTÜ tek kolon — ortalamada erimesin (first-class alarm)
    psi_degenerate_count = Column(Integer, nullable=True)  # ölçülemeyen kolon SAYISI ("temiz ≠ ölçülmedi")
    psi_per_feature = Column(JSON, nullable=True)       # {col: {psi, n_bins, degenerate, reason, importance_weight}}
    # ── Parça 3b (importance-ağırlıklandırma) ──
    importance_attributed_ratio = Column(Float, nullable=True)  # sıralamanın kapsadığı model importance payı (orphan=1−bu)
    degenerate_importance_mass = Column(Float, nullable=True)    # ölçülemeyen kolonların MODEL AĞIRLIĞI (sayı değil kütle)
    # ── Parça 4 (CBPE + veri kalitesi) ──
    # CBPE ⚠️ BİLİNEN SINIR: veri/covariate drift altında çalışır, CONCEPT DRIFT altında ÇALIŞMAZ —
    # concept drift olduğunda tahmin SİSTEMATİK İYİMSER sapar (doğru ölçümün en gerektiği anda yanıltır).
    # cbpe_estimate daima "assumes":"covariate_drift_only" taşır (dashboard "tahmini" damgası bundan);
    # gerçekleşen performans (actual_label backfill) geldiğinde CBPE'nin yerini alır.
    cbpe_estimate = Column(JSON, nullable=True)      # {estimated_roc_auc/accuracy, baseline_*, assumes} ya da NULL (bastırıldı)
    cbpe_gate_status = Column(String, nullable=True) # ok | suppressed_calibration | suppressed_no_baseline | insufficient_n
    quality_rates = Column(JSON, nullable=True)      # {unseen:{count,rate}, out_of_range:{count,rate}, n, alarm_eligible}
    # ── GERÇEKLEŞEN performans (actual_label backfill; CBPE'nin doğrulaması) ──
    # ``label_coverage`` HER ZAMAN dolu (0 dahil, min-n altında dahil) → dashboard "etiket kapsamı %12"
    # satırı realized_metrics NULL iken bile render eder (psi_max/degenerate_count first-class deseni).
    label_coverage = Column(Float, nullable=True)        # labeled_n / n_total
    # cbpe_estimate DESENİ (yan yana dursun: dashboard "tahmini 0.81 / gerçekleşen 0.79"). İçerik:
    # {accuracy, roc_auc(|None), baseline_accuracy, baseline_roc_auc, labeled_n, labeled_positive_rate,
    #  baseline_positive_rate, positive_rate_deviation, assumes} — regresyonda {rmse, mae, ...}.
    # ⚠️ SEÇİLİM YANLILIĞI: etiketler rastgele gelmez (olayı gerçekleşen önce etiketlenir) → düşük
    # kapsamda gerçekleşen metrik YANLI olabilir. assumes="labels_missing_at_random" + positive_rate_deviation
    # DETERMİNİSTİK KANIT taşır (labeled pozitif oran vs baseline sapması); dashboard düşük kapsamda uyarır.
    # (CBPE'nin concept-drift damgasıyla aynı disiplin: sayı saklanır ama sınırı beyan edilir.)
    realized_metrics = Column(JSON, nullable=True)       # gerçekleşen metrik ya da NULL (no_labels/insufficient_labels)
    realized_gate_status = Column(String, nullable=True) # ok | insufficient_labels | no_labels

    # ── TRIAGE (Faz 0 — overall_status roll-up, `src/services/monitoring/triage.py`) ──
    # Dört monitörü DÖRT değere indirger: healthy | warning | critical | **unknown** (ölçülemedi →
    # "temiz ≠ ölçülmedi" rozet karşılığı; çekirdek sinyaller yoksa yeşil ASLA verilmez). Aggregator
    # pencere hesabının SONUNDA yazar (endpoint HESAPLAMAZ, aynen taşır → dashboard yeniden yorumlamaz).
    overall_status = Column(String, nullable=True)       # healthy | warning | critical | unknown
    # Rozeti HANGİ sinyalin belirlediği — {driver, code(makine anahtarı, i18n), value, core_unmeasured,
    # suppressed_axes, measurement_coverage}. `code` prose DEĞİL → dashboard TR/EN'e çevirir. suppressed_axes:
    # ölçülmüş gerçek öncü göstergeyi ezdiyse EZİLEN eksenin ham şiddeti (bilgi gizlenmez, rozete yansımaz).
    overall_status_reason = Column(JSON, nullable=True)

    # ── PSI trend skalerleri (Faz 0 item 3 — list endpoint sparkline'ı) ──
    # {col: psi_float} — `psi_per_feature`'ın HAFİF projeksiyonu (yalnız psi). List endpoint bunu döner
    # (~14.6KB/30 pencere ölçüldü); tam per_feature'ı (88.9KB) YÜKLEMEZ → sparkline ucuz.
    psi_scalars = Column(JSON, nullable=True)            # {col: float}
    # ── PSI dağılım overlay (Faz 0 item 2 — YALNIZ detay endpoint) ──
    # {col: {kind, edges/categories, ref_props, prod_counts, prod_total}} — histogram overlay verisi.
    # ±inf kenarları JSON'da `null` (allow_nan=False + psycopg2 inf reddeder). deferred(): worst-case 39KB
    # (20 kategorik×50 vocab) → list SORGUSU bunu YÜKLEMESİN, yalnız detay `.undefer()` eder (hot-path temiz).
    psi_distribution = deferred(Column(JSON, nullable=True))
