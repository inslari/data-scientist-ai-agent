from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, ForeignKey
from api.core.database import Base
import time


class PredictionModel(Base):
    """Katman 2 temeli — her skorlama isteği satırının kaydı (input drift + performance backfill).

    Migration-safe: ``actual_label`` + ``label_backfilled_at`` BAŞTAN nullable → CBPE için sonradan
    etiket backfill'i migration GEREKTİRMEZ. ``artifact_id`` şart: champion zamanla değişir, drift
    belirli sürüme atfedilmeli. ``created_at`` index → pencereli drift sorguları. ``input_row`` RAW
    space → ``training_reference.per_feature`` (PSI) ile DOĞRUDAN kıyaslanabilir.

    ⚠️ PII NOTU: ``input_row`` HAM girdiyi saklar → isim/e-posta/PII taşıyabilir. Boyut da büyür
    (batch × ham kolon). Katman 2 ALAN-BAZLI örnekleme: ``input_row`` yalnız örneklem satırlarda
    dolu (gerisi NULL), ``input_row_sampled`` hangi satırın örneklendiğini işaretler. Diğer alanlar
    (prediction/probability/input_warnings/actual_label) HER satırda tam → yalnız PSI referansı
    seyrekleşir, gerçekleşen/tahmini performans + veri kalitesi monitörleri tam veri görür.
    Örnekleme oranı ``settings.SERVING_INPUT_ROW_SAMPLE_RATE`` (varsayılan 1.0 = tam sakla; asimetri:
    atılan geri gelmez). PII'nin gerçek çözümü maskeleme+retention (ayrı iş), örnekleme değil.

    YAZIM FAIL-OPEN: skorlama artifact-persistence ile aynı felsefe — monitoring ≠ serving doğruluğu.
    ``prediction_id`` yazımdan BAĞIMSIZ üretilir (response'ta her hâlükârda döner)."""

    __tablename__ = "predictions"

    prediction_id = Column(String, primary_key=True, index=True)     # uuid hex — response'ta döner
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    artifact_id = Column(String, index=True)                         # serve eden champion (drift atfı)
    created_at = Column(Integer, default=lambda: int(time.time()), index=True)
    # KAYNAK (Katman 2 filtre): "api" = üretim trafiği (harici/programatik çağrı) · "ui_manual" =
    # dashboard Scoring playground'dan MANUEL deneme. Aggregator ui_manual'ı VARSAYILAN olarak DIŞLAR
    # (drift/gerçekleşen/tahmini perf yalnız gerçek trafik) ama ``excluded_ui_count`` ile GÖRÜNÜR tutar
    # (sessizce yutulmaz). NULL = alan-öncesi kayıt → coalesce ile "api" sayılır (körleşme yok, güvenli
    # taraf: geçmiş üretim tahminleri dışlanmaz). default="api": kaynak omit edilen çağrı üretim varsayılır.
    source = Column(String, nullable=True, default="api")
    # İŞ ANAHTARI (v2 erken-şartı): geçmiş kayıt sonradan düzeltilemez → kolon ŞİMDİ, doldurma
    # skorlama anında (artifact bir entity kolonu bildirdiyse). backfill-by-business_key (bir anahtar
    # → çok tahmin → belirsizlik-çözüm kuralı ister) v2'ye BLOKLU; v1 backfill YALNIZ prediction_id ile.
    # Genelde entity/id kolonu model girdisi DEĞİL (serving required'a girmez) → v1'de çoğunlukla NULL.
    business_key = Column(String, nullable=True, index=True)
    input_row = Column(JSON, nullable=True)                          # RAW girdi (PSI); yalnız örneklem satırlarda dolu
    input_row_sampled = Column(Boolean, nullable=True)               # bu satır input_row örneklemine girdi mi (alan-bazlı)
    prediction = Column(JSON, nullable=True)                         # sınıf etiketi / reg değeri (tip-agnostik)
    probability = Column(JSON, nullable=True)                        # binary: float; multiclass: {sınıf: p}
    threshold = Column(Float, nullable=True)
    input_warnings = Column(JSON, nullable=True)                     # BATCH-agregatı (geriye-uyum; pencerede toplanamaz)
    input_warning_flags = Column(JSON, nullable=True)                # PER-ROW {has_unseen, has_out_of_range} — kalite pencere agregasyonu bundan
    # GERÇEKLEŞEN performans (CBPE'nin doğrulaması) için sonradan doldurulur (nullable → migration yok).
    # ``actual_label`` = /projects/{id}/labels endpoint'i (batch API / CSV) prediction_id ile eşleyip yazar.
    # ``label_backfilled_at`` FLOAT epoch (int değil) → dirty-kontrolün 1-saniye sınır yarışı kapanır:
    # aggregator computed_at'i (float) da run-başında yakalar; geç etiket her zaman > computed_at olur.
    actual_label = Column(JSON, nullable=True)
    label_backfilled_at = Column(Float, nullable=True)
