from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    user_query: str                  # Raw user query

    # Domain (Sector) Pack — pluggable Merchant Intelligence layer
    domain_pack: str                 # Active pack name ("none" disables the layer)
    merchant_context: dict           # MerchantContext.to_dict() — selected expert(s) & framing
    domain_doc: str                  # Per-data-group dataset description (markdown context)
    merchant_analysis: dict          # MerchantAnalysis.to_dict() — agentic expert findings

    # Intent Node output
    intent: str                      # "eda", "classification", "regression", "clustering", "report", "sql_query"
    intent_confidence: float         # 0-1 confidence score

    # DB Context Node output
    raw_schema: str                  # Raw MCP schema output
    db_intelligence: dict            # DatabaseIntelligenceModel dict
    relevant_tables: list[str]       # Filtered tables for planner

    # Manuel hedef seçimi (UI'dan gelir) — doluysa target_discovery_node bypass edilir
    target_col: str                  # Kullanıcının UI'dan el ile seçtiği hedef sütun

    # Veri modality ekseni (domain_pack'ten bağımsız) — UI'dan gelir
    modality: str                    # "" / "tabular" (varsayılan) | "timeseries"
    time_col: str                    # timeseries için kronolojik split'te satırları sıralayan zaman sütunu
    entity_col: str                  # opsiyonel multi-entity gruplama sütunu — "" = GLOBAL (tek seri); dolu = grup-içi lag/rolling/split

    # Rapor çıktı dili (UI dilinden gelir) — report_generator + quality_gate kullanır
    report_language: str             # "tr" | "en" — boş/eksik → "tr" (mevcut müşteri varsayılanı)

    # Target Discovery Node output
    target_definition: dict          # Target definition for modeling

    # Planner output
    analysis_plan: list[dict]        # Plan steps

    # Feature Plan output
    feature_plan: dict               # Feature engineering plan from LLM
    # Analist → feature_plan kanalı: HEDEF-KÖR yapısal ipuçları (yalnızca kolon adı,
    # sıfır istatistik). analysis_engine.extract_plan_hints() üretir; target-farkındalıklı
    # hiçbir bulgu (korelasyon skoru, feature önem sırası) bu kanaldan AKMAZ.
    analysis_hints: dict             # {generated_from, target_aware, constant_columns, ...}

    # Execution Node output
    data_path: str
    sql_results: list[dict]          # Raw data records
    profiling_report: dict           # Column statistics, null rates, dtypes

    # Feature Engineering output
    feature_report: dict             # Summary of applied transformations
    # Katman 1 inference artifact (wiring): ağır Artifact (param_store sklearn estimator taşır →
    # checkpoint'e giremez) diske yazılır, state'te YALNIZ id taşınır (model/split deseniyle aynı).
    inference_artifact_id: str       # data/artifacts/<id>/ ; None → yazılamadı ya da legacy mod
    inference_artifact: dict         # {artifact_id, deployable, phase, reason} — rapor/Katman 2 okur
    X_train: list[dict]              # Training features (DEPRECATED inline; bkz. *_path)
    X_test: list[dict]               # Test features (DEPRECATED inline; bkz. *_path)
    y_train: list                    # Training labels (DEPRECATED inline; bkz. *_path)
    y_test: list                     # Test labels (DEPRECATED inline; bkz. *_path)
    # Ağır split dizileri checkpoint payload'unu şişirmesin diye diske (parquet)
    # yazılır; state'te yalnızca dosya yolu taşınır (src/utils/dataset_store.py).
    X_train_path: str                # data/splits/*_xtrain.parquet
    X_test_path: str                 # data/splits/*_xtest.parquet
    y_train_path: str                # data/splits/*_ytrain.parquet
    y_test_path: str                 # data/splits/*_ytest.parquet

    # AutoML output
    model_name: str                  # Selected model name
    model_params: dict               # Hyperparameters
    model_artifact: Any              # Picklable trained model
    top_models: list[str]            # Top 2 Best Models
    
    # Evaluation output
    metrics: dict                    # Model evaluation metrics
    insights: list[str]              # LLM generated insights
    evaluation_report: dict          # Comprehensive evaluation report
    optimal_threshold: float         # Youden's J optimal decision threshold

    # Report Node output
    report: str                      # Final markdown report — report_sections'tan RENDER edilir
    # Katmanlı raporun YAPISAL kaynağı (src/core/report_schema.py). Her bölüm
    # headline/explanation (LLM yazar) + evidence/status/not_computed_reason (kod
    # doldurur, LLM dokunamaz) taşır. 'report' bundan üretilir; mevcut UI 'report'u
    # okumaya devam eder (kırılma yok).
    report_sections: list[dict]

    # HAM test taşıyıcısı: split ANINDA, encode/scale ÖNCESİ alınan test
    # partisyonunun test satırlarına POZİSYONEL hizalı tam snapshot'ı. X_test
    # encode+scale edilmiş olduğu için ham değer orada YOKTUR ve dump_split
    # index=False yazdığı için ham satıra geri join edilemez — bu taşıyıcı o
    # boşluğu kapatır. İki tüketici: fairness (ham kategorik gruplar) ve hata
    # analizi (ham sayısal değerler — "hatalı vakalarda 42 saat, doğrularda 17").
    raw_test_path: str               # data/splits/*_rawtest.parquet
    raw_test_meta: dict              # {segment_columns, carried_columns, mode, skipped, n_rows, reason}

    # Veri İlişkileri özeti (src/services/relations.py). Split ANINDA, gerçek train
    # partisyonundan, encode/scale ÖNCESİ hesaplanır → generated_from="train_only"
    # yapısal bir olgudur, iddia değil. Unsupervised dalda holdout yok →
    # "full_data_descriptive". report_sections katmanlı bölüme çevirir.
    relations_report: dict

    # SQL Query Node output
    sql_query_raw: str               # SQL Query 
    sql_query_results: list[dict]    # Raw query results (list of rows)
    sql_query_columns: list[str]     # Column names
    sql_query_chart: dict            # Chart data: {type, labels, datasets, title}
    sql_query_summary: str           # Generated short description
    sql_query_warning: str           # Warning message if approximate columns were used (empty = exact match)

    # Critic & Self-Correction
    retry_count: int                 # Number of retry attempts made
    max_retries: int                 # Maximum allowed retries
    # Append-only across retry loops — nodes emit only the new items (deltas);
    # the reducer concatenates them so feedback/history accumulate automatically.
    critic_feedback: Annotated[list[str], operator.add]   # Feedback from the critic node
    next_retry_node: str             # Node to route to after critic
    
    # Leakage Detection
    domain_leakage_report: dict
    # Critic'in 'too-good/leakage' teşhisinden gelen, retry'da deterministik
    # düşürülecek kolonlar (importance-bazlı leakage_suspects). feature_builder okur.
    forced_leakage_drops: list[str]

    # Entity Resolution
    entity_resolution: dict          # EntityResolution raporu (action, signals, stats)

    # Quality Gate
    quality_gate: dict

    # Data Quality / Validation Layer (iki katmanlı: yapısal yükleme-anı + anlamsal
    # modelleme-öncesi). Tek tutarlı katman, tek durum. data_quality_node yazar.
    #   status: "ok" | "degraded" | "failed"
    #   report: {structural, semantic, target_health, quarantine, autofixes, notes}
    # 'failed' → route_after_data_quality kısa-devre report_node; worker job'ı FAILED işaretler.
    data_quality_status: str
    data_quality_report: dict

    # Büyük-N model guard: n_train eşik üstündeyse ağır CPU-kilitli modeller kısıldı.
    # automl_node yazar, report_generator not düşer (sessiz değil). {} = tetiklenmedi.
    heavy_model_guard: dict

    # Retry History — append-only delta per turn (see critic_feedback note)
    retry_history: Annotated[list[dict], operator.add]
    # Audit Trail / Tracking
    experiment_logs: list[dict]      # Logs of completed ML experiments

    # Graph Control
    current_step: str                # Current active node
    error: Optional[str]             # Errors
    error_category: str              # classify_error() kategorisi: data/db/llm/timeout/memory/unknown
    messages: list                   # LLM conversation history
    excluded_models: list[str]       # Models excluded due to 
    
    # Analysis Report
    analysis_report: dict            # Deep analysis report