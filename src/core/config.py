import os
from dotenv import load_dotenv

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

class Settings:
    # API & MCP Configuration
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    # Break-glass admin password reset (closed platform, no email). Set at setup to a
    # long random value; empty disables the /auth/admin-reset-with-code endpoint.
    ADMIN_RESET_CODE: str = os.getenv("ADMIN_RESET_CODE", "")
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "9001"))
    MCP_SERVER_PATH: str = os.getenv("MCP_SERVER_PATH", "/mcp")
    RANDOM_SEED : int = 42
    ENABLE_DETERMINISM : bool = True

    # Multi-seed bagging: retrain the final single model under several seeds and
    # average their predictions (pure variance reduction). Only fires on the
    # single-model path (stacking / OOF-blend are already ensembles) and only for
    # seed-sensitive models. ROUNDS<2 disables it.
    SEED_BAGGING_ENABLED : bool = os.getenv("SEED_BAGGING_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
    SEED_BAGGING_ROUNDS : int = int(os.getenv("SEED_BAGGING_ROUNDS", "3"))

    # Extra baseline classifiers (decision_tree, naive_bayes, knn, svm, mlp) merged
    # into the classification catalog. Adds breadth at some runtime cost.
    EXTRA_MODELS_ENABLED : bool = os.getenv("EXTRA_MODELS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

    # Deployment environment. When set to "local" (env ENVIRONMENT=local), the
    # default dataset DB points at data/financial.db so the MCP server serves the
    # financial transaction dataset locally without any .env DB overrides.
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "").strip().lower()

    @property
    def MCP_SERVER_URL(self) -> str:
        return f"http://{self.MCP_SERVER_HOST}:{self.MCP_SERVER_PORT}{self.MCP_SERVER_PATH}"

    # Database Configuration
    # Parse DATABASES env var as JSON if available, otherwise fallback to DB_PROVIDER/DB_CONNECTION_STRING
    _databases_json: str = os.getenv("DATABASES", "{}")
    import json
    try:
        DATABASES: dict = json.loads(_databases_json)
    except json.JSONDecodeError:
        DATABASES = {}

    if not DATABASES:
        logger.info(DATABASES)
        DATABASES = {
            "default": {
                "provider": os.getenv("DB_PROVIDER", "sqlite"),
                "connection_string": os.getenv(
                    "DB_CONNECTION_STRING")
            }
        }


    HISTORY_DB_PROVIDER: str = os.getenv("HISTORY_DB_PROVIDER", "sqlite")
    HISTORY_DB_CONNECTION_STRING: str = os.getenv(
        "HISTORY_DB_CONNECTION_STRING",
        f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'history.db')}"
    )

    # LLM Configuration
    DEFAULT_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    DEFAULT_LLM_MODEL: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    INTELLIGENCE_ENGINE_PROVIDER: str = os.getenv("INTELLIGENCE_ENGINE_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
    INTELLIGENCE_ENGINE_MODEL: str = os.getenv("INTELLIGENCE_ENGINE_MODEL", "gpt-4o")
    DEFAULT_TEMPERATURE: float = 0.0

    # ── LLM reliability primitives (C1) ───────────────────────────────────────
    # Every get_llm() instance carries these so a single transient 429/500/timeout
    # is retried by the provider SDK (exponential backoff) instead of aborting a
    # multi-minute job. Applied via each provider's native constructor params so
    # the returned object stays a BaseChatModel (callers still use
    # .with_structured_output / create_agent on it).
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
    # 0/negatif → cap uygulanmaz (yapısal JSON çıktısını kesmemek için varsayılan 0).
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "0"))

    # Tool-using expert council + synthesizer model tier (H2). Multi-step ReAct
    # reasoning degrades on a mini model; route the council to the heavier
    # intelligence-engine model by default. Falls back to INTELLIGENCE_ENGINE_*.
    COUNCIL_MODEL: str = os.getenv("COUNCIL_MODEL", os.getenv("INTELLIGENCE_ENGINE_MODEL", "gpt-4o"))
    COUNCIL_PROVIDER: str = os.getenv("COUNCIL_PROVIDER", os.getenv("INTELLIGENCE_ENGINE_PROVIDER", os.getenv("LLM_PROVIDER", "openai")))

    # ── Cost / token accounting + budget (C3) ─────────────────────────────────
    # A LangChain callback records usage_metadata per call into the llm_usage
    # table (keyed by project_id). ENABLE_LLM_USAGE_TRACKING toggles recording;
    # LLM_TOKEN_BUDGET (>0) aborts a run once cumulative tokens exceed the cap.
    ENABLE_LLM_USAGE_TRACKING: bool = os.getenv("ENABLE_LLM_USAGE_TRACKING", "true").strip().lower() in ("1", "true", "yes", "on")
    LLM_TOKEN_BUDGET: int = int(os.getenv("LLM_TOKEN_BUDGET", "0"))

    # ── Merchant/Domain council gating (H1) ───────────────────────────────────
    # The expensive concurrent council only adds value on the intents that
    # consume its findings (feature engineering / interpretation). Plain SQL, EDA
    # and report intents skip it. Comma-separated intent allow-list.
    COUNCIL_INTENTS: set = {
        s.strip() for s in os.getenv(
            "COUNCIL_INTENTS",
            "classification,regression,clustering,deep_analysis",
        ).split(",") if s.strip()
    }

    # Ollama generation limits. The defaults (num_ctx=2048, capped num_predict)
    # are far too small for schema-analysis prompts + their large JSON output:
    # the prompt overflows the context window (→ empty response) or the JSON is
    # truncated mid-object (→ JSONDecodeError). Bump both generously.
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))
    # Ollama server endpoint. Default is the local daemon; a DB-configured provider
    # can override this per-connection (get_llm threads base_url into ChatOllama).
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Intelligence Engine schema-analysis cache (project bazlı, application DB'de).
    # analyze_schema çıktısı aynı şema için deterministik (temperature=0); cache
    # pahalı LLM çağrısını tekrar etmeyi önler. Cache key (model + şema) hash'i
    # olduğu için şema/dataset değişince otomatik invalidate olur.
    ENABLE_INTELLIGENCE_CACHE: bool = os.getenv("ENABLE_INTELLIGENCE_CACHE", "true").lower() in ("1", "true", "yes")

    # Domain (Sector) Pack Configuration
    # Selects the active Merchant/Domain Intelligence pack that augments the
    # pipeline (intent → query → feature engineering → interpretation).
    # "none" → NullDomainPack (no behavioral change; pipeline stays domain-agnostic).
    # Per-run override is possible via initial_state["domain_pack"].
    DOMAIN_PACK: str = os.getenv("DOMAIN_PACK", "none")

    # ── Database Maintenance / Cleanup ────────────────────────────────────────
    # Worker sürecinde çalışan (API'yi bloklamayan) periyodik temizlik görevi:
    # şişen checkpoint tabloları (LangGraph), eski background_jobs, app_logs ve
    # orphan cache satırlarını budar. İlk çalıştırmada bir kez de temizlik yapar.
    MAINTENANCE_ENABLED: bool = os.getenv("MAINTENANCE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
    MAINTENANCE_RUN_ON_STARTUP: bool = os.getenv("MAINTENANCE_RUN_ON_STARTUP", "true").strip().lower() in ("1", "true", "yes", "on")
    MAINTENANCE_INTERVAL_HOURS: float = float(os.getenv("MAINTENANCE_INTERVAL_HOURS", "24"))
    # Terminal (completed/failed) projelerin checkpoint'leri bu günden eskiyse silinir.
    # Orphan (projesi silinmiş) thread'ler yaşına bakılmaksızın silinir.
    CHECKPOINT_RETENTION_DAYS: int = int(os.getenv("CHECKPOINT_RETENTION_DAYS", "7"))
    # Tamamlanmış/başarısız background_jobs bu günden eskiyse silinir.
    JOB_RETENTION_DAYS: int = int(os.getenv("JOB_RETENTION_DAYS", "7"))
    # app_logs retention (gün). 0/negatif → log budama atlanır.
    LOG_RETENTION_DAYS: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))

    # ── Data Quality / Validation Layer (iki katmanlı: yapısal + anlamsal) ─────
    # Bozuk CSV / kolon kayması / rol-içerik çelişkisi otonom pipeline'ı sessizce
    # bozmasın diye deterministik (LLM'siz) veri kalitesi kapısının eşikleri.
    # Hepsi env override'lı. Sinyaller domain/dil bağımsız (ad-morfolojisi + istatistik).
    #
    # nunique BİRİNCİL sinyaldir: nunique==DQ_BINARY_NUNIQUE (2) → KESİN binary,
    # distinct_ratio ne olursa olsun "continuous/bozuk" DENMEZ. distinct_ratio yalnız
    # yüksek-kardinaliteli target'ta destekleyici teyit sinyalidir.
    DQ_BINARY_NUNIQUE: int = int(os.getenv("DQ_BINARY_NUNIQUE", "2"))
    # Classification target için üst kardinalite. Üstü → sınıf etiketi değil (aday sürekli).
    DQ_TARGET_MAX_CARD: int = int(os.getenv("DQ_TARGET_MAX_CARD", "50"))
    # Yalnız destekleyici: numeric target'ta distinct/non_null bunun üstündeyse sürekliliğe kanıt.
    DQ_CONTINUOUS_RATIO: float = float(os.getenv("DQ_CONTINUOUS_RATIO", "0.20"))
    # Küçük-n koruması: target sürekli-bozulma verdiği için gereken min non-null satır.
    DQ_MIN_ROWS_TARGET_CHECK: int = int(os.getenv("DQ_MIN_ROWS_TARGET_CHECK", "30"))
    # VARCHAR kolonun bu oranda TRY_CAST-numeric parse edilebilmesi → "sayı ama string"
    # kesin reader-misconfig → otomatik cast (auto-fix).
    DQ_NUMERIC_STRING_RATE: float = float(os.getenv("DQ_NUMERIC_STRING_RATE", "0.95"))
    # Yapısal karantina oranı (alan sayısı ≠ header) bunu aşarsa → her akışta 'failed'.
    # Altında: target'lı akışta yine 'failed' (etiket hizası kanıtlanamaz), target'sız akışta 'degraded'.
    DQ_STRUCT_FAIL_RATE: float = float(os.getenv("DQ_STRUCT_FAIL_RATE", "0.02"))

    # ── Satır capleri: transport tavanı vs stratified training cap ────────────
    # İKİ AYRI kavram (bilinçli): transport first-N keser, training cap stratified
    # örnekler. Aynı değer ikisine verilirse stratifikasyon imkânsız olur (feature
    # engineering, örnekleyeceği popülasyonu göremez). Bkz. tasarım notu.
    #
    # MAX_FETCH_ROWS — TRANSPORT/BELLEK tavanı: kaç satır tel üzerinden taşınıp RAM'e
    #   alınır. Kullanılan yer: data_executor fetch LIMIT + iki encode_result çağrısı
    #   (mcp_server, data_group CSV). Tam veri feature engineering'e ulaşsın diye yüksek.
    MAX_FETCH_ROWS: int = int(os.getenv("MAX_FETCH_ROWS", "300000"))
    # MAX_TRAINING_ROWS — STRATIFIED training cap: kaç satırda eğitilir. Aşımda sınıf
    #   oranını KORUYAN stratified örnekleme (imbalance downstream SMOTE/class_weight'te
    #   halledilir) + loglu + rapora yansıyan (sessiz truncate YOK). Kademeli açmak için
    #   düşük/güvenli default; env ile büyütülür (ör. 300000 → tam eğitim).
    MAX_TRAINING_ROWS: int = int(os.getenv("MAX_TRAINING_ROWS", "50000"))

    # ── Dengesiz veride model seçim metriği ───────────────────────────────────
    # PR_AUC_SCORING_THRESHOLD — azınlık sınıfının PAYI (min/total) bu eşiğin altına
    #   düşerse binary classification'da CV scoring'i roc_auc yerine average_precision
    #   (PR-AUC) olur. Gerekçe: aşırı dengesizlikte devasa negatif kütle FPR'ı ezer,
    #   ROC iyimser ve ayırt etmez hale gelir; iş açısından anlamlı soru "kaç pozitifi
    #   kullanılabilir kesinlikle yakalıyoruz", o da PR-AUC'tur.
    #   Eşik ÖLÇÜMLE seçildi: referans veri setlerinde iki metrik ancak ~%5 altında
    #   farklı model seçiyor (kredi %21.8 ve churn %20.4'te kazanan AYNI; e-ticaret
    #   fraud %2.2 ve ULB %0.17'de DEĞİŞİYOR). %5 üstünde anahtarı çevirmek referans
    #   skorları bedelsiz kaydırırdı.
    PR_AUC_SCORING_THRESHOLD: float = float(os.getenv("PR_AUC_SCORING_THRESHOLD", "0.05"))

    # ── Büyük-N model guard ───────────────────────────────────────────────────
    # Eğitim satırı (n_train) bu eşiği aşınca ağır CPU-kilitli modeller roster'dan
    # kısılır: AdaBoost düşer (yavaş + boosting üçlüsünün gerisinde), RF/ExtraTrees
    # n_estimators HEAVY_MODEL_TREES'e çekilir (model roster'da kalır → şampiyon
    # seçimi stabil; dropping değil). Sinyal SATIR SAYISINA bağlı, kolon/veri adına
    # değil. Eşik-altı → tüm roster açık (mevcut davranış birebir). Sessiz değil:
    # loglanır + rapora yansır (TR/EN). Default 100k: credit(~26k)/churn(~8k)/
    # fraud@50k(40k) eşiğin ALTINDA → roster + ROC hiç değişmez.
    HEAVY_MODEL_MAX_ROWS: int = int(os.getenv("HEAVY_MODEL_MAX_ROWS", "100000"))
    # Guard tetiklenince RF/ExtraTrees'in düşürüleceği ağaç sayısı (200→bu).
    HEAVY_MODEL_TREES: int = int(os.getenv("HEAVY_MODEL_TREES", "100"))

    # ── Analist → feature_plan ipucu kanalı ───────────────────────────────────
    # ENABLE_PLAN_HINTS — analysis_engine.extract_plan_hints() çıktısının feature
    #   plan prompt'una akıp akmayacağı. Kapalıysa prompt birebir eski haline döner
    #   (A/B ölçümünün "hints-off" kolu budur).
    ENABLE_PLAN_HINTS: bool = os.getenv("ENABLE_PLAN_HINTS", "true").strip().lower() in ("1", "true", "yes", "on")
    # ENABLE_LOCKED_HOLDOUT — ÖLÇÜM MODU (varsayılan kapalı). Açıkken ipuçları tüm
    #   veriden değil, feature_builder'ın train partisyonu YENİDEN ÜRETİLEREK yalnızca
    #   train satırlarından hesaplanır: aynı seed, aynı TEST_SIZE, hedefe göre stratify.
    #   Test satırları analistin görüş alanına KİLİTLENİR (eğitimden çıkarılmaz →
    #   eğitim verisi kaybı YOK, baseline kaymaz).
    #   Kullanımı: aynı veri setini ENABLE_LOCKED_HOLDOUT=0/1 ile koştur, AYNI iç test
    #   setindeki ROC'u karşılaştır. Fark ±0.005'i aşıyorsa ipucu kanalı test bilgisi
    #   taşıyor demektir → sızıntı. Fark yoksa kanal temiz.
    #   Sınır: yalnızca satır düzeyli planlarda (aggregation.required=False) split
    #   birebir yeniden üretilebilir; plan agrege ediyorsa mod kendini devre dışı bırakır
    #   ve generated_from="full_data" olarak işaretlenir (sessiz geçilmez, loglanır).
    ENABLE_LOCKED_HOLDOUT: bool = os.getenv("ENABLE_LOCKED_HOLDOUT", "false").strip().lower() in ("1", "true", "yes", "on")

    # ── Fairness / segment taşıyıcısı ─────────────────────────────────────────
    # FAIRNESS_SEGMENT_MODE — split anında HAM hâlde snapshot'lanacak kolonlar:
    #   "narrow" (varsayılan) → yalnızca demografik sezgiselin yakaladıkları
    #       (age/sex/gender/race/... + db_intelligence semantic_type=demographic).
    #       Bugünkü fairness davranışının kapsamı; state boyutu küçük kalır.
    #   "wide"   → TÜM düşük kardinaliteli kategorikler. "Her segmentte performans"
    #       analizini açar ama checkpoint payload'unu büyütür. Kod değişikliği
    #       gerektirmez — yalnızca bu ayar.
    FAIRNESS_SEGMENT_MODE: str = os.getenv("FAIRNESS_SEGMENT_MODE", "narrow").strip().lower()
    # "wide" modda bir kategorik kolonun segment sayılması için üst kardinalite sınırı.
    FAIRNESS_SEGMENT_MAX_CARDINALITY: int = int(os.getenv("FAIRNESS_SEGMENT_MAX_CARDINALITY", "10"))
    # Bir grup için metrik HESAPLANMASININ alt sınırı. Altındaki gruplar rapordan
    # SİLİNMEZ — görünür sebeple ("örneklem çok ince") listelenir. Sessiz atlama yok.
    FAIRNESS_MIN_GROUP_SIZE: int = int(os.getenv("FAIRNESS_MIN_GROUP_SIZE", "5"))

settings = Settings()
