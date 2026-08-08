import os
from dotenv import load_dotenv

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Load environment variables from .env file.
# override=False: GERÇEK env var .env'i EZER (api/core/database.py ile aynı disiplin). Aksi hâlde .env'deki
# remote API_DATABASE_URL, testin/migration'ın açıkça set ettiği sqlite sandbox'ı ezerdi → koşu ortamı
# prod'a bağlanırdı. Yapısal kapı (assert_connection_allowed) backstop; bu satır deliği baştan kapatır.
load_dotenv(override=False)

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

    # Katman 1 terminus — feature pipeline'i KIM calistiriyor (gecici, T5'te KALDIRILACAK).
    #   legacy       : eski build_features yolu
    #   shadow       : eski yol CIKTIYI URETIR + orkestrator paralel kosar, fark LOGLANIR
    #                  (asla firlatilmaz). Amac: kapsam kapisini GERCEK veride sinamak.
    #   orchestrator : orkestrator uretim yolu (T2+) — VARSAYILAN (T5c cutover: date_features portlandi,
    #                  kapsam reddi tek sebep de kalkti → orkestrator artik varsayilan uretim yolu).
    # DIKKAT: bu flag iki kod yolu demektir — A-refactor'unun yapisal olarak imkansiz
    # kilmak istedigi sey. CIKIS OLCUTU olay-tabanli ve backlog'da AYRI kalem: T5d'de shadow dali +
    # legacy dal + flag BIRLIKTE silinir. shadow modu T5d'ye kadar KORUNUR (silinmez) — yalnizca
    # VARSAYILAN legacy'den orchestrator'a cevrildi (T5c). Olcutsuz birakilirsa kalici ikize doner.
    FEATURE_PIPELINE_MODE: str = os.getenv("FEATURE_PIPELINE_MODE", "orchestrator").strip().lower()

    # Katman 1 skorlama endpoint'i. SERVING_CACHE_MAX_ARTIFACTS: process-içi model cache
    # kapasitesi (LRU). DÜŞÜK varsayılan (2): joblib.load sıkıştırılmışı TAM boyutta RAM'e açar
    # (compress RAM'i DÜŞÜRMEZ) → 147MB×2 + API + transform frame'leri. İhtiyaç olursa artır.
    SERVING_CACHE_MAX_ARTIFACTS: int = int(os.getenv("SERVING_CACHE_MAX_ARTIFACTS", "2"))
    SERVING_MAX_ROWS: int = int(os.getenv("SERVING_MAX_ROWS", "10000"))

    # Katman 2 monitoring — input_row ALAN-BAZLI örnekleme (yalnız PSI referansı için ham girdi
    # saklanır; probability/prediction/input_warnings/actual_label HER satırda tam kalır → diğer
    # üç monitör körleşmez). Karar deterministik hash-Bernoulli(prediction_id): içerikten/sıradan
    # bağımsız uniform → sistematik-sapmasız, stateless (yarış yok). VARSAYILAN 1.0 (tam sakla):
    # asimetri kuralı — atılan satır geri gelmez, saklanan sonra silinebilir → tersinir tarafı seç.
    # Hacim şu an SIFIR (T5c açılmadı, üretimde champion yok); gerçek hacim gelince yeniden
    # değerlendirilir (handoff §3 backlog). PII'yi p ÇÖZMEZ (maskeleme+retention işi) — p yalnız
    # depolama sınırı. Mekanizma p<1.0 ile testli → tek config değişikliğiyle açılır.
    SERVING_INPUT_ROW_SAMPLE_RATE: float = float(os.getenv("SERVING_INPUT_ROW_SAMPLE_RATE", "1.0"))
    # Aggregator okuma-tarafı KESİN cap: pencerede saklanan input_row'dan en çok bu kadarı PSI'a
    # girer (min_n=300'e 3× marj — per-feature null + champion-bölme pencereyi küçültür).
    SERVING_INPUT_ROW_SAMPLE_TARGET: int = int(os.getenv("SERVING_INPUT_ROW_SAMPLE_TARGET", "1000"))

    # Katman 2 monitoring aggregator (zamanlanmış worker, `run_maintenance` alt-görevi). Pencere
    # başına satır sayısı bu eşiğin altındaysa status=insufficient_n. **PSI eşiği** (PSI ilk gerçek
    # metrik; input_row DOLU satırları kullanır → n_sampled ile kıyaslanır). İsim şimdiden `_PSI` ekli:
    # Parça 4'te `MONITORING_MIN_N_CBPE`(500)/`_QUALITY`(100) YANINA eklenir → yeniden-adlandırma borcu yok.
    MONITORING_MIN_N_PSI: int = int(os.getenv("MONITORING_MIN_N_PSI", "300"))
    # Parça 4 — CBPE (tahmini performans) + veri kalitesi min-n eşikleri. CBPE n_total (per-row
    # probability), kalite n_total (per-row bayrak). PSI'ın n_sampled'ından FARKLI (PSI input_row örneklem).
    MONITORING_MIN_N_CBPE: int = int(os.getenv("MONITORING_MIN_N_CBPE", "500"))
    MONITORING_MIN_N_QUALITY: int = int(os.getenv("MONITORING_MIN_N_QUALITY", "100"))
    # PSI sıfır-kova düzeltmesi: Laplace (add-α) pseudo-count, HEM referans HEM prod tarafına. Epsilon
    # DEĞİL — epsilon keyfi + N-duyarsız; Laplace N-farkında (n=300'de boş kova ≠ n=10k) + ilkeli
    # (Jeffreys α=0.5). Kategorik "other" kovası ref tarafında 0 olsa da (tail_mass=0) Laplace inf'i önler.
    MONITORING_PSI_LAPLACE_ALPHA: float = float(os.getenv("MONITORING_PSI_LAPLACE_ALPHA", "0.5"))
    # GERÇEKLEŞEN performans (actual_label backfill) min-n → labeled_n (etiketi GELEN satır sayısı),
    # n_total DEĞİL. 200: gerçekleşen metrik CBPE'nin doğrulaması → 100'de ROC-AUC gürültülü, yanlış bir
    # "gerçekleşen 0.79" CBPE'yi haksız şüpheli gösterir. accuracy 100'de yeterli ama ikisini ayrı eşiğe
    # bölmek karmaşık → tek eşik 200. label_coverage min-n altında bile HER ZAMAN görünür.
    MONITORING_MIN_N_REALIZED: int = int(os.getenv("MONITORING_MIN_N_REALIZED", "200"))
    # Seçilim yanlılığı güven eşiği: label_coverage bunun altındaysa gerçekleşen metrik "geçici/olası-yanlı"
    # işaretlenir (etiketler rastgele gelmez → düşük kapsam pozitif-ağırlıklı olabilir). Dashboard uygular.
    MONITORING_LABEL_COVERAGE_TRUST: float = float(os.getenv("MONITORING_LABEL_COVERAGE_TRUST", "0.5"))
    # Monitoring API (read-only, dashboard). `?days` varsayılanı + ÜST SINIR. Sınır ŞART: psi_per_feature
    # her pencerede tam JSON → days=3650 tüm geçmişi + kolon×pencere JSON'u çeker → yanıt patlar. summary
    # endpoint'i per_feature'ı DÖNDÜRMEZ (trend psi_overall/psi_max ile çizilir); per_feature yalnız latest +
    # tek-pencere detay endpoint'inde. days > MAX → MAX'a kırpılır (hata değil; dashboard sınırı config'ten okur).
    MONITORING_DEFAULT_WINDOW_DAYS: int = int(os.getenv("MONITORING_DEFAULT_WINDOW_DAYS", "30"))
    MONITORING_MAX_WINDOW_DAYS: int = int(os.getenv("MONITORING_MAX_WINDOW_DAYS", "90"))
    # FAZ 1 — schema (scoring playground formu) kategorik dropdown pratik-eşiği. raw_input_signature
    # `known_values`'u ZATEN `_SIG_MAX_VOCAB=50`'ye cap'li (üstü None); ama 50 değerli bir dropdown da
    # kullanışsız → form açısından pratik eşik DAHA DÜŞÜK. Bu eşiğin üstünde schema `known_values=null` +
    # `known_values_truncated=true` döner → form serbest metin + uyarıya geçer (dropdown yerine). İCAT
    # (form-ergonomisi kararı, literatür değil); env'den ayarlanır.
    MONITORING_SCHEMA_MAX_KNOWN_VALUES: int = int(os.getenv("MONITORING_SCHEMA_MAX_KNOWN_VALUES", "20"))
    # FAZ 1 — predictions log sayfa boyutu (keyset/cursor). DEFAULT istekte `limit` verilmezse; MAX üst-sınır
    # (limit > MAX → kırp, hata değil — days deseni). predictions TEK büyüyen tablo → büyük limit yanıtı şişirir.
    MONITORING_PREDICTIONS_PAGE_SIZE: int = int(os.getenv("MONITORING_PREDICTIONS_PAGE_SIZE", "50"))
    MONITORING_PREDICTIONS_PAGE_MAX: int = int(os.getenv("MONITORING_PREDICTIONS_PAGE_MAX", "200"))
    # MANUEL PROMOTION — artifact_events log sayfa boyutu (Deployments altı "Olaylar" paneli). Promotion
    # NADİR olay (champion elle değişir) → küçük default yeter; MAX üst-sınır (limit > MAX → kırp, hata değil).
    MONITORING_EVENTS_PAGE_SIZE: int = int(os.getenv("MONITORING_EVENTS_PAGE_SIZE", "50"))
    MONITORING_EVENTS_PAGE_MAX: int = int(os.getenv("MONITORING_EVENTS_PAGE_MAX", "200"))
    # ⚠️ LOOKBACK TUZAĞI (aggregator henüz tam-tarama, YAGNI): ileride `MONITORING_LOOKBACK_DAYS` alt-sınırı
    # eklenirse BEKLENEN ETİKET GECİKMESİNDEN büyük olmalı — churn'de etiket AYLAR sonra gelir; lookback
    # penceresi dışında kalan eski pencerenin geç etiketi asla recompute edilmez (dirty-kontrol onu görmez).

    # ══════════════════════════════════════════════════════════════════════════════════════════════
    # TRIAGE EŞİKLERİ (overall_status roll-up, `src/services/monitoring/triage.py`)
    # ⚠️ BU BÖLÜMDEKİ EŞİKLER LİTERATÜR DEĞİL, BAŞLANGIÇ DEĞERİDİR — shadow koşusu verisiyle KALİBRE
    # EDİLECEK (kalibrasyon turunda hepsi burada; dağınık olurlarsa bulmak zor). TEK İSTİSNA: PSI
    # 0.10/0.25 bandı literatürdür (endüstri standardı), aşağıda ayrıca işaretli.
    # ══════════════════════════════════════════════════════════════════════════════════════════════
    # PSI drift bandı — LİTERATÜR (icat DEĞİL; standart PSI eşikleri, tüm sektörde ortak).
    MONITORING_PSI_WARN: float = float(os.getenv("MONITORING_PSI_WARN", "0.10"))
    MONITORING_PSI_CRITICAL: float = float(os.getenv("MONITORING_PSI_CRITICAL", "0.25"))
    # Gerçekleşen performans düşüşü (baseline − realized) — İCAT, shadow-kalibre. En güçlü sinyal →
    # critical üretebilen TEK eksen (ölçülmüş gerçek). accuracy/roc_auc'ta mutlak düşüş.
    MONITORING_REALIZED_DROP_WARN: float = float(os.getenv("MONITORING_REALIZED_DROP_WARN", "0.05"))
    MONITORING_REALIZED_DROP_CRITICAL: float = float(os.getenv("MONITORING_REALIZED_DROP_CRITICAL", "0.10"))
    # Regresyon realized (rmse) — İCAT, shadow-kalibre. accuracy/roc 0-1 skalası regresyona uymaz →
    # GÖRELİ rmse artışı: (rmse − baseline_rmse)/baseline_rmse. Regresyon banded olmazsa bozulan model
    # sessizce "healthy" görünürdü (DEGEN_MASS'la aynı false-healthy riski) → banded tutulur.
    MONITORING_REALIZED_RMSE_INCREASE_WARN: float = float(os.getenv("MONITORING_REALIZED_RMSE_INCREASE_WARN", "0.10"))
    MONITORING_REALIZED_RMSE_INCREASE_CRITICAL: float = float(os.getenv("MONITORING_REALIZED_RMSE_INCREASE_CRITICAL", "0.25"))
    # CBPE tahmini düşüşü (baseline − estimated) — İCAT, shadow-kalibre. CBPE bir TAHMİN + concept-drift'te
    # iyimser sapar → şiddeti ne olursa olsun WARNING tavanı (asla critical); realized gelince devralır.
    MONITORING_CBPE_DROP_WARN: float = float(os.getenv("MONITORING_CBPE_DROP_WARN", "0.05"))
    # Drift ÖLÇÜM KAPSAMI yeterlilik eşikleri — İCAT, shadow-kalibre. importance_attributed_ratio bunun
    # ALTINDA ya da degenerate_importance_mass bunun ÜSTÜNDE ise drift ölçümü EKSİK → drift ekseni healthy
    # OLAMAZ (warning tabanı). DEGEN_MASS 0.15 (0.25 değil): modelin dörtte biri ölçülemezken "healthy"
    # kabul edilemez; muhafazakâr taraf ucuz (yanlış warning rahatsız eder, yanlış healthy YANILTIR).
    MONITORING_DRIFT_ATTR_MIN: float = float(os.getenv("MONITORING_DRIFT_ATTR_MIN", "0.75"))
    MONITORING_DEGEN_MASS_MAX: float = float(os.getenv("MONITORING_DEGEN_MASS_MAX", "0.15"))
    # Veri kalitesi alarm oranları — İCAT, shadow-kalibre. PSI'ın 0.10/0.25'i gibi bir LİTERATÜR STANDARDI
    # YOK. İlkeli tail_mass-çapası bu sinyale oturmaz: quality SATIR-düzeyi agregat (has_unseen boolean),
    # tail_mass KOLON-başına → eşlenemez; kolon-başına unseen toplamak örneklenmiş input_row körlüğünü geri
    # getirir (Parça 1/4). → config sabiti. (Gelecek: kolon-başına kalite sinyali gerekirse ayrı iş.)
    MONITORING_QUALITY_ALARM_UNSEEN_RATE: float = float(os.getenv("MONITORING_QUALITY_ALARM_UNSEEN_RATE", "0.05"))
    MONITORING_QUALITY_ALARM_OOR_RATE: float = float(os.getenv("MONITORING_QUALITY_ALARM_OOR_RATE", "0.03"))

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
