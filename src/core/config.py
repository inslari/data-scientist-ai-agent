import os
from dotenv import load_dotenv

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

class Settings:
    # API & MCP Configuration
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "9001"))
    MCP_SERVER_PATH: str = os.getenv("MCP_SERVER_PATH", "/mcp")
    RANDOM_SEED : int = 42
    ENABLE_DETERMINISM : bool = True

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

    # Embedding Configuration (semantic text clustering — text-analysis feature).
    # Mirrors the LLM provider pattern: switch providers purely via env.
    # Providers: "openai" (default), "ollama", "google". Empty/unknown model name
    # falls back to a sensible per-provider default in get_embeddings().
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Ollama generation limits. The defaults (num_ctx=2048, capped num_predict)
    # are far too small for schema-analysis prompts + their large JSON output:
    # the prompt overflows the context window (→ empty response) or the JSON is
    # truncated mid-object (→ JSONDecodeError). Bump both generously.
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))

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

settings = Settings()
