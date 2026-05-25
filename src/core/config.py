import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class Settings:
    # API & MCP Configuration
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "9001"))
    MCP_SERVER_PATH: str = os.getenv("MCP_SERVER_PATH", "/mcp")
    
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
        DATABASES = {
            "default": {
                "provider": os.getenv("DB_PROVIDER", "sqlite"),
                "connection_string": os.getenv(
                    "DB_CONNECTION_STRING",
                    f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'adult_income.db')}"
                )
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
    INTELLIGENCE_ENGINE_MODEL: str = os.getenv("INTELLIGENCE_ENGINE_MODEL", "gpt-4o")
    DEFAULT_TEMPERATURE: float = 0.0

settings = Settings()
