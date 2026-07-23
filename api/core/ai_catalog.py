"""
Curated, static catalog of known chat models per provider type. Single source of
truth for the admin UI's model dropdowns (served via GET /admin/ai/catalog).

These lists are hints, not hard limits: the API accepts any model name so new
releases and custom/Ollama tags aren't blocked (``allow_custom`` tells the UI to
also render a free-text field). Keep entries curated but current.
"""
from typing import Dict, List

SUPPORTED_PROVIDER_TYPES = ("openai", "anthropic", "google", "ollama")

# Ollama's default local endpoint; surfaced so the UI can pre-fill the base_url field.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

PROVIDER_CATALOG: Dict[str, List[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o3-mini",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "google": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ],
    # Ollama tags are user-specific (whatever they've pulled locally); these are just
    # common defaults and the UI allows free-text entry.
    "ollama": [
        "llama3.1",
        "llama3.2",
        "qwen2.5",
        "mistral",
        "gemma2",
    ],
}

# Providers that must have an API key before a model of theirs can be assigned to a
# role. Ollama runs locally and needs only a base_url.
REQUIRES_API_KEY = {"openai", "anthropic", "google"}


def get_catalog() -> dict:
    """Shape returned to the frontend: per-provider models + UI hints."""
    return {
        "providers": [
            {
                "type": ptype,
                "models": models,
                "requires_api_key": ptype in REQUIRES_API_KEY,
                "allow_custom": ptype == "ollama",
                "default_base_url": DEFAULT_OLLAMA_BASE_URL if ptype == "ollama" else None,
            }
            for ptype, models in PROVIDER_CATALOG.items()
        ]
    }
