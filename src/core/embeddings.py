from typing import Optional
from langchain_core.embeddings import Embeddings
from src.core.config import settings

# Per-provider sensible default embedding models. Used when the configured
# EMBEDDING_MODEL clearly does not belong to the selected provider (e.g. the
# OpenAI default was left in place but the provider switched to ollama).
_PROVIDER_DEFAULTS = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "google": "models/text-embedding-004",
}


def get_embeddings(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> Embeddings:
    """
    Returns an embeddings model with the specified or default configuration.
    Dynamically switches between providers (openai, ollama, google) purely via
    env — same philosophy as get_llm() in src/core/llm.py.

    Args:
        model_name: Embedding model name. Defaults to settings.EMBEDDING_MODEL.
        provider:   Provider ('openai', 'ollama', 'google'). Defaults to
                    settings.EMBEDDING_PROVIDER.

    Returns:
        Embeddings: a configured LangChain embeddings instance.
    """
    actual_provider = (provider if provider is not None else settings.EMBEDDING_PROVIDER).lower()
    actual_model_name = model_name if model_name is not None else settings.EMBEDDING_MODEL

    # If the model name still matches the global default but the provider isn't
    # OpenAI, swap in that provider's default so a single env var switch works.
    if actual_model_name == settings.EMBEDDING_MODEL and actual_provider != "openai":
        actual_model_name = _PROVIDER_DEFAULTS.get(actual_provider, actual_model_name)

    if actual_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=actual_model_name)

    elif actual_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=actual_model_name)

    elif actual_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=actual_model_name)

    else:
        raise ValueError(
            f"Unsupported embedding provider: {actual_provider}. "
            "Supported providers: openai, ollama, google"
        )
