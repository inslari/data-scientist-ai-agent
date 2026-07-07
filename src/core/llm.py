from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from src.core.config import settings
from src.core.llm_usage import usage_callback


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    provider: Optional[str] = None
) -> BaseChatModel:
    """
    Returns an instance of a language model with the specified or default configuration.
    Dynamically switches between different providers (openai, anthropic, google).

    Reliability primitives (C1) are baked in for every instance: provider-native
    ``max_retries`` (exponential backoff on transient 429/500/timeout) and a
    ``request_timeout`` so one blip no longer aborts a multi-minute job. These use
    each SDK's constructor params, so the return value stays a real ``BaseChatModel``
    — callers can still ``.with_structured_output(...)`` / pass it to ``create_agent``.

    Every instance also carries the usage-accounting callback (C3): when a
    ``track_usage`` block is active (the worker enters one per job) token usage is
    recorded and the per-run budget is enforced; otherwise the callback is a no-op.

    Args:
        model_name: The LLM model to use. Defaults to settings.DEFAULT_LLM_MODEL.
        temperature: Sampling temperature. Defaults to settings.DEFAULT_TEMPERATURE.
        provider: 'openai' | 'anthropic' | 'google' | 'ollama'. Defaults to
            settings.DEFAULT_LLM_PROVIDER.

    Returns:
        BaseChatModel: The configured language model instance.
    """
    actual_provider = (provider if provider is not None else settings.DEFAULT_LLM_PROVIDER).lower()
    actual_model_name = model_name if model_name is not None else settings.DEFAULT_LLM_MODEL
    actual_temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE

    # Cross-provider reliability + observability kwargs. `metadata` tags each call
    # so the usage callback can attribute tokens to the right provider/model.
    common = {
        "callbacks": [usage_callback],
        "metadata": {"provider": actual_provider, "model": actual_model_name},
    }
    max_retries = settings.LLM_MAX_RETRIES
    timeout = settings.LLM_REQUEST_TIMEOUT
    max_tokens = settings.LLM_MAX_TOKENS  # 0 → leave unset (don't truncate JSON output)

    if actual_provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = dict(model=actual_model_name, temperature=actual_temperature,
                      max_retries=max_retries, timeout=timeout, **common)
        if max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    elif actual_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Default fallback for anthropic if openai default was passed implicitly
        if actual_model_name == settings.DEFAULT_LLM_MODEL and "claude" not in actual_model_name:
            actual_model_name = "claude-3-5-sonnet-20240620"
            common["metadata"]["model"] = actual_model_name
        # Anthropic requires max_tokens; use the configured cap or a sane default.
        return ChatAnthropic(
            model_name=actual_model_name, temperature=actual_temperature,
            max_retries=max_retries, timeout=timeout,
            max_tokens=max_tokens if max_tokens > 0 else 4096, **common,
        )

    elif actual_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Default fallback for google if openai default was passed implicitly
        if actual_model_name == settings.DEFAULT_LLM_MODEL and "gemini" not in actual_model_name:
            actual_model_name = "gemini-1.5-pro"
            common["metadata"]["model"] = actual_model_name
        kwargs = dict(model=actual_model_name, temperature=actual_temperature,
                      max_retries=max_retries, timeout=timeout, **common)
        if max_tokens > 0:
            kwargs["max_output_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    elif actual_provider == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx must hold the (large schema) prompt AND leave room for the JSON
        # output; num_predict caps generation. Ollama's defaults are too small and
        # truncate the structured response — see settings for details. Local models
        # have no API rate limits, so retries/timeout aren't applied here.
        return ChatOllama(
            model=actual_model_name,
            temperature=actual_temperature,
            num_ctx=settings.OLLAMA_NUM_CTX,
            num_predict=settings.OLLAMA_NUM_PREDICT,
            **common,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {actual_provider}. Supported providers: openai, anthropic, google, ollama")
