import sys
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from src.core.config import settings
from src.core.llm_usage import usage_callback
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# Per-role env fallbacks. Used when the DB has no assignment for the role and the
# caller passed no explicit provider/model (preserves the pre-DB behaviour).
def _env_defaults(role: str) -> tuple[str, str]:
    if role == "intelligence":
        return settings.INTELLIGENCE_ENGINE_PROVIDER, settings.INTELLIGENCE_ENGINE_MODEL
    if role == "council":
        return settings.COUNCIL_PROVIDER, settings.COUNCIL_MODEL
    # "main" and anything unrecognised
    return settings.DEFAULT_LLM_PROVIDER, settings.DEFAULT_LLM_MODEL


def _resolve_role_config(role: str) -> Optional[dict]:
    """
    Look up the DB-configured provider/model/credentials for ``role``. Imported lazily
    so ``src`` stays importable without the ``api`` package and to avoid an import
    cycle. Any failure → None (caller falls back to explicit args / env).
    """
    try:
        from api.core.ai_settings import get_role_config
        return get_role_config(role)
    except Exception:
        return None


def _log_model_choice(role: str, provider: str, model: str, source: str, temperature: float) -> None:
    """
    Hangi node'un hangi modelle koştuğunu tek satırda yazar. Kaynak alanı seçimin
    NEREDEN geldiğini söyler: ``db`` (admin panelindeki rol ataması), ``arg``
    (çağıranın verdiği provider/model) veya ``env`` (settings default'u) — model
    beklenenden farklı çıktığında hangi katmanın kazandığı buradan okunur.

    Çağıran modül ismi stack'ten alınır (``sys._getframe``; inspect.stack() aksine
    frame nesnesi kurmaz, node başına birkaç mikrosaniye). İnce ``_get_llm``
    sarmalayıcıları (talktodata/utils.py, report_generator.py) atlanır — yoksa her
    satırda sarmalayıcı görünür, asıl node görünmezdi. Log seviyesi INFO: bir graph
    koşusunda ~10-30 satır eder, gürültü yapmaz.
    """
    caller = "?"
    try:
        frame = sys._getframe(2)
        for _ in range(5):
            if frame is None:
                break
            if frame.f_code.co_name != "_get_llm":
                caller = frame.f_globals.get("__name__", "?")
                break
            frame = frame.f_back
    except Exception:
        pass
    logger.info(
        "LLM seçimi → %s | model=%s | rol=%s | kaynak=%s | temp=%s | çağıran=%s",
        provider, model, role, source, temperature, caller,
    )


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
    role: str = "main",
) -> BaseChatModel:
    """
    Returns an instance of a language model with the specified or default configuration.
    Dynamically switches between different providers (openai, anthropic, google, ollama).

    Configuration is resolved per ``role`` ("main" | "intelligence" | "council") in this
    order of precedence:
      1. **DB assignment** — an admin-configured provider/model for the role (with its
         api_key / base_url) is authoritative and overrides everything.
      2. **Explicit args** — the ``provider`` / ``model_name`` passed by the caller.
      3. **Env defaults** — ``settings`` values for that role (legacy behaviour).
    With no DB rows the behaviour is identical to the pre-DB config, so existing callers
    keep working unchanged.

    Reliability primitives (C1) are baked in for every instance: provider-native
    ``max_retries`` (exponential backoff on transient 429/500/timeout) and a
    ``request_timeout`` so one blip no longer aborts a multi-minute job. These use
    each SDK's constructor params, so the return value stays a real ``BaseChatModel`` —
    callers can still ``.with_structured_output(...)`` / pass it to ``create_agent``.

    Every instance also carries the usage-accounting callback (C3): when a
    ``track_usage`` block is active (the worker enters one per job) token usage is
    recorded and the per-run budget is enforced; otherwise the callback is a no-op.

    Args:
        model_name: The LLM model to use. Fallback only — a DB assignment wins.
        temperature: Sampling temperature. Defaults to settings.DEFAULT_TEMPERATURE.
        provider: 'openai' | 'anthropic' | 'google' | 'ollama'. Fallback only.
        role: Which system role this call is for — selects the DB assignment / env
            defaults to fall back to. Defaults to 'main'.

    Returns:
        BaseChatModel: The configured language model instance.
    """
    role_cfg = _resolve_role_config(role)

    # Credentials only come from a DB-configured provider; env-based providers read
    # their keys from the environment inside each SDK (unchanged legacy path).
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    if role_cfg:
        actual_provider = role_cfg["provider_type"]
        actual_model_name = role_cfg["model_name"]
        api_key = role_cfg.get("api_key")
        base_url = role_cfg.get("base_url")
        source = "db"
    else:
        env_provider, env_model = _env_defaults(role)
        actual_provider = provider if provider is not None else env_provider
        actual_model_name = model_name if model_name is not None else env_model
        source = "arg" if (provider is not None or model_name is not None) else "env"

    actual_provider = actual_provider.lower()
    actual_temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE

    # Provider-özel default düzeltmeleri. Log satırından ÖNCE yapılır ki loglanan
    # model gerçekten çağrılan model olsun (aşağıdaki branch'ler bunu tekrar etmez).
    if actual_provider == "anthropic" and actual_model_name == settings.DEFAULT_LLM_MODEL \
            and "claude" not in actual_model_name:
        actual_model_name = "claude-3-5-sonnet-20240620"
    elif actual_provider == "google" and actual_model_name == settings.DEFAULT_LLM_MODEL \
            and "gemini" not in actual_model_name:
        actual_model_name = "gemini-1.5-pro"

    _log_model_choice(role, actual_provider, actual_model_name, source, actual_temperature)

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
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    elif actual_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(
            model_name=actual_model_name, temperature=actual_temperature,
            max_retries=max_retries, timeout=timeout,
            # Anthropic requires max_tokens; use the configured cap or a sane default.
            max_tokens=max_tokens if max_tokens > 0 else 4096, **common,
        )
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)

    elif actual_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(model=actual_model_name, temperature=actual_temperature,
                      max_retries=max_retries, timeout=timeout, **common)
        if max_tokens > 0:
            kwargs["max_output_tokens"] = max_tokens
        if api_key:
            kwargs["google_api_key"] = api_key
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
            base_url=base_url or settings.OLLAMA_BASE_URL,
            **common,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {actual_provider}. Supported providers: openai, anthropic, google, ollama")
