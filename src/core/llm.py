from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from src.core.config import settings

def get_llm(
    model_name: Optional[str] = None, 
    temperature: Optional[float] = None,
    provider: Optional[str] = None
) -> BaseChatModel:
    """
    Returns an instance of a language model with the specified or default configuration.
    Dynamically switches between different providers (openai, anthropic, google).
    
    Args:
        model_name: The name of the LLM model to use. Defaults to settings.DEFAULT_LLM_MODEL.
        temperature: The temperature parameter for the LLM. Defaults to settings.DEFAULT_TEMPERATURE.
        provider: The provider to use ('openai', 'anthropic', 'google'). Defaults to settings.DEFAULT_LLM_PROVIDER.
        
    Returns:
        BaseChatModel: The configured language model instance.
    """
    actual_provider = (provider if provider is not None else settings.DEFAULT_LLM_PROVIDER).lower()
    actual_model_name = model_name if model_name is not None else settings.DEFAULT_LLM_MODEL
    actual_temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
    
    if actual_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=actual_model_name, temperature=actual_temperature)
        
    elif actual_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Default fallback for anthropic if openai default was passed implicitly
        if actual_model_name == settings.DEFAULT_LLM_MODEL and "claude" not in actual_model_name:
            actual_model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(model_name=actual_model_name, temperature=actual_temperature)
        
    elif actual_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Default fallback for google if openai default was passed implicitly
        if actual_model_name == settings.DEFAULT_LLM_MODEL and "gemini" not in actual_model_name:
            actual_model_name = "gemini-1.5-pro"
        return ChatGoogleGenerativeAI(model=actual_model_name, temperature=actual_temperature)
        
    elif actual_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=actual_model_name, temperature=actual_temperature)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {actual_provider}. Supported providers: openai, anthropic, google, ollama")
