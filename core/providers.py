"""Provider factory: LLM and Embedding instances for any supported backend.

All provider-specific imports are localised here. The rest of the codebase
calls ``get_llm()`` and ``get_embeddings()`` without knowing which provider
is active.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama pre-flight checks
# ---------------------------------------------------------------------------
def check_ollama_connection(base_url: str | None = None) -> tuple[bool, str]:
    """Check if Ollama server is reachable. Returns (ok, message)."""
    import httpx

    url = base_url or config.OLLAMA_BASE_URL
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return True, f"Connected to Ollama at {url}"
    except httpx.ConnectError:
        return False, f"Cannot connect to Ollama at {url}. Is it running?"
    except Exception as e:
        return False, f"Ollama connection error: {e}"


def check_ollama_model(model: str, base_url: str | None = None) -> tuple[bool, str]:
    """Check if a specific model is available on the Ollama server."""
    import httpx

    url = base_url or config.OLLAMA_BASE_URL
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        # Match with or without :latest tag
        matched = any(
            model == name or model == name.split(":")[0] or f"{model}:latest" == name
            for name in available
        )
        if matched:
            return True, f"Model '{model}' is available"
        return False, (
            f"Model '{model}' not found on Ollama at {url}. "
            f"Available models: {', '.join(available) or 'none'}. "
            f"Pull it with: ollama pull {model}"
        )
    except httpx.ConnectError:
        return False, f"Cannot connect to Ollama at {url}"
    except Exception as e:
        return False, f"Error checking model: {e}"


def validate_provider_setup(
    llm_provider: str | None = None,
    llm_model: str | None = None,
    emb_provider: str | None = None,
    emb_model: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate that the selected providers and models are accessible.

    Returns (all_ok, list_of_error_messages).
    """
    errors: list[str] = []
    llm_provider = (llm_provider or config.LLM_PROVIDER).lower()
    llm_model = llm_model or config.LLM_MODEL
    emb_provider = (emb_provider or config.EMBEDDING_PROVIDER).lower()
    emb_model = emb_model or config.EMBEDDING_MODEL

    if llm_provider == "ollama":
        ok, msg = check_ollama_connection()
        if not ok:
            errors.append(f"LLM: {msg}")
        else:
            ok, msg = check_ollama_model(llm_model)
            if not ok:
                errors.append(f"LLM: {msg}")

    if emb_provider == "ollama":
        ok, msg = check_ollama_connection()
        if not ok:
            errors.append(f"Embeddings: {msg}")
        else:
            ok, msg = check_ollama_model(emb_model)
            if not ok:
                errors.append(f"Embeddings: {msg}")

    if llm_provider == "openai" and not config.OPENAI_API_KEY:
        errors.append("LLM: OpenAI API key not set in .env.local")
    if llm_provider == "anthropic" and not config.ANTHROPIC_API_KEY:
        errors.append("LLM: Anthropic API key not set in .env.local")
    if llm_provider == "groq" and not config.GROQ_API_KEY:
        errors.append("LLM: Groq API key not set in .env.local")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Embedding factory (with caching to avoid reloading models)
# ---------------------------------------------------------------------------
_embedding_cache: dict[str, Embeddings] = {}


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """Return a LangChain Embeddings instance for the given provider/model.
    
    The instance is cached based on provider+model+kwargs to avoid reloading
    heavy models (especially HuggingFace) on every call.
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL

    # Create cache key from provider, model, and relevant kwargs
    cache_key_parts = [provider, model]
    if provider == "ollama":
        cache_key_parts.append(str(kwargs.get("base_url", config.OLLAMA_BASE_URL)))
    elif provider == "openai":
        cache_key_parts.append(str(kwargs.get("api_key", config.OPENAI_API_KEY)[:10] if config.OPENAI_API_KEY else ""))
    cache_key = "|".join(cache_key_parts)

    # Return cached instance if available
    if cache_key in _embedding_cache:
        logger.debug("Reusing cached embedding model: %s", cache_key)
        return _embedding_cache[cache_key]

    # Create new instance
    logger.info("Loading embedding model: %s (%s)", model, provider)
    
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        instance = OllamaEmbeddings(
            model=model,
            base_url=kwargs.get("base_url", config.OLLAMA_BASE_URL),
        )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        instance = OpenAIEmbeddings(
            model=model,
            api_key=kwargs.get("api_key", config.OPENAI_API_KEY),
        )

    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # HuggingFaceEmbeddings will use cached model files from ~/.cache/huggingface/
        # The first load downloads the model; subsequent loads use the cache
        instance = HuggingFaceEmbeddings(model_name=model)

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    # Cache and return
    _embedding_cache[cache_key] = instance
    return instance


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    streaming: bool = True,
    **kwargs,
) -> BaseChatModel:
    """Return a LangChain ChatModel instance for the given provider/model."""
    provider = (provider or config.LLM_PROVIDER).lower()
    model = model or config.LLM_MODEL

    max_tokens = kwargs.get("max_tokens")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        ollama_kwargs: dict = dict(
            model=model,
            temperature=temperature,
            base_url=kwargs.get("base_url", config.OLLAMA_BASE_URL),
            streaming=streaming,
        )
        if max_tokens is not None:
            ollama_kwargs["num_predict"] = max_tokens
        return ChatOllama(**ollama_kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        openai_kwargs: dict = dict(
            model=model,
            temperature=temperature,
            api_key=kwargs.get("api_key", config.OPENAI_API_KEY),
            streaming=streaming,
        )
        # Support custom base URL for OpenAI-compatible APIs (e.g. university GPU labs)
        base_url = kwargs.get("base_url") or config.OPENAI_API_BASE
        if base_url:
            openai_kwargs["base_url"] = base_url
        if max_tokens is not None:
            openai_kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**openai_kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs: dict = dict(
            model=model,
            temperature=temperature,
            api_key=kwargs.get("api_key", config.ANTHROPIC_API_KEY),
            streaming=streaming,
        )
        if max_tokens is not None:
            anthropic_kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**anthropic_kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        groq_kwargs: dict = dict(
            model=model,
            temperature=temperature,
            api_key=kwargs.get("api_key", config.GROQ_API_KEY),
            streaming=streaming,
        )
        if max_tokens is not None:
            groq_kwargs["max_tokens"] = max_tokens
        return ChatGroq(**groq_kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Helpers for UI dropdowns
# ---------------------------------------------------------------------------
def get_available_llm_models(provider: str) -> list[str]:
    """Return the list of known models for a given LLM provider."""
    return config.AVAILABLE_LLM_MODELS.get(provider.lower(), [])


def get_available_embedding_models(provider: str) -> list[str]:
    """Return the list of known models for a given embedding provider."""
    return config.AVAILABLE_EMBEDDING_MODELS.get(provider.lower(), [])
