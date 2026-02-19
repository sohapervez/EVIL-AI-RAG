"""Provider factory: LLM and Embedding instances for any supported backend.

All provider-specific imports are localised here. The rest of the codebase
calls ``get_llm()`` and ``get_embedding_model()`` without knowing which provider
is active.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding factory (with caching to avoid reloading models)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def get_embedding_model(provider: str | None = None, model: str | None = None):
    """Return a LlamaIndex BaseEmbedding instance for the given provider/model.

    The instance is cached via lru_cache to avoid reloading heavy models
    (especially HuggingFace) on every call.
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL

    logger.info("Loading embedding model: %s (%s)", model, provider)

    if provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model)

    elif provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        kwargs: dict = {"model": model, "api_key": config.OPENAI_API_KEY}
        if config.OPENAI_API_BASE:
            kwargs["api_base"] = config.OPENAI_API_BASE
        return OpenAIEmbedding(**kwargs)

    elif provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(model_name=model, base_url=config.OLLAMA_BASE_URL)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# ---------------------------------------------------------------------------
# Backward-compatible wrapper so callers using LangChain method names
# (embed_query, embed_documents) continue to work without changes.
# ---------------------------------------------------------------------------
class _EmbeddingCompat:
    """Thin adapter that maps LangChain Embeddings method names to LlamaIndex."""

    def __init__(self, llama_embed):
        self._embed = llama_embed

    def embed_query(self, text: str) -> list[float]:
        return self._embed.get_query_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed.get_text_embedding_batch(texts)

    def __getattr__(self, name):
        # Forward anything else to the underlying LlamaIndex model
        return getattr(self._embed, name)


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
):
    """Backward-compatible alias for ``get_embedding_model``.

    Returns a wrapper that exposes ``embed_query()`` and ``embed_documents()``
    so that existing callers (ingest.py, etc.) work unchanged.
    """
    llama_embed = get_embedding_model(provider, model)
    return _EmbeddingCompat(llama_embed)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
_llm_cache: dict[str, object] = {}
_llm_lock = threading.Lock()


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs,
):
    """Return a LlamaIndex LLM instance for the given provider/model."""
    provider = (provider or config.LLM_PROVIDER).lower()
    model = model or config.LLM_MODEL
    temperature = temperature if temperature is not None else 0.3

    cache_key = f"{provider}|{model}|{temperature}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    with _llm_lock:
        # Double-check after acquiring lock
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]

        logger.info("Loading LLM: %s (%s, temp=%.2f)", model, provider, temperature)

        instance = None

        if provider == "ollama":
            from llama_index.llms.ollama import Ollama

            instance = Ollama(
                model=model,
                base_url=config.OLLAMA_BASE_URL,
                temperature=temperature,
                request_timeout=120.0,
            )

        elif provider == "openai":
            from llama_index.llms.openai import OpenAI

            openai_kwargs: dict = {
                "model": model,
                "temperature": temperature,
                "api_key": config.OPENAI_API_KEY,
            }
            base_url = config.OPENAI_API_BASE
            if base_url:
                openai_kwargs["api_base"] = base_url
            instance = OpenAI(**openai_kwargs)

        elif provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic

            instance = Anthropic(
                model=model,
                temperature=temperature,
                api_key=config.ANTHROPIC_API_KEY,
            )

        elif provider == "groq":
            from llama_index.llms.groq import Groq

            instance = Groq(
                model=model,
                temperature=temperature,
                api_key=config.GROQ_API_KEY,
            )

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        _llm_cache[cache_key] = instance
        return instance


# ---------------------------------------------------------------------------
# Helpers for UI dropdowns
# ---------------------------------------------------------------------------
def get_available_llm_models(provider: str) -> list[str]:
    """Return the list of known models for a given LLM provider."""
    return config.AVAILABLE_LLM_MODELS.get(provider.lower(), [])


def get_available_embedding_models(provider: str) -> list[str]:
    """Return the list of known models for a given embedding provider."""
    return config.AVAILABLE_EMBEDDING_MODELS.get(provider.lower(), [])
