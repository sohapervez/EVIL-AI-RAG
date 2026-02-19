"""RAG chain: retrieval + prompt construction + streaming generation + conversation memory."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from core.providers import get_llm
from core.retriever import RetrievedContext, retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class RAGResponse:
    """Holds the final answer and the sources used."""

    answer: str
    sources: list[RetrievedContext] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------
def _format_context(contexts: list[RetrievedContext]) -> str:
    """Format retrieved parent chunks into a string for the prompt."""
    parts: list[str] = []
    for i, ctx in enumerate(contexts, 1):
        header = f"[Source {i}: {ctx.source}, Page {ctx.page}, Section: {ctx.section}]"
        parts.append(f"{header}\n{ctx.content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: list[ChatMessage], max_turns: int = 5) -> str:
    """Format the last N turns of conversation history."""
    recent = history[-(max_turns * 2) :]  # each turn is user + assistant
    lines: list[str] = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Non-streaming query
# ---------------------------------------------------------------------------
def query(
    question: str,
    history: list[ChatMessage] | None = None,
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    temperature: float = 0.3,
    top_k: int | None = None,
    use_hybrid: bool | None = None,
    bm25_weight: float | None = None,
    use_reranking: bool | None = None,
    filter_source: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> RAGResponse:
    """Run the full RAG pipeline: retrieve context, build prompt, generate answer."""
    history = history or []

    # Step 1: Retrieve
    contexts = retrieve(
        query=question,
        top_k=top_k,
        use_hybrid=use_hybrid,
        bm25_weight=bm25_weight,
        use_reranking=use_reranking,
        filter_source=filter_source,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )

    # Step 2: Build prompt
    context_str = _format_context(contexts) if contexts else "No relevant context found."
    history_str = _format_history(history) if history else "No prior conversation."

    messages = [
        SystemMessage(content=config.SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context from research papers:\n{context_str}\n\n"
                f"Conversation history:\n{history_str}\n\n"
                f"Question: {question}"
            )
        ),
    ]

    # Step 3: Generate
    llm = get_llm(provider=llm_provider, model=llm_model, temperature=temperature, streaming=False)
    response = llm.invoke(messages)

    return RAGResponse(answer=response.content, sources=contexts)


# ---------------------------------------------------------------------------
# Streaming query (for Streamlit)
# ---------------------------------------------------------------------------
def query_stream(
    question: str,
    history: list[ChatMessage] | None = None,
    *,
    system_prompt: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    top_k: int | None = None,
    use_hybrid: bool | None = None,
    bm25_weight: float | None = None,
    use_reranking: bool | None = None,
    filter_source: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> tuple[Generator[str, None, None], list[RetrievedContext]]:
    """Stream the RAG response token-by-token.

    Returns a (token_generator, sources) tuple.  The caller iterates the
    generator to collect streamed tokens while having immediate access to
    the source list for displaying citations.

    Parameters
    ----------
    system_prompt : str | None
        Optional override for the system prompt. If *None*, falls back to
        ``config.SYSTEM_PROMPT``.
    """
    history = history or []

    # Step 1: Retrieve
    contexts = retrieve(
        query=question,
        top_k=top_k,
        use_hybrid=use_hybrid,
        bm25_weight=bm25_weight,
        use_reranking=use_reranking,
        filter_source=filter_source,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )

    # Step 2: Build prompt
    context_str = _format_context(contexts) if contexts else "No relevant context found."
    history_str = _format_history(history) if history else "No prior conversation."

    messages = [
        SystemMessage(content=system_prompt or config.SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context from research papers:\n{context_str}\n\n"
                f"Conversation history:\n{history_str}\n\n"
                f"Question: {question}"
            )
        ),
    ]

    # Step 3: Stream
    llm_kwargs = {}
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens
    llm = get_llm(provider=llm_provider, model=llm_model, temperature=temperature, streaming=True, **llm_kwargs)

    def _token_generator() -> Generator[str, None, None]:
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    return _token_generator(), contexts
