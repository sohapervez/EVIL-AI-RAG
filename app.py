"""Streamlit chatbot UI for the Research Paper RAG system.

PDFs are loaded automatically from ``data/papers/``.  Drop new PDFs into
that folder and click "Re-sync Papers" in the sidebar (or restart the app)
to pick them up.  No manual upload or indexing buttons needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

import config
from core.providers import get_available_embedding_models, get_available_llm_models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_papers" not in st.session_state:
    st.session_state.indexed_papers = []
if "auto_indexed" not in st.session_state:
    st.session_state.auto_indexed = False


# ---------------------------------------------------------------------------
# Auto-indexing logic  (runs once on first load, or on "Re-sync")
# ---------------------------------------------------------------------------
def _get_already_indexed() -> set[str]:
    """Return the set of filenames already present in ChromaDB."""
    import chromadb

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        child_col = client.get_or_create_collection(
            name=config.CHROMA_CHILD_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        if child_col.count() == 0:
            return set()
        all_meta = child_col.get(include=["metadatas"])
        return {m.get("source", "") for m in all_meta["metadatas"]}
    except Exception:
        return set()


def _sync_papers(force: bool = False) -> str:
    """Index any new PDFs found in data/papers/ that aren't in ChromaDB yet.

    Returns a human-readable status message.
    """
    from core.providers import validate_provider_setup

    papers_dir = Path(config.DATA_DIR)
    papers_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        st.session_state.indexed_papers = []
        return "No PDFs found in `data/papers/`."

    already_indexed = _get_already_indexed()

    # Determine which files are new
    new_files = [f for f in pdf_files if f.name not in already_indexed]

    # Update the sidebar list with ALL papers (indexed + new about to be indexed)
    st.session_state.indexed_papers = sorted(f.name for f in pdf_files)

    if not new_files and not force:
        return f"{len(pdf_files)} paper(s) already indexed. Nothing new to process."

    if force:
        new_files = pdf_files  # re-process everything

    # Validate providers before attempting to embed
    emb_provider = st.session_state.get("embedding_provider", config.EMBEDDING_PROVIDER)
    emb_model = st.session_state.get("embedding_model", config.EMBEDDING_MODEL)
    llm_provider = st.session_state.get("llm_provider", config.LLM_PROVIDER)
    llm_model = st.session_state.get("llm_model", config.LLM_MODEL)

    all_ok, errors = validate_provider_setup(
        llm_provider=llm_provider,
        llm_model=llm_model,
        emb_provider=emb_provider,
        emb_model=emb_model,
    )
    if not all_ok:
        return "Provider check failed:\n" + "\n".join(f"• {e}" for e in errors)

    # Run ingestion (reuses ingest.py logic)
    from ingest import ingest_directory

    summary = ingest_directory(
        pdf_dir=papers_dir,
        clear=force,
    )

    st.session_state.indexed_papers = sorted(f.name for f in papers_dir.glob("*.pdf"))

    if summary["files"] == 0:
        return f"{len(pdf_files)} paper(s) already indexed. Nothing new to process."

    return (
        f"Indexed **{summary['files']}** new paper(s): "
        f"{summary['parents']} parent chunks, {summary['children']} child chunks."
    )


# ---------------------------------------------------------------------------
# Auto-index on first load
# ---------------------------------------------------------------------------
if not st.session_state.auto_indexed:
    with st.spinner("Scanning data/papers/ and indexing new PDFs..."):
        status_msg = _sync_papers()
    st.session_state.auto_indexed = True
    if "Provider check failed" in status_msg:
        st.error(status_msg)
    elif "Nothing new" not in status_msg and "No PDFs" not in status_msg:
        st.toast(status_msg, icon="✅")
else:
    # Even on reruns, refresh the paper list from ChromaDB
    already = _get_already_indexed()
    if already:
        st.session_state.indexed_papers = sorted(already)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Indexed Papers")
    st.caption(f"Drop PDFs into **`data/papers/`** and click Re-sync.")

    if st.session_state.indexed_papers:
        for paper in st.session_state.indexed_papers:
            st.text(f"• {paper}")
        st.caption(f"{len(st.session_state.indexed_papers)} paper(s) indexed")
    else:
        st.info("No papers found. Add PDFs to:\n\n`data/papers/`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-sync", use_container_width=True, help="Scan data/papers/ for new PDFs and index them"):
            with st.spinner("Syncing papers..."):
                status_msg = _sync_papers()
            if "Provider check failed" in status_msg:
                st.error(status_msg)
            else:
                st.success(status_msg)
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True, help="Delete the vector index (papers stay on disk)"):
            import chromadb

            client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            try:
                client.delete_collection(config.CHROMA_CHILD_COLLECTION)
            except Exception:
                pass
            try:
                client.delete_collection(config.CHROMA_PARENT_COLLECTION)
            except Exception:
                pass
            st.session_state.indexed_papers = []
            st.session_state.auto_indexed = False
            st.success("Index cleared. Click Re-sync to re-index.")
            st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # --- Model Settings ---
    st.divider()
    st.header("🤖 Model Settings")

    llm_provider = st.selectbox(
        "LLM Provider",
        options=list(config.AVAILABLE_LLM_MODELS.keys()),
        index=list(config.AVAILABLE_LLM_MODELS.keys()).index(config.LLM_PROVIDER)
        if config.LLM_PROVIDER in config.AVAILABLE_LLM_MODELS
        else 0,
        key="llm_provider",
    )
    llm_models = get_available_llm_models(llm_provider)
    llm_model = st.selectbox(
        "LLM Model",
        options=llm_models,
        index=llm_models.index(config.LLM_MODEL) if config.LLM_MODEL in llm_models else 0,
        key="llm_model",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, key="temperature")

    emb_provider = st.selectbox(
        "Embedding Provider",
        options=list(config.AVAILABLE_EMBEDDING_MODELS.keys()),
        index=list(config.AVAILABLE_EMBEDDING_MODELS.keys()).index(config.EMBEDDING_PROVIDER)
        if config.EMBEDDING_PROVIDER in config.AVAILABLE_EMBEDDING_MODELS
        else 0,
        key="embedding_provider",
    )
    emb_models = get_available_embedding_models(emb_provider)
    emb_model = st.selectbox(
        "Embedding Model",
        options=emb_models,
        index=emb_models.index(config.EMBEDDING_MODEL) if config.EMBEDDING_MODEL in emb_models else 0,
        key="embedding_model",
    )

    # --- Retrieval Settings ---
    st.divider()
    st.header("🔍 Retrieval Settings")
    st.slider("Top-K Results", 1, 20, config.TOP_K, 1, key="top_k")
    st.toggle("Hybrid Search (Vector + BM25)", value=config.USE_HYBRID_SEARCH, key="use_hybrid")
    if st.session_state.get("use_hybrid", config.USE_HYBRID_SEARCH):
        st.slider("BM25 Weight", 0.0, 1.0, config.BM25_WEIGHT, 0.05, key="bm25_weight")
    st.toggle("Re-ranking (Cross-Encoder)", value=config.USE_RERANKING, key="use_reranking")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("📚 Research Paper RAG Chatbot")
st.caption("Ask questions about the research papers in `data/papers/`.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📖 Sources ({len(msg['sources'])} references)"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}** — Page {src['page']}, "
                        f"Section: {src['section']} ({src['content_type']})"
                    )
                    st.caption(src["content"][:300] + ("..." if len(src["content"]) > 300 else ""))
                    st.divider()

# Chat input
if user_input := st.chat_input("Ask about your research papers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation history for the chain
    from core.rag_chain import ChatMessage, query_stream

    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
    ]

    # Stream response
    with st.chat_message("assistant"):
        try:
            token_gen, sources = query_stream(
                question=user_input,
                history=history,
                llm_provider=st.session_state.get("llm_provider", config.LLM_PROVIDER),
                llm_model=st.session_state.get("llm_model", config.LLM_MODEL),
                temperature=st.session_state.get("temperature", 0.3),
                top_k=st.session_state.get("top_k", config.TOP_K),
                use_hybrid=st.session_state.get("use_hybrid", config.USE_HYBRID_SEARCH),
                bm25_weight=st.session_state.get("bm25_weight", config.BM25_WEIGHT),
                use_reranking=st.session_state.get("use_reranking", config.USE_RERANKING),
                embedding_provider=st.session_state.get("embedding_provider", config.EMBEDDING_PROVIDER),
                embedding_model=st.session_state.get("embedding_model", config.EMBEDDING_MODEL),
            )

            full_response = st.write_stream(token_gen)

            # Show sources
            source_dicts = []
            if sources:
                source_dicts = [
                    {
                        "source": s.source,
                        "page": s.page,
                        "section": s.section,
                        "content_type": s.content_type,
                        "content": s.content,
                    }
                    for s in sources
                ]
                with st.expander(f"📖 Sources ({len(sources)} references)"):
                    for src in source_dicts:
                        st.markdown(
                            f"**{src['source']}** — Page {src['page']}, "
                            f"Section: {src['section']} ({src['content_type']})"
                        )
                        st.caption(src["content"][:300] + ("..." if len(src["content"]) > 300 else ""))
                        st.divider()

            # Save assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": source_dicts,
                }
            )

        except Exception as e:
            error_msg = f"Error: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
