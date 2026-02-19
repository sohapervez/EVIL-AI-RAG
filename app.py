"""EVIL-AI Research Paper RAG Chatbot.

A straightforward Q&A chatbot where users ask questions and get answers
based on research papers (PDFs) stored in ``data/papers/``.
The user drives the conversation — no follow-up questions are asked.

Usage:
    streamlit run app.py
"""

# --- 1. Setup & Imports ---
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

import config
from core.paper_metadata import (
    extract_paper_titles,
    enhance_query_with_paper_info,
    is_general_question,
    is_paper_specific_question,
)
from core.rag_chain import ChatMessage, query_stream

logger = logging.getLogger(__name__)

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="EVIL-AI Research Q&A",
    page_icon="📚",
)

# --- 3. Session State Defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "auto_indexed" not in st.session_state:
    st.session_state.auto_indexed = False
if "paper_info" not in st.session_state:
    st.session_state.paper_info = ""


# --- 4. Auto-Index PDFs on First Load ---
def _sync_papers() -> str:
    """Index any new PDFs found in data/papers/ into ChromaDB."""
    import chromadb
    from ingest import ingest_directory

    papers_dir = Path(config.DATA_DIR)
    papers_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        return "No PDFs found in `data/papers/`."

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        child_col = client.get_or_create_collection(
            name=config.CHROMA_CHILD_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        if child_col.count() > 0:
            all_meta = child_col.get(include=["metadatas"])
            already = {m.get("source", "") for m in all_meta["metadatas"]}
        else:
            already = set()
    except Exception:
        already = set()

    new_files = [f for f in pdf_files if f.name not in already]
    if not new_files:
        return f"{len(pdf_files)} paper(s) already indexed."

    summary = ingest_directory(pdf_dir=papers_dir, clear=False)

    if summary["files"] == 0:
        return f"{len(pdf_files)} paper(s) already indexed."

    return (
        f"Indexed **{summary['files']}** new paper(s): "
        f"{summary['parents']} parent chunks, {summary['children']} child chunks."
    )


if not st.session_state.auto_indexed:
    with st.spinner("Indexing research papers..."):
        status_msg = _sync_papers()
    st.session_state.paper_info = extract_paper_titles()
    st.session_state.auto_indexed = True
    if "No PDFs" in status_msg:
        st.warning(status_msg)
    elif "already indexed" not in status_msg:
        st.toast(status_msg, icon="✅")
    
    # Check for failed papers and show warning with specific file names
    if "[FAILED TO INDEX]" in st.session_state.paper_info:
        # Extract failed file names from paper_info
        failed_files = [
            line.split("(file: ")[1].split(")")[0]
            for line in st.session_state.paper_info.split("\n")
            if "[FAILED TO INDEX]" in line
        ]
        failed_list = ", ".join(failed_files)
        st.warning(
            f"⚠️ **Some papers failed to index** (PDF parsing errors):\n\n"
            f"**Failed files:** {failed_list}\n\n"
            f"These papers are listed but won't be searchable. "
            f"The PDFs may be corrupted or have unsupported formatting. "
            f"Try re-saving the PDFs or converting them to a standard format."
        )

if not st.session_state.paper_info:
    st.session_state.paper_info = extract_paper_titles()
    if "[FAILED TO INDEX]" in st.session_state.paper_info:
        failed_files = [
            line.split("(file: ")[1].split(")")[0]
            for line in st.session_state.paper_info.split("\n")
            if "[FAILED TO INDEX]" in line
        ]
        failed_list = ", ".join(failed_files)
        st.warning(
            f"⚠️ **Some papers failed to index** (PDF parsing errors):\n\n"
            f"**Failed files:** {failed_list}\n\n"
            f"These papers are listed but won't be searchable."
        )


# --- 6. System Prompt ---
SYSTEM_PROMPT = f"""You are a research assistant for the EVIL-AI project. \
Your role is to answer questions about the research papers by synthesizing information from the provided context.

The EVIL-AI project has these research papers:
{st.session_state.paper_info}

IMPORTANT INSTRUCTIONS:

GENERAL ANSWERING RULES:
- Answer ONLY using the provided context from the research papers.
- Do NOT use outside knowledge.
- If the context does not contain enough information, clearly say so.
- Cite sources as [Source: filename, Page X].

PAPER-SPECIFIC QUESTIONS:
- If the question clearly refers to a specific paper (by title, filename, or paper number),
  you MUST restrict your answer to that paper only.
- Do NOT include information from other papers unless the user explicitly asks to compare or combine them.
- If retrieved context includes chunks from unrelated papers, ignore them.

GENERAL OR CROSS-PAPER QUESTIONS:
- Only synthesize across multiple papers when:
  • The user explicitly asks about multiple papers, OR
  • The user asks about overall EVIL-AI research, OR
  • The question is clearly comparative.
- When synthesizing across papers, include only papers for which relevant context was retrieved.
- Do NOT force inclusion of every paper unless the question explicitly requires all papers.

SUMMARIES:
- When asked for a summary, abstract, or overview of a specific paper,
  synthesize key points from that paper only (e.g., introduction, methodology, findings, conclusions).
- When asked for an overall project summary,
  synthesize across relevant papers and clearly indicate which findings come from which paper.

CONTEXT DISCIPLINE:
- If multiple filenames appear in the retrieved context, do NOT automatically use all of them.
- Use only the context that directly answers the question.
- Ignore context that is unrelated to the question, even if retrieved.

ACCURACY PRIORITY:
- Prefer precision over breadth.
- Do not speculate.
- Do not infer beyond what is explicitly supported by the context.

CRITICAL AUTHOR RULES:
- NEVER mention author names in summaries, discussions, or content answers
  UNLESS the user explicitly asks about authors.
- When discussing paper content, refer ONLY to the exact paper titles listed above.
- Author names should appear ONLY when the question specifically asks for them.
- If author information appears in the retrieved context, ignore it unless the question is about authors.

PAPER TITLE RULES:
- When referring to papers, use ONLY the exact titles listed above.
- Never invent or modify titles.
- When asked to list paper titles, list ALL papers shown above,
  including those marked "[FAILED TO INDEX]".
- When listing titles, do NOT include authors or affiliations.

AUTHOR LISTING RULES:
- When asked about authors, use ONLY the author information provided in the paper list above.
- Do NOT include ORCID IDs, brackets, superscripts, or digits.
- Example correct format:
  "Amanda Aline F.C. Vicenzi, José Siqueira de Cerqueira, Pekka Abrahamsson, Edna Dias Canedo"

FAILURE CONDITION:
- Only say "I don't have that information in the provided context"
  if the context truly contains no relevant information at all."""


# --- 7. Title ---
st.title("📚 EVIL-AI Research Q&A")

# --- 8. Display Chat History ---
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
                    st.caption(
                        src["content"][:300]
                        + ("..." if len(src["content"]) > 300 else "")
                    )
                    st.divider()

# --- 9. Chat Input & Response ---
if user_input := st.chat_input("Ask a question about the research papers..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation history
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
    ]

    # Stream the answer
    with st.chat_message("assistant"):
        try:
            # First, check if user is asking about a specific paper (use original query)
            target_source = is_paper_specific_question(
                user_input,  # Use original query for title detection
                st.session_state.paper_info
            )
            
            # Then enhance query: map paper numbers to titles/filenames
            enhanced_query, general_hint = enhance_query_with_paper_info(
                user_input,
                st.session_state.paper_info
            )
            
            # If we detected a specific paper but enhancement didn't add filename, add it now
            if target_source and target_source not in enhanced_query:
                enhanced_query = f"{enhanced_query} {target_source}"

            # If no clear paper specified → treat as general query
            is_general_query = (
                general_hint
                or is_general_question(enhanced_query)
                or (target_source is None)
            )

            # Apply filter only for clearly paper-specific queries
            filter_source = None if is_general_query else target_source

            # Set retrieval breadth based on mode
            if is_general_query:
                top_k_value = max(12, config.TOP_K * 3)
                max_tokens_value = 2048
            else:
                top_k_value = config.TOP_K
                max_tokens_value = 1024

            # Call RAG pipeline
            token_gen, sources = query_stream(
                question=enhanced_query,
                history=history,
                system_prompt=SYSTEM_PROMPT,
                llm_provider=config.LLM_PROVIDER,
                llm_model=config.LLM_MODEL,
                temperature=0.3,
                max_tokens=max_tokens_value,
                top_k=top_k_value,
                filter_source=filter_source,
                use_hybrid=config.USE_HYBRID_SEARCH,
                bm25_weight=config.BM25_WEIGHT,
                use_reranking=config.USE_RERANKING,
                embedding_provider=config.EMBEDDING_PROVIDER,
                embedding_model=config.EMBEDDING_MODEL,
            )

            full_response = st.write_stream(token_gen)

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
                        st.caption(
                            src["content"][:300]
                            + ("..." if len(src["content"]) > 300 else "")
                        )
                        st.divider()

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
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

