"""FastAPI backend for the EVIL-AI RAG chatbot.

Endpoints
---------
Public:
    POST /api/v1/chat         — SSE streaming chat
    GET  /api/v1/papers       — list indexed papers
    GET  /api/v1/health       — health check

Protected (Bearer token):
    POST   /api/v1/papers     — upload & ingest a PDF
    DELETE /api/v1/papers/{fn} — remove a paper
    POST   /api/v1/reindex    — full re-index
    GET    /api/v1/analytics   — summary stats
    GET    /api/v1/analytics/questions — paginated log
    GET    /api/v1/analytics/export   — CSV download
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import config
from core.analytics import AnalyticsLogger
from core.paper_metadata import (
    enhance_query_with_paper_info,
    extract_paper_titles,
    is_general_question,
    is_paper_specific_question,
)
from core.rag_chain import ChatMessage, query_stream
from ingest import clear_collections, delete_paper_chunks, ingest_directory, ingest_single_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="EVIL-AI RAG API", version="1.0.0")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded", status_code=429)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://evil-ai.eu",
        "https://www.evil-ai.eu",
        "https://evil-ai-rag.rahtiapp.fi",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8501",
    ],
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
)

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
_static_dir = Path(__file__).resolve().parent / "chat-widget-wordpress" / "widget"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
analytics = AnalyticsLogger()

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def verify_token(authorization: str = Header(...)):
    if not config.API_SECRET_KEY:
        raise HTTPException(503, "API key not configured")
    if authorization != f"Bearer {config.API_SECRET_KEY}":
        raise HTTPException(401, "Invalid API key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    session_id: str = ""
    history: list[ChatHistoryItem] = []


# ---------------------------------------------------------------------------
# System prompt builder (mirrors app1.py logic)
# ---------------------------------------------------------------------------


def _build_system_prompt(paper_info: str) -> str:
    return f"""You are a research assistant for the EVIL-AI project. \
Your role is to answer questions about the research papers by synthesizing information from the provided context.

The EVIL-AI project has these research papers:
{paper_info}

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
  * The user explicitly asks about multiple papers, OR
  * The user asks about overall EVIL-AI research, OR
  * The question is clearly comparative.
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
  "Amanda Aline F.C. Vicenzi, Jose Siqueira de Cerqueira, Pekka Abrahamsson, Edna Dias Canedo"

FAILURE CONDITION:
- Only say "I don't have that information in the provided context"
  if the context truly contains no relevant information at all."""


# ---------------------------------------------------------------------------
# PUBLIC: Health
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# PUBLIC: Papers list
# ---------------------------------------------------------------------------


@app.get("/api/v1/papers")
async def list_papers():
    paper_info = extract_paper_titles()
    return {"papers": paper_info}


# ---------------------------------------------------------------------------
# PUBLIC: Chat (SSE streaming)
# ---------------------------------------------------------------------------


@app.post("/api/v1/chat")
@limiter.limit(config.RATE_LIMIT)
async def chat(body: ChatRequest, request: Request):
    start = time.time()
    ip_raw = get_remote_address(request) or ""
    ip_hash = hashlib.sha256(ip_raw.encode()).hexdigest()[:16]

    # Paper-aware query enhancement (mirrors app1.py lines 762-812)
    paper_info = extract_paper_titles()
    system_prompt = _build_system_prompt(paper_info)

    # 1. Detect specific paper
    target_source = is_paper_specific_question(body.question, paper_info)

    # 2. Enhance query
    enhanced_query, general_hint = enhance_query_with_paper_info(body.question, paper_info)

    # 3. Append filename if needed
    if target_source and target_source not in enhanced_query:
        enhanced_query = f"{enhanced_query} {target_source}"

    # 4. Determine general vs specific
    is_general = general_hint or is_general_question(body.question) or target_source is None

    # 5. Filter & retrieval params
    filter_source = None if is_general else target_source
    top_k = max(12, config.TOP_K * 3) if is_general else config.TOP_K
    max_tokens = 2048 if is_general else 1024

    # Build history
    history = [ChatMessage(role=h.role, content=h.content) for h in body.history]

    # Call RAG pipeline (retrieval happens here, offloaded to thread)
    token_gen, sources = await asyncio.to_thread(
        query_stream,
        question=enhanced_query,
        history=history,
        system_prompt=system_prompt,
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
        temperature=0.3,
        max_tokens=max_tokens,
        top_k=top_k,
        filter_source=filter_source,
        use_hybrid=config.USE_HYBRID_SEARCH,
        bm25_weight=config.BM25_WEIGHT,
        use_reranking=config.USE_RERANKING,
        embedding_provider=config.EMBEDDING_PROVIDER,
        embedding_model=config.EMBEDDING_MODEL,
    )

    source_data = [
        {
            "source": s.source,
            "page": s.page,
            "section": s.section,
            "content_type": s.content_type,
            "content": s.content[:300],
        }
        for s in sources
    ]

    papers_cited = list({s.source for s in sources})
    query_type = "general" if is_general else "specific"

    async def event_stream():
        full_answer = []
        error_text = ""
        try:
            for token in token_gen:
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0)

            yield f"data: {json.dumps({'type': 'sources', 'data': source_data})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            error_text = str(exc)
            yield f"data: {json.dumps({'type': 'error', 'message': error_text})}\n\n"
        finally:
            # Log analytics
            answer_text = "".join(full_answer)
            latency_ms = int((time.time() - start) * 1000)
            try:
                analytics.log_chat(
                    session_id=body.session_id,
                    question=body.question,
                    answer_preview=answer_text[:200],
                    papers_cited=papers_cited,
                    chunks_retrieved=len(sources),
                    response_length=len(answer_text),
                    latency_ms=latency_ms,
                    query_type=query_type,
                    error=error_text,
                    ip_hash=ip_hash,
                )
            except Exception as log_exc:
                logger.warning("Analytics logging failed: %s", log_exc)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# PROTECTED: Upload paper
# ---------------------------------------------------------------------------


@app.post("/api/v1/papers", dependencies=[Depends(verify_token)])
async def upload_paper(file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename)
    if not safe_name or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    papers_dir = Path(config.DATA_DIR)
    papers_dir.mkdir(parents=True, exist_ok=True)
    dest = papers_dir / safe_name
    if not dest.resolve().is_relative_to(papers_dir.resolve()):
        raise HTTPException(400, "Invalid filename")

    content = await file.read()
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "File too large (max 50 MB)")
    dest.write_bytes(content)

    try:
        result = await asyncio.to_thread(ingest_single_pdf, dest)
    except Exception as exc:
        raise HTTPException(500, f"Ingestion failed: {exc}")

    return {"filename": safe_name, "ingestion": result}


# ---------------------------------------------------------------------------
# PROTECTED: Delete paper
# ---------------------------------------------------------------------------


@app.delete("/api/v1/papers/{filename}", dependencies=[Depends(verify_token)])
async def remove_paper(filename: str):
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename:
        raise HTTPException(400, "Invalid filename")
    pdf_path = Path(config.DATA_DIR) / safe_name
    if not pdf_path.resolve().is_relative_to(Path(config.DATA_DIR).resolve()):
        raise HTTPException(400, "Invalid filename")

    # Remove from vector store
    result = delete_paper_chunks(safe_name)

    # Remove PDF file
    if pdf_path.exists():
        pdf_path.unlink()

    return {"filename": safe_name, "deleted": result}


# ---------------------------------------------------------------------------
# PROTECTED: Reindex
# ---------------------------------------------------------------------------


@app.post("/api/v1/reindex", dependencies=[Depends(verify_token)])
async def reindex():
    from core.retriever import _get_chroma_client

    client = _get_chroma_client()
    await asyncio.to_thread(clear_collections, client)
    result = await asyncio.to_thread(ingest_directory, pdf_dir=config.DATA_DIR, clear=False)
    return {"reindex": result}


# ---------------------------------------------------------------------------
# PROTECTED: Analytics
# ---------------------------------------------------------------------------


@app.get("/api/v1/analytics", dependencies=[Depends(verify_token)])
async def analytics_summary(days: int = 30):
    return analytics.get_summary(days=days)


@app.get("/api/v1/analytics/questions", dependencies=[Depends(verify_token)])
async def analytics_questions(page: int = 1, per_page: int = 20):
    return analytics.get_questions(page=page, per_page=per_page)


@app.get("/api/v1/analytics/export", dependencies=[Depends(verify_token)])
async def analytics_export():
    csv_data = analytics.export_csv()
    return PlainTextResponse(
        csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analytics.csv"},
    )
