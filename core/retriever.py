"""Hybrid retrieval pipeline: vector + BM25 ensemble, cross-encoder re-ranking, parent expansion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from langchain_core.documents import Document

import config
from core.providers import get_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure returned to the RAG chain
# ---------------------------------------------------------------------------
@dataclass
class RetrievedContext:
    """A parent chunk with metadata, ready to be injected into the prompt."""

    content: str
    source: str
    page: int
    section: str
    content_type: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------
def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def _get_child_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(
        name=config.CHROMA_CHILD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _get_parent_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(name=config.CHROMA_PARENT_COLLECTION)


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------
def _bm25_search(
    query: str,
    child_collection,
    top_n: int,
    where_filter: dict | None = None,
) -> list[tuple[str, float]]:
    """Return (child_id, score) pairs using BM25 over child chunks (optionally filtered)."""
    from rank_bm25 import BM25Okapi

    # Fetch children only from the filtered subset (if provided)
    if where_filter:
        all_children = child_collection.get(where=where_filter, include=["documents", "metadatas"])
    else:
        all_children = child_collection.get(include=["documents", "metadatas"])
    
    if not all_children["documents"]:
        return []

    ids = all_children["ids"]
    docs = all_children["documents"]

    tokenised = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenised)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Pair ids with scores and sort
    scored = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking
# ---------------------------------------------------------------------------
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def _rerank(query: str, docs: list[dict], top_k: int) -> list[dict]:
    """Re-rank documents using a cross-encoder model.

    Each doc dict must have a 'content' key.
    """
    if not docs:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc["content"]) for doc in docs]
    scores = reranker.predict(pairs)

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k: Optional[int] = None,
    use_hybrid: Optional[bool] = None,
    bm25_weight: Optional[float] = None,
    use_reranking: Optional[bool] = None,
    filter_source: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> list[RetrievedContext]:
    """Run the full retrieval pipeline and return parent-expanded context.

    Steps:
      A. Dual retrieval on child chunks (vector + BM25)
      B. Cross-encoder re-ranking
      C. Parent expansion
      D. Deduplication
    """
    top_k = top_k if top_k is not None else config.TOP_K
    use_hybrid = use_hybrid if use_hybrid is not None else config.USE_HYBRID_SEARCH
    bm25_weight = bm25_weight if bm25_weight is not None else config.BM25_WEIGHT
    use_reranking = use_reranking if use_reranking is not None else config.USE_RERANKING

    client = _get_chroma_client()
    child_col = _get_child_collection(client)
    parent_col = _get_parent_collection(client)

    # Check that we have data
    if child_col.count() == 0:
        logger.warning("Child collection is empty — no documents indexed.")
        return []

    candidate_n = top_k * 3  # over-fetch for re-ranking

    # ----- Stage A: Vector search on children -----
    embeddings = get_embeddings(embedding_provider, embedding_model)
    query_embedding = embeddings.embed_query(query)

    where_filter = None
    if filter_source:
        where_filter = {"source": filter_source}

    vector_results = child_col.query(
        query_embeddings=[query_embedding],
        n_results=candidate_n,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Build candidate dict: child_id -> {content, metadata, score}
    candidates: dict[str, dict] = {}

    if vector_results["ids"] and vector_results["ids"][0]:
        for cid, doc, meta, dist in zip(
            vector_results["ids"][0],
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0],
        ):
            # ChromaDB cosine distance: lower = more similar; convert to similarity
            score = 1.0 - dist
            candidates[cid] = {
                "child_id": cid,
                "content": doc,
                "metadata": meta,
                "vector_score": score,
                "bm25_score": 0.0,
            }

    # ----- Stage A (cont.): BM25 search -----
    if use_hybrid:
        bm25_results = _bm25_search(query, child_col, candidate_n, where_filter=where_filter)
        max_bm25 = bm25_results[0][1] if bm25_results else 1.0
        max_bm25 = max(max_bm25, 1e-9)  # avoid division by zero

        for cid, score in bm25_results:
            normalised = score / max_bm25  # normalise to 0-1
            if cid in candidates:
                candidates[cid]["bm25_score"] = normalised
            else:
                # Fetch the document content for this child
                child_data = child_col.get(ids=[cid], include=["documents", "metadatas"])
                if child_data["documents"]:
                    candidates[cid] = {
                        "child_id": cid,
                        "content": child_data["documents"][0],
                        "metadata": child_data["metadatas"][0],
                        "vector_score": 0.0,
                        "bm25_score": normalised,
                    }

    # Combine scores: weighted fusion
    vector_weight = 1.0 - bm25_weight if use_hybrid else 1.0
    for c in candidates.values():
        c["combined_score"] = (
            vector_weight * c.get("vector_score", 0)
            + (bm25_weight if use_hybrid else 0) * c.get("bm25_score", 0)
        )

    # Sort by combined score
    sorted_candidates = sorted(candidates.values(), key=lambda x: x["combined_score"], reverse=True)

    # ----- Stage B: Re-ranking -----
    if use_reranking and sorted_candidates:
        try:
            sorted_candidates = _rerank(query, sorted_candidates, top_k)
        except Exception as e:
            logger.warning("Re-ranking failed, using combined score order: %s", e)
            sorted_candidates = sorted_candidates[:top_k]
    else:
        sorted_candidates = sorted_candidates[:top_k]

    # ----- Stage C: Parent expansion -----
    seen_parents: set[str] = set()
    results: list[RetrievedContext] = []

    for cand in sorted_candidates:
        parent_id = cand["metadata"].get("parent_id", "")
        if not parent_id or parent_id in seen_parents:
            # If no parent or already seen, use the child content directly
            if parent_id in seen_parents:
                continue
            results.append(
                RetrievedContext(
                    content=cand["content"],
                    source=cand["metadata"].get("source", ""),
                    page=cand["metadata"].get("page", 0),
                    section=cand["metadata"].get("section", ""),
                    content_type=cand["metadata"].get("content_type", "text"),
                    score=cand.get("rerank_score", cand.get("combined_score", 0)),
                )
            )
            continue

        seen_parents.add(parent_id)

        # Fetch parent chunk
        try:
            parent_data = parent_col.get(ids=[parent_id], include=["documents", "metadatas"])
            if parent_data["documents"]:
                pmeta = parent_data["metadatas"][0] if parent_data["metadatas"] else {}
                results.append(
                    RetrievedContext(
                        content=parent_data["documents"][0],
                        source=pmeta.get("source", cand["metadata"].get("source", "")),
                        page=pmeta.get("page", cand["metadata"].get("page", 0)),
                        section=pmeta.get("section", cand["metadata"].get("section", "")),
                        content_type=pmeta.get("content_type", "text"),
                        score=cand.get("rerank_score", cand.get("combined_score", 0)),
                    )
                )
            else:
                # Fallback to child content
                results.append(
                    RetrievedContext(
                        content=cand["content"],
                        source=cand["metadata"].get("source", ""),
                        page=cand["metadata"].get("page", 0),
                        section=cand["metadata"].get("section", ""),
                        content_type=cand["metadata"].get("content_type", "text"),
                        score=cand.get("rerank_score", cand.get("combined_score", 0)),
                    )
                )
        except Exception as e:
            logger.warning("Parent lookup failed for %s: %s", parent_id, e)
            results.append(
                RetrievedContext(
                    content=cand["content"],
                    source=cand["metadata"].get("source", ""),
                    page=cand["metadata"].get("page", 0),
                    section=cand["metadata"].get("section", ""),
                    content_type=cand["metadata"].get("content_type", "text"),
                    score=cand.get("rerank_score", cand.get("combined_score", 0)),
                )
            )

    logger.info("Retrieved %d parent contexts for query", len(results))
    return results
