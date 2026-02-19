"""CLI script to batch-ingest PDFs from data/papers/ into ChromaDB.

Usage:
    python ingest.py                        # ingest all PDFs in data/papers/
    python ingest.py --dir /path/to/pdfs    # ingest from a custom directory
    python ingest.py --clear                # wipe the vector store first
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb

import config
from core.chunker import chunk_elements
from core.pdf_parser import parse_pdf
from core.providers import get_embeddings
from core.retriever import _get_chroma_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def clear_collections(client: chromadb.ClientAPI) -> None:
    """Delete both parent and child collections."""
    for name in (config.CHROMA_CHILD_COLLECTION, config.CHROMA_PARENT_COLLECTION):
        try:
            client.delete_collection(name)
            logger.info("Deleted collection: %s", name)
        except Exception:
            pass


def _process_single_pdf(
    pdf_path: Path,
    use_vision: bool | None,
    parent_chunk_size: int | None,
    child_chunk_size: int | None,
    child_chunk_overlap: int | None,
    use_semantic: bool | None,
    use_parent_child: bool | None,
) -> tuple[str, list, list] | None:
    """Process a single PDF: parse and chunk.
    
    Returns (filename, parents, children) or None if failed.
    This function is designed to be called in parallel.
    """
    filename = pdf_path.name
    try:
        # Parse
        elements = parse_pdf(pdf_path, use_vision=use_vision)
        if not elements:
            logger.warning("No content extracted from %s", filename)
            return None
        
        # Chunk
        parents, children = chunk_elements(
            elements,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            use_semantic=use_semantic,
            use_parent_child=use_parent_child,
        )
        
        return (filename, parents, children)
    except Exception as e:
        logger.error("Failed to process %s: %s", filename, e)
        return None


def ingest_directory(
    pdf_dir: str | Path,
    clear: bool = False,
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    child_chunk_overlap: int | None = None,
    use_semantic: bool | None = None,
    use_vision: bool | None = None,
    use_parent_child: bool | None = None,
) -> dict:
    """Ingest all PDFs from a directory into ChromaDB.

    Returns a summary dict with counts.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return {"files": 0, "parents": 0, "children": 0}

    logger.info("Found %d PDF file(s) in %s", len(pdf_files), pdf_dir)

    # ChromaDB client and collections
    client = _get_chroma_client()

    if clear:
        clear_collections(client)

    child_col = client.get_or_create_collection(
        name=config.CHROMA_CHILD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    parent_col = client.get_or_create_collection(
        name=config.CHROMA_PARENT_COLLECTION,
    )

    # Embedding model
    embeddings = get_embeddings()
    
    # Chunking mode
    use_parent_child_chunking = use_parent_child if use_parent_child is not None else config.USE_PARENT_CHILD_CHUNKING

    # Track already-indexed sources to skip duplicates
    existing_sources: set[str] = set()
    try:
        existing_meta = child_col.get(include=["metadatas"])
        existing_sources = {m.get("source", "") for m in existing_meta["metadatas"]}
    except Exception:
        pass

    # Filter out already-indexed files
    pdfs_to_process = [
        pdf_path
        for pdf_path in pdf_files
        if clear or pdf_path.name not in existing_sources
    ]
    
    if not pdfs_to_process:
        logger.info("All files already indexed. Use --clear to re-index.")
        return {"files": 0, "parents": 0, "children": 0}

    chunking_mode = "parent-child" if use_parent_child_chunking else "simple"
    logger.info("Processing %d PDF file(s) (parallel parsing enabled, chunking: %s)", len(pdfs_to_process), chunking_mode)

    # Process PDFs in parallel (parsing and chunking are CPU-bound)
    # Use ThreadPoolExecutor for I/O-bound PDF parsing
    # Increase workers for better parallelization (up to CPU count)
    import os as os_module
    cpu_count = os_module.cpu_count() or 4
    max_workers = min(max(2, cpu_count - 1), len(pdfs_to_process))  # Use CPU count - 1, min 2
    processed_results: list[tuple[str, list, list]] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(
                _process_single_pdf,
                pdf_path,
                use_vision,
                parent_chunk_size,
                child_chunk_size,
                child_chunk_overlap,
                use_semantic,
                use_parent_child_chunking,
            ): pdf_path
            for pdf_path in pdfs_to_process
        }
        
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    filename, parents, children = result
                    processed_results.append((filename, parents, children))
                    logger.info(
                        "Parsed %s: %d parent chunks, %d child chunks",
                        filename,
                        len(parents),
                        len(children),
                    )
            except Exception as e:
                logger.error("Error processing %s: %s", pdf_path.name, e)

    # Now store all results sequentially (ChromaDB operations)
    total_parents = 0
    total_children = 0
    files_processed = 0
    
    # Collect all children for batch embedding
    all_children: list = []
    all_parents: list = []
    
    for filename, parents, children in processed_results:
        all_parents.extend(parents)
        all_children.extend(children)
        files_processed += 1

    # Store parents in batches
    if all_parents:
        parent_col.upsert(
            ids=[p.parent_id for p in all_parents],
            documents=[p.content for p in all_parents],
            metadatas=[
                {
                    "source": p.source,
                    "page": p.page,
                    "section": p.section,
                    "content_type": p.content_type,
                }
                for p in all_parents
            ],
        )
        total_parents = len(all_parents)

    # Store children with embeddings in batches
    if all_children:
        # Embed in batches (increased batch size for speed)
        batch_size = 200  # Increased from 100 for faster processing
        child_texts = [c.content for c in all_children]
        child_metas = [
            {
                "parent_id": c.parent_id,
                "source": c.source,
                "page": c.page,
                "section": c.section,
                "content_type": c.content_type,
                "chunk_index": c.chunk_index,
            }
            for c in all_children
        ]
        child_ids = [c.child_id for c in all_children]

        logger.info("Generating embeddings for %d child chunks in batches of %d...", len(all_children), batch_size)
        for i in range(0, len(all_children), batch_size):
            batch_texts = child_texts[i : i + batch_size]
            batch_ids = child_ids[i : i + batch_size]
            batch_metas = child_metas[i : i + batch_size]

            batch_embeddings = embeddings.embed_documents(batch_texts)

            child_col.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            logger.info("  Embedded batch %d/%d", (i // batch_size) + 1, (len(all_children) + batch_size - 1) // batch_size)

        total_children = len(all_children)

    summary = {
        "files": files_processed,
        "parents": total_parents,
        "children": total_children,
    }

    logger.info(
        "Ingestion complete: %d files, %d parents, %d children",
        summary["files"],
        summary["parents"],
        summary["children"],
    )
    return summary


def delete_paper_chunks(filename: str) -> dict:
    """Remove all chunks for a specific PDF from both collections."""
    client = _get_chroma_client()
    child_col = client.get_or_create_collection(
        name=config.CHROMA_CHILD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    parent_col = client.get_or_create_collection(name=config.CHROMA_PARENT_COLLECTION)

    child_data = child_col.get(where={"source": filename}, include=["metadatas"])
    child_ids = child_data["ids"]
    parent_ids = list({m["parent_id"] for m in child_data["metadatas"] if "parent_id" in m})

    if child_ids:
        child_col.delete(ids=child_ids)
    if parent_ids:
        parent_col.delete(ids=parent_ids)

    return {"deleted_children": len(child_ids), "deleted_parents": len(parent_ids)}


def ingest_single_pdf(pdf_path: Path) -> dict:
    """Save and ingest a single PDF file."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = _get_chroma_client()
    child_col = client.get_or_create_collection(
        name=config.CHROMA_CHILD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    parent_col = client.get_or_create_collection(name=config.CHROMA_PARENT_COLLECTION)

    use_parent_child = config.USE_PARENT_CHILD_CHUNKING
    result = _process_single_pdf(
        pdf_path,
        use_vision=None,
        parent_chunk_size=None,
        child_chunk_size=None,
        child_chunk_overlap=None,
        use_semantic=None,
        use_parent_child=use_parent_child,
    )

    if result is None:
        return {"files": 0, "parents": 0, "children": 0}

    filename, parents, children = result
    embeddings = get_embeddings()

    if parents:
        parent_col.upsert(
            ids=[p.parent_id for p in parents],
            documents=[p.content for p in parents],
            metadatas=[
                {
                    "source": p.source,
                    "page": p.page,
                    "section": p.section,
                    "content_type": p.content_type,
                }
                for p in parents
            ],
        )

    if children:
        batch_size = 200
        child_texts = [c.content for c in children]
        child_metas = [
            {
                "parent_id": c.parent_id,
                "source": c.source,
                "page": c.page,
                "section": c.section,
                "content_type": c.content_type,
                "chunk_index": c.chunk_index,
            }
            for c in children
        ]
        child_ids = [c.child_id for c in children]

        for i in range(0, len(children), batch_size):
            batch_texts = child_texts[i : i + batch_size]
            batch_ids = child_ids[i : i + batch_size]
            batch_metas = child_metas[i : i + batch_size]
            batch_embeddings = embeddings.embed_documents(batch_texts)
            child_col.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )

    return {"files": 1, "parents": len(parents), "children": len(children)}


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF research papers into ChromaDB")
    parser.add_argument(
        "--dir",
        type=str,
        default=config.DATA_DIR,
        help=f"Directory containing PDF files (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collections before ingesting",
    )
    parser.add_argument("--parent-chunk-size", type=int, default=None)
    parser.add_argument("--child-chunk-size", type=int, default=None)
    parser.add_argument("--child-chunk-overlap", type=int, default=None)
    parser.add_argument("--semantic", action="store_true", default=None)
    parser.add_argument("--vision", action="store_true", default=None)
    parser.add_argument("--simple-chunking", action="store_true", help="Use simple chunking instead of parent-child (faster)")

    args = parser.parse_args()

    ingest_directory(
        pdf_dir=args.dir,
        clear=args.clear,
        parent_chunk_size=args.parent_chunk_size,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        use_semantic=args.semantic,
        use_vision=args.vision,
        use_parent_child=not args.simple_chunking if args.simple_chunking else None,
    )


if __name__ == "__main__":
    main()
