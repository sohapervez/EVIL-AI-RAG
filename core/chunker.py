"""Parent-Child chunking: sections -> parent chunks (4k) -> child chunks (1k, 200 overlap).

Produces two lists:
- Parent chunks  (stored as text, looked up by parent_id)
- Child  chunks  (embedded + indexed, each carrying a parent_id)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter

import config
from core.pdf_parser import ParsedElement

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section header patterns for research papers
# ---------------------------------------------------------------------------
_SECTION_PATTERN = re.compile(
    r"(?:^|\n)"  # start of string or newline
    r"("
    r"(?:Abstract|ABSTRACT)"  # Abstract
    r"|(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)"  # References
    r"|(?:\d+\.?\d*\.?\d*\.?\s+[A-Z][^\n]{2,80})"  # Numbered sections: "1. Introduction", "3.1 Method"
    r"|(?:[IVXLC]+\.\s+[A-Z][^\n]{2,80})"  # Roman numeral sections
    r")"
)

_REFERENCES_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ParentChunk:
    """A parent chunk stored for context expansion."""

    parent_id: str
    content: str
    source: str
    page: int
    section: str
    content_type: str  # "text" | "table" | "image_ocr" | "image_description"
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    """A child chunk that gets embedded and indexed."""

    child_id: str
    parent_id: str
    content: str
    source: str
    page: int
    section: str
    content_type: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_id(prefix: str, source: str, page: int, section: str, idx: int, content: str = "") -> str:
    """Create a deterministic ID for deduplication on re-ingestion.

    Includes a hash of the actual content to avoid collisions when
    structural metadata (source/page/section/idx) is identical ---
    e.g. repeated section headers from PDF headers/footers.
    """
    raw = f"{prefix}:{source}:{page}:{section}:{idx}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _split_into_sections(full_text: str) -> list[tuple[str, str]]:
    """Split paper text into (section_name, section_text) tuples.

    Uses regex to detect section headers. Falls back to a single section
    named 'Body' if no headers are found.
    """
    # Find all section header positions
    matches = list(_SECTION_PATTERN.finditer(full_text))

    if not matches:
        return [("Body", full_text.strip())]

    sections: list[tuple[str, str]] = []

    # Text before the first section header
    pre_text = full_text[: matches[0].start()].strip()
    if pre_text:
        sections.append(("Preamble", pre_text))

    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        if body:
            sections.append((header, body))

    return sections


def _should_exclude_section(section_name: str) -> bool:
    """Return True for sections that add noise (e.g. References)."""
    return bool(_REFERENCES_PATTERN.match(f"\n{section_name}\n"))


# ---------------------------------------------------------------------------
# Semantic chunking helper (optional, uses embeddings)
# ---------------------------------------------------------------------------
def _get_semantic_splitter():
    """Lazy-build a SemanticSplitterNodeParser using the configured embedding model.

    Falls back gracefully if the LlamaIndex semantic splitter is unavailable.
    """
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    from core.providers import get_embedding_model

    embed_model = get_embedding_model(config.EMBEDDING_PROVIDER, config.EMBEDDING_MODEL)
    return SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95,
    )


def _semantic_split_text(semantic_splitter, text: str) -> list[str]:
    """Use the semantic splitter to split raw text.

    The SemanticSplitterNodeParser works on Document objects, so we wrap
    the text in a Document, split, and extract the text back out.
    """
    from llama_index.core.schema import Document

    doc = Document(text=text)
    nodes = semantic_splitter.get_nodes_from_documents([doc])
    return [node.get_content() for node in nodes]


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------
def chunk_elements(
    elements: list[ParsedElement],
    parent_chunk_size: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    child_chunk_overlap: Optional[int] = None,
    use_semantic: Optional[bool] = None,
    use_parent_child: Optional[bool] = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Run the chunking pipeline on parsed PDF elements.

    If use_parent_child is False, uses simple single-level chunking (faster, fewer chunks).
    Otherwise uses parent-child chunking (slower, better context expansion).

    Returns (parent_chunks, child_chunks).
    """
    use_parent_child = use_parent_child if use_parent_child is not None else config.USE_PARENT_CHILD_CHUNKING

    # If simple chunking, use that instead
    if not use_parent_child:
        return _chunk_elements_simple(elements, parent_chunk_size, child_chunk_size, child_chunk_overlap, use_semantic)

    parent_chunk_size = parent_chunk_size or config.PARENT_CHUNK_SIZE
    child_chunk_size = child_chunk_size or config.CHILD_CHUNK_SIZE
    child_chunk_overlap = child_chunk_overlap or config.CHILD_CHUNK_OVERLAP
    use_semantic = use_semantic if use_semantic is not None else config.USE_SEMANTIC_CHUNKING

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    # Splitters (LlamaIndex SentenceSplitter replaces LangChain RecursiveCharacterTextSplitter)
    parent_splitter = SentenceSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=0,
    )
    child_splitter = SentenceSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )
    semantic_splitter = None
    if use_semantic:
        try:
            semantic_splitter = _get_semantic_splitter()
        except Exception as e:
            logger.warning("Semantic chunker unavailable, falling back to sentence splitter: %s", e)

    child_index = 0  # global child counter for ordering

    for element in elements:
        source = element.source
        page = element.page
        content_type = element.content_type

        # ----- Tables and images: single child, single parent -----
        if content_type in ("table", "image_ocr", "image_description"):
            pid = _make_id("parent", source, page, content_type, child_index, element.content)
            parent = ParentChunk(
                parent_id=pid,
                content=element.content,
                source=source,
                page=page,
                section=content_type,
                content_type=content_type,
                metadata=element.metadata,
            )
            parents.append(parent)

            cid = _make_id("child", source, page, content_type, child_index, element.content)
            child = ChildChunk(
                child_id=cid,
                parent_id=pid,
                content=element.content,
                source=source,
                page=page,
                section=content_type,
                content_type=content_type,
                chunk_index=child_index,
                metadata=element.metadata,
            )
            children.append(child)
            child_index += 1
            continue

        # ----- Text: section-aware -> parent -> child -----
        sections = _split_into_sections(element.content)

        for section_name, section_text in sections:
            if _should_exclude_section(section_name):
                logger.debug("Excluding section: %s", section_name)
                continue

            # Stage 2: split section into parent chunks
            parent_texts = parent_splitter.split_text(section_text)

            for p_idx, p_text in enumerate(parent_texts):
                pid = _make_id("parent", source, page, section_name, p_idx, p_text)
                parent = ParentChunk(
                    parent_id=pid,
                    content=p_text,
                    source=source,
                    page=page,
                    section=section_name,
                    content_type="text",
                )
                parents.append(parent)

                # Stage 3: split parent into child chunks
                if semantic_splitter is not None:
                    try:
                        child_texts = _semantic_split_text(semantic_splitter, p_text)
                    except Exception:
                        child_texts = child_splitter.split_text(p_text)
                else:
                    child_texts = child_splitter.split_text(p_text)

                for c_text in child_texts:
                    cid = _make_id("child", source, page, section_name, child_index, c_text)
                    child = ChildChunk(
                        child_id=cid,
                        parent_id=pid,
                        content=c_text,
                        source=source,
                        page=page,
                        section=section_name,
                        content_type="text",
                        chunk_index=child_index,
                    )
                    children.append(child)
                    child_index += 1

    logger.info(
        "Chunking complete: %d parents, %d children from %d elements",
        len(parents),
        len(children),
        len(elements),
    )
    return parents, children


def _chunk_elements_simple(
    elements: list[ParsedElement],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_semantic: Optional[bool] = None,
    parent_chunk_size: Optional[int] = None,  # Ignored, kept for API compatibility
    child_chunk_size: Optional[int] = None,  # Used as chunk_size if provided
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Simple single-level chunking: creates fewer chunks for faster embedding.

    Creates chunks directly from sections without parent-child hierarchy.
    Each chunk is both a parent and child (for API compatibility).
    """
    chunk_size = child_chunk_size or chunk_size or config.SIMPLE_CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.SIMPLE_CHUNK_OVERLAP
    use_semantic = use_semantic if use_semantic is not None else config.USE_SEMANTIC_CHUNKING

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    # Single splitter for simple chunking (LlamaIndex SentenceSplitter)
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    semantic_splitter = None
    if use_semantic:
        try:
            semantic_splitter = _get_semantic_splitter()
        except Exception as e:
            logger.warning("Semantic chunker unavailable, falling back to sentence splitter: %s", e)

    chunk_index = 0

    for element in elements:
        source = element.source
        page = element.page
        content_type = element.content_type

        # Tables and images: single chunk
        if content_type in ("table", "image_ocr", "image_description"):
            pid = _make_id("parent", source, page, content_type, chunk_index, element.content)
            cid = _make_id("child", source, page, content_type, chunk_index, element.content)

            parent = ParentChunk(
                parent_id=pid,
                content=element.content,
                source=source,
                page=page,
                section=content_type,
                content_type=content_type,
                metadata=element.metadata,
            )
            parents.append(parent)

            child = ChildChunk(
                child_id=cid,
                parent_id=pid,  # Same as parent_id for simple chunking
                content=element.content,
                source=source,
                page=page,
                section=content_type,
                content_type=content_type,
                chunk_index=chunk_index,
                metadata=element.metadata,
            )
            children.append(child)
            chunk_index += 1
            continue

        # Text: section-aware -> simple chunks
        sections = _split_into_sections(element.content)

        for section_name, section_text in sections:
            if _should_exclude_section(section_name):
                logger.debug("Excluding section: %s", section_name)
                continue

            # Split section into chunks
            if semantic_splitter is not None:
                try:
                    chunk_texts = _semantic_split_text(semantic_splitter, section_text)
                except Exception:
                    chunk_texts = splitter.split_text(section_text)
            else:
                chunk_texts = splitter.split_text(section_text)

            for c_idx, c_text in enumerate(chunk_texts):
                pid = _make_id("parent", source, page, section_name, chunk_index, c_text)
                cid = _make_id("child", source, page, section_name, chunk_index, c_text)

                parent = ParentChunk(
                    parent_id=pid,
                    content=c_text,
                    source=source,
                    page=page,
                    section=section_name,
                    content_type="text",
                )
                parents.append(parent)

                child = ChildChunk(
                    child_id=cid,
                    parent_id=pid,  # Same as parent_id for simple chunking
                    content=c_text,
                    source=source,
                    page=page,
                    section=section_name,
                    content_type="text",
                    chunk_index=chunk_index,
                )
                children.append(child)
                chunk_index += 1

    logger.info(
        "Simple chunking complete: %d chunks from %d elements",
        len(children),
        len(elements),
    )
    return parents, children
