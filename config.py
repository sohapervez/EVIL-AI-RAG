"""Centralized configuration for the RAG system.

Loads environment variables with the following priority (highest first):
  1. .env.local   (your machine-specific overrides — git-ignored)
  2. .env          (shared project defaults)
  3. OS environment variables
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent

# Load in reverse priority order: .env first (base), then .env.local (overrides)
load_dotenv(_project_root / ".env", override=False)
load_dotenv(_project_root / ".env.local", override=True)


# ---------------------------------------------------------------------------
# Provider Selection
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1")
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# API Keys (only needed for cloud providers)
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "")  # Custom base URL for OpenAI-compatible APIs
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Parent-Child Chunking
# ---------------------------------------------------------------------------
PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", "4000"))
CHILD_CHUNK_SIZE: int = int(os.getenv("CHILD_CHUNK_SIZE", "1000"))
CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "200"))
USE_SEMANTIC_CHUNKING: bool = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
USE_VISION_DESCRIPTIONS: bool = os.getenv("USE_VISION_DESCRIPTIONS", "false").lower() == "true"
USE_IMAGE_OCR: bool = os.getenv("USE_IMAGE_OCR", "false").lower() == "true"  # Skip OCR by default for speed
EXTRACT_IMAGES: bool = os.getenv("EXTRACT_IMAGES", "true").lower() == "true"  # Extract images from PDFs
EXTRACT_TABLES: bool = os.getenv("EXTRACT_TABLES", "true").lower() == "true"  # Extract tables from PDFs
USE_PARENT_CHILD_CHUNKING: bool = os.getenv("USE_PARENT_CHILD_CHUNKING", "true").lower() == "true"  # Use parent-child chunking (slower but better context)
SIMPLE_CHUNK_SIZE: int = int(os.getenv("SIMPLE_CHUNK_SIZE", "2000"))  # Chunk size when parent-child is disabled
SIMPLE_CHUNK_OVERLAP: int = int(os.getenv("SIMPLE_CHUNK_OVERLAP", "200"))  # Overlap when parent-child is disabled

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "5"))
USE_HYBRID_SEARCH: bool = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))
USE_RERANKING: bool = os.getenv("USE_RERANKING", "true").lower() == "true"
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "research_papers")

# Derived collection names for parent-child pattern
CHROMA_CHILD_COLLECTION: str = f"{CHROMA_COLLECTION}_children"
CHROMA_PARENT_COLLECTION: str = f"{CHROMA_COLLECTION}_parents"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data", "papers")

# ---------------------------------------------------------------------------
# Available models per provider (for UI dropdowns)
# ---------------------------------------------------------------------------
AVAILABLE_LLM_MODELS: dict[str, list[str]] = {
    "ollama": ["llama3.1", "llama3.2", "llama2-uncensored:latest", "mistral", "gemma2", "phi3", "qwen2.5"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    "groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
}

AVAILABLE_EMBEDDING_MODELS: dict[str, list[str]] = {
    "ollama": ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
    "huggingface": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are a research assistant. Answer the question using ONLY the provided \
context from research papers. Cite your sources as [Source: filename, Page X]. \
If the context does not contain enough information, say so clearly. \
Be precise and thorough in your answers."""
