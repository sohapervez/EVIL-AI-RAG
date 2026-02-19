# EVIL-AI RAG

A Retrieval-Augmented Generation system for querying research papers. Upload PDFs, ask questions, get answers with source citations.

## Architecture

```
PDF --> Parser --> Chunker --> Embeddings --> ChromaDB --> Hybrid Retrieval --> Reranker --> LLM --> Response
```

- **PDF parsing**: PyMuPDF with layout-aware extraction (`pymupdf4llm`), optional table and image extraction
- **Chunking**: Parent-child strategy with section detection (LlamaIndex `SentenceSplitter`)
- **Embeddings**: HuggingFace `BAAI/bge-base-en-v1.5` (768 dim), also supports OpenAI and Ollama
- **Vector store**: ChromaDB with dual collections (parents for context, children for search)
- **Retrieval**: Hybrid vector + BM25 search with weighted fusion, cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), parent context expansion
- **LLM**: Supports Ollama, OpenAI, Anthropic, Groq (configurable)

## Setup

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.local.template .env.local   # edit with your API keys and provider settings
```

Place PDFs in `data/papers/`, then:

```bash
# Streamlit UI (install dev deps first: pip install -e ".[dev]")
streamlit run app.py

# Or run the API server
uvicorn api:app --host 0.0.0.0 --port 8080

# CLI ingestion
python ingest.py              # index new PDFs
python ingest.py --clear      # re-index everything
```

### Docker

```bash
docker compose up --build
```

This starts the API on port 8080 and ChromaDB on port 8000.

### Rahti (OpenShift)

Kubernetes manifests are in `rahti/`. Copy `rahti/secrets.yaml.template` to `rahti/secrets.yaml`, fill in your secrets, then:

```bash
oc apply -f rahti/
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/chat` | - | SSE streaming chat |
| GET | `/api/v1/papers` | - | List indexed papers |
| GET | `/api/v1/health` | - | Health check |
| POST | `/api/v1/papers` | Bearer | Upload and ingest a PDF |
| DELETE | `/api/v1/papers/{filename}` | Bearer | Remove a paper |
| POST | `/api/v1/reindex` | Bearer | Full re-index |
| GET | `/api/v1/analytics` | Bearer | Usage summary |
| GET | `/api/v1/analytics/questions` | Bearer | Paginated question log |
| GET | `/api/v1/analytics/export` | Bearer | CSV export |

## Configuration

All settings are loaded from `.env.local` (see `.env.local.template`). Key variables:

```bash
LLM_PROVIDER=ollama              # ollama | openai | anthropic | groq
LLM_MODEL=llama3.1
EMBEDDING_PROVIDER=huggingface   # huggingface | openai | ollama
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

USE_PARENT_CHILD_CHUNKING=true
USE_HYBRID_SEARCH=true
BM25_WEIGHT=0.4
USE_RERANKING=true
TOP_K=5
```

## Project Structure

```
api.py                  # FastAPI backend (production)
app.py                  # Streamlit UI (local development)
config.py               # Configuration loader
ingest.py               # CLI PDF ingestion
core/
  pdf_parser.py         # PDF text/table/image extraction
  chunker.py            # Parent-child and simple chunking
  providers.py          # LLM and embedding provider factory
  retriever.py          # Hybrid retrieval pipeline
  rag_chain.py          # Prompt construction and LLM streaming
  paper_metadata.py     # Paper title/author extraction, query classification
  analytics.py          # Usage logging (SQLite)
chat-widget-wordpress/  # Embeddable chat widget for WordPress
rahti/                  # OpenShift/Kubernetes deployment manifests
```
