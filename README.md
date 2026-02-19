# EVIL-AI Research Paper RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for querying research papers in PDF format. This system enables users to ask questions about research papers and receive accurate answers based on the content of the indexed documents.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Models & Embeddings](#models--embeddings)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance Optimizations](#performance-optimizations)

---

## 🎯 Overview

This RAG system is designed specifically for the **EVIL-AI** research project. It allows users to:

- Query multiple research papers simultaneously
- Get answers based on specific papers or across all papers
- Receive accurate citations with source and page numbers
- Ask questions about abstracts, discussions, authors, and general research summaries

The system automatically indexes PDF files from the `data/papers/` directory and provides a clean, full-window chat interface for interaction.

---

## 🏗️ Architecture

### High-Level Flow

```
PDF Files → PDF Parser → Chunker → Embeddings → ChromaDB → Retrieval → RAG Chain → LLM → Response
```

### Components

1. **PDF Parser** (`core/pdf_parser.py`)
   - Extracts text, images (with OCR), and tables from PDFs
   - Uses PyMuPDF for robust PDF handling
   - Supports layout-aware text extraction via `pymupdf4llm`

2. **Chunker** (`core/chunker.py`)
   - Implements parent-child chunking strategy
   - Section-aware chunking for research papers
   - Supports both parent-child and simple chunking modes

3. **Vector Store** (ChromaDB)
   - Stores document embeddings
   - Maintains parent-child relationships
   - Enables fast similarity search

4. **Retriever** (`core/retriever.py`)
   - Hybrid retrieval (vector + BM25)
   - Cross-encoder re-ranking
   - Parent context expansion

5. **RAG Chain** (`core/rag_chain.py`)
   - Constructs prompts with retrieved context
   - Streams LLM responses
   - Handles conversation history

6. **UI** (`app1.py`)
   - Streamlit-based chat interface
   - Paper title extraction and display
   - Query enhancement and paper-specific filtering

---

## ✨ Key Features

### 1. **Intelligent Paper Detection**
- Automatically detects when a user is asking about a specific paper
- Supports paper number references ("3rd paper", "paper 3")
- Title-based matching with strict thresholds to prevent false positives
- Filename-based filtering for precise retrieval

### 2. **Dual Chunking Modes**
- **Parent-Child Chunking** (default): Better context expansion, slower indexing
- **Simple Chunking**: Faster indexing, fewer chunks, good for most queries

### 3. **Hybrid Retrieval**
- **Vector Search**: Semantic similarity using embeddings
- **BM25 Search**: Keyword-based retrieval
- **Weighted Fusion**: Combines both methods for optimal results
- **Cross-Encoder Re-ranking**: Fine-tunes results for relevance

### 4. **Multimodal PDF Processing**
- Text extraction with layout awareness
- Table extraction (Markdown format)
- Image OCR (optional, disabled by default for speed)
- Vision descriptions (optional, requires LLaVA model)

### 5. **Smart Query Enhancement**
- Maps paper numbers to actual titles
- Adds paper titles/filenames to queries for better retrieval
- Detects general vs. specific questions
- Adjusts retrieval breadth dynamically

---

## 🔧 Technology Stack

### Core Libraries

- **Streamlit** (`>=1.31.0`): Web UI framework
- **LangChain** (`>=0.3.0`): LLM orchestration framework
- **ChromaDB** (`>=0.5.0`): Vector database
- **PyMuPDF** (`>=1.24.0`): PDF parsing
- **pymupdf4llm** (`>=0.0.10`): Layout-aware text extraction
- **rank-bm25** (`>=0.2.2`): BM25 keyword search
- **sentence-transformers** (`>=3.0.0`): Embedding models and re-ranking

### Provider Support

The system supports multiple providers (configurable via `.env.local`):
- **LLM Providers**: Ollama, OpenAI, Anthropic, Groq
- **Embedding Providers**: Ollama, OpenAI, HuggingFace

**Current Configuration**:
- **LLM Provider**: Ollama (accessed via OpenAI-compatible endpoint)
- **LLM Model**: `llama3.1:70b`
- **Embedding Provider**: HuggingFace
- **Embedding Model**: `BAAI/bge-base-en-v1.5`

---

## 🤖 Models & Embeddings

### Models Used in This Project

#### LLM (Language Model)
- **Provider**: Ollama (via OpenAI-compatible endpoint)
- **Model**: `llama3.1:70b`
- **Configuration**: Uses OpenAI-compatible protocol with custom base URL
- **Base URL**: Configurable via `OPENAI_API_BASE` in `.env.local`

#### Embeddings
- **Provider**: HuggingFace
- **Model**: `BAAI/bge-base-en-v1.5`
- **Dimensions**: 768
- **Description**: High-quality embedding model optimized for English text retrieval

#### Re-ranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Description**: Cross-encoder model for fine-grained relevance scoring
- **Purpose**: Re-ranks retrieved documents to improve answer quality

### Note on Provider Support

While this project uses specific models, the system architecture supports multiple providers:
- **LLM Providers**: Ollama, OpenAI, Anthropic, Groq (configurable)
- **Embedding Providers**: Ollama, OpenAI, HuggingFace (configurable)

See `config.py` for all supported models and providers.

---

## 📄 Chunking Strategy

### Parent-Child Chunking (Default)

**Purpose**: Better context expansion during retrieval

**Process**:
1. **Section Detection**: Identifies paper sections (Abstract, Introduction, Methodology, etc.)
2. **Parent Chunks**: Large chunks (~4000 tokens) preserving full context
3. **Child Chunks**: Smaller chunks (~1000 tokens, 200 overlap) for precise retrieval
4. **Relationship**: Each child chunk references its parent chunk

**Benefits**:
- Retrieves precise child chunks
- Expands to full parent context for comprehensive answers
- Better handling of long-form content

**Trade-offs**:
- Creates more chunks (3-4x more than simple chunking)
- Slower indexing (~2x slower)
- More storage required

### Simple Chunking (Optional)

**Purpose**: Faster indexing with fewer chunks

**Process**:
1. **Section Detection**: Same as parent-child
2. **Direct Chunks**: Single-level chunks (~2000 tokens, 200 overlap)
3. **No Hierarchy**: Each chunk is independent

**Benefits**:
- Faster indexing (~50% faster)
- Fewer embeddings to generate
- Simpler architecture

**Trade-offs**:
- Less context expansion
- May miss some nuanced relationships

**Configuration**:
```bash
USE_PARENT_CHILD_CHUNKING=false  # Enable simple chunking
SIMPLE_CHUNK_SIZE=2000           # Chunk size
SIMPLE_CHUNK_OVERLAP=200         # Overlap between chunks
```

---

## 🔍 Retrieval Pipeline

### Stage A: Dual Retrieval

1. **Vector Search**
   - Embeds query using selected embedding model
   - Searches ChromaDB child collection using cosine similarity
   - Returns top `candidate_n` results (default: `top_k * 3`)

2. **BM25 Search**
   - Tokenizes query and documents
   - Calculates BM25 scores (keyword-based)
   - Returns top `candidate_n` results
   - **Note**: Respects `where_filter` when filtering by paper

3. **Score Fusion**
   - Combines vector and BM25 scores with weighted average
   - Default weights: Vector (60%), BM25 (40%)
   - Configurable via `BM25_WEIGHT`

### Stage B: Re-ranking

- Uses cross-encoder model to re-score candidates
- Considers query-document pairs for fine-grained relevance
- Re-orders results by re-ranking scores

### Stage C: Parent Expansion

- For each retrieved child chunk:
  - Looks up parent chunk using `parent_id`
  - Returns full parent context (larger, more complete)
- Deduplicates by parent to avoid redundant context

### Stage D: Context Formatting

- Formats retrieved contexts with metadata
- Includes source filename, page number, section
- Prepares for prompt injection

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Ollama (if using local LLM) or API keys for cloud providers

### Step 1: Clone/Download Project

```bash
cd /path/to/RAG
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

1. Copy the template:
   ```bash
   cp .env.local.template .env.local
   ```

2. Edit `.env.local` with your settings:
   ```bash
   # Provider Selection (Current Configuration)
   LLM_PROVIDER=openai  # Using OpenAI-compatible endpoint
   LLM_MODEL=llama3.1:70b
   EMBEDDING_PROVIDER=huggingface
   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
   
   # API Configuration
   OPENAI_API_KEY=your-key-here
   OPENAI_API_BASE=https://your-endpoint.com/v1  # Custom OpenAI-compatible endpoint
   ```

### Step 5: Add PDF Files

Place your research papers in `data/papers/`:
```bash
mkdir -p data/papers
# Copy your PDF files to data/papers/
```

### Step 6: Run the Application

```bash
streamlit run app1.py
```

The app will:
1. Automatically index PDFs on first run
2. Extract paper titles and authors
3. Open in your browser at `http://localhost:8501`

---

## ⚙️ Configuration

### Environment Variables

All configuration is done via `.env.local` file. See `.env.local.template` for a complete template.

#### Provider Selection (Current Configuration)
```bash
LLM_PROVIDER=openai              # Using OpenAI-compatible protocol
LLM_MODEL=llama3.1:70b          # Ollama model (llama3.1:70b)
EMBEDDING_PROVIDER=huggingface   # HuggingFace embeddings
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5  # 768-dimensional embeddings
OPENAI_API_BASE=https://your-endpoint.com/v1  # Custom OpenAI-compatible endpoint
OLLAMA_BASE_URL=http://localhost:11434  # For direct Ollama usage (if needed)
```

#### Chunking Configuration
```bash
# Parent-Child Chunking
PARENT_CHUNK_SIZE=4000           # Parent chunk size (tokens)
CHILD_CHUNK_SIZE=1000           # Child chunk size (tokens)
CHILD_CHUNK_OVERLAP=200         # Overlap between child chunks

# Simple Chunking (when USE_PARENT_CHILD_CHUNKING=false)
SIMPLE_CHUNK_SIZE=2000          # Chunk size for simple mode
SIMPLE_CHUNK_OVERLAP=200        # Overlap for simple mode

# Chunking Mode
USE_PARENT_CHILD_CHUNKING=false # true = parent-child, false = simple
USE_SEMANTIC_CHUNKING=false     # Use semantic chunking (experimental)
```

#### PDF Processing
```bash
EXTRACT_IMAGES=false            # Extract images from PDFs
USE_IMAGE_OCR=false             # Run OCR on images (slow)
EXTRACT_TABLES=true             # Extract tables as Markdown
USE_VISION_DESCRIPTIONS=false   # Generate vision descriptions (requires LLaVA)
```

#### Retrieval Configuration
```bash
TOP_K=5                         # Number of chunks to retrieve
USE_HYBRID_SEARCH=true          # Enable hybrid (vector + BM25) search
BM25_WEIGHT=0.4                 # BM25 weight (0.0-1.0), vector = 1.0 - BM25_WEIGHT
USE_RERANKING=true              # Enable cross-encoder re-ranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

#### ChromaDB Configuration
```bash
CHROMA_PERSIST_DIR=./vectorstore  # Where to store the vector database
CHROMA_COLLECTION=research_papers  # Collection name
```

---

## 💻 Usage

### Starting the Application

```bash
streamlit run app1.py
```

### Querying Papers

#### General Questions
Ask questions that span multiple papers:
- "What research has been done under EVIL-AI?"
- "Summarize all the work done"
- "What are the main findings across all papers?"

#### Paper-Specific Questions
Ask about a specific paper by:
- **Paper number**: "What is the summary of the 3rd paper?"
- **Paper title**: "What are the core components in Creation of AI-driven smart spaces..."
- **Filename**: Questions mentioning the PDF filename

#### Author Questions
- "Who are the authors of paper number 3?"
- "List the authors of [paper title]"

#### Content Questions
- "What is the abstract of paper 1?"
- "What has been discussed in [paper title]?"
- "Explain the methodology used in the first paper"

### Re-indexing Papers

If you add new PDFs or change chunking settings:

```bash
# Delete existing index
rm -rf vectorstore/

# Restart the app (it will auto-index)
streamlit run app1.py
```

Or use the CLI:
```bash
python ingest.py --clear  # Re-index all PDFs
```

---

## 📁 Project Structure

```
RAG/
├── app1.py                 # Main Streamlit application
├── app.py                  # Alternative UI (with sidebar)
├── ingest.py               # CLI script for batch indexing
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── .env.local             # Your local configuration (git-ignored)
├── .env.local.template    # Configuration template
├── README.md              # This file
│
├── core/                  # Core RAG components
│   ├── __init__.py
│   ├── pdf_parser.py      # PDF parsing (text, images, tables)
│   ├── chunker.py         # Parent-child and simple chunking
│   ├── providers.py        # LLM and embedding provider factory
│   ├── retriever.py       # Hybrid retrieval pipeline
│   └── rag_chain.py       # RAG chain construction and streaming
│
├── data/
│   └── papers/            # Place PDF files here
│       ├── paper1.pdf
│       ├── paper2.pdf
│       └── ...
│
└── vectorstore/           # ChromaDB storage (auto-created)
    ├── research_papers_children/  # Child chunks with embeddings
    └── research_papers_parents/   # Parent chunks (text only)
```

---

## ⚡ Performance Optimizations

### Indexing Speed

1. **Parallel PDF Processing**
   - Processes multiple PDFs concurrently
   - Uses `ThreadPoolExecutor` with CPU count - 1 workers

2. **Simple Chunking Mode**
   - Reduces chunks by ~50%
   - Faster embedding generation
   - Enable: `USE_PARENT_CHILD_CHUNKING=false`

3. **Disabled Image Processing**
   - Skips OCR (saves ~4 seconds per PDF)
   - Skips image extraction (saves ~2-3 seconds per PDF)
   - Configure: `EXTRACT_IMAGES=false`, `USE_IMAGE_OCR=false`

4. **Optimized Title Extraction**
   - Fetches only page 1 metadata first
   - Loads only necessary documents
   - Reduces ChromaDB query time

5. **Batch Embedding**
   - Processes embeddings in batches of 200
   - Reduces API calls and improves throughput

### Query Speed

1. **Source Filtering**
   - Filters by specific paper when detected
   - Reduces search space significantly

2. **Dynamic Top-K**
   - General queries: `top_k * 3` (more context)
   - Specific queries: `top_k` (faster)

3. **Embedding Model Caching**
   - Loads embedding models once per session
   - Reuses across multiple queries

---

## 🔬 Technical Details

### Embedding Dimensions

**Current Model**: `BAAI/bge-base-en-v1.5` (768 dimensions)

**Important**: If you change the embedding model, you **must** re-index by deleting the `vectorstore/` folder. Mismatched dimensions will cause retrieval errors.

**Supported Models**:
- `BAAI/bge-base-en-v1.5`: 768 dimensions (currently used)
- `all-MiniLM-L6-v2`: 384 dimensions
- `all-mpnet-base-v2`: 768 dimensions

### ChromaDB Collections

- **Child Collection**: Stores child chunks with embeddings
  - Used for vector search
  - Metadata includes: `source`, `page`, `section`, `content_type`, `parent_id`
  
- **Parent Collection**: Stores parent chunks as text
  - Used for context expansion
  - Metadata includes: `source`, `page`, `section`, `content_type`

### Query Enhancement Logic

The system enhances queries in several ways:

1. **Paper Number Mapping**: "3rd paper" → Adds paper title and filename
2. **Title Detection**: Detects paper titles in queries → Adds filename
3. **General Query Detection**: Adds all paper titles for comprehensive retrieval

### Title Matching Algorithm

Uses strict scoring to prevent wrong paper selection:

1. **Token Normalization**: Lowercase, remove punctuation, split on spaces/hyphens
2. **Stop Word Removal**: Filters common words
3. **Overlap Calculation**: `overlap = (# common tokens) / (# title tokens)`
4. **Thresholds**: Requires `common >= 3` AND `overlap >= 0.35`
5. **Ambiguity Check**: Must be 10% better than second-best match

---

## 🐛 Troubleshooting

### "502 Bad Gateway" Error

If using a custom OpenAI-compatible endpoint:
- Ensure `OPENAI_API_BASE` doesn't end with `/v1` (LangChain adds it)
- Check that the endpoint is accessible
- Verify API key is correct

### Wrong Paper Detected

- The title matching uses strict thresholds
- If a wrong paper is detected, the query might be ambiguous
- Try being more specific: include more words from the paper title

### Slow Indexing

- Enable simple chunking: `USE_PARENT_CHILD_CHUNKING=false`
- Disable image processing: `EXTRACT_IMAGES=false`
- Increase parallel workers (automatic based on CPU count)

### "Failed to Index" Warning

- PDF might be corrupted or have unsupported formatting
- Try re-saving the PDF
- Check PDF is not password-protected
- Verify PDF is not just images (needs extractable text)

### Embedding Dimension Mismatch

- Delete `vectorstore/` folder
- Re-index with the new embedding model
- Ensure all PDFs are re-indexed

---

## 📝 Notes

- The system automatically extracts paper titles and authors from page 1
- Failed PDFs are marked as `[FAILED TO INDEX]` but still listed
- Author names are cleaned (removes ORCID IDs, superscripts, brackets)
- The system never mentions authors unless explicitly asked
- Citations follow format: `[Source: filename.pdf, Page X]`

---

## 🔄 Version History

- **v1.0**: Initial RAG system with parent-child chunking
- **v1.1**: Added simple chunking mode for faster indexing
- **v1.2**: Improved title matching with strict thresholds
- **v1.3**: Added parallel PDF processing and performance optimizations
- **v1.4**: Enhanced query enhancement and paper-specific filtering

---

## 📄 License

[Add your license here]

---

## 👥 Contributors

[Add contributors here]

---

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- PDF processing with [PyMuPDF](https://pymupdf.readthedocs.io/)

---

For questions or issues, please refer to the code comments or create an issue in the repository.
