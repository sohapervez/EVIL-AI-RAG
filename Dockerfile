FROM python:3.12-slim AS builder

WORKDIR /app

# Copy project definition and source code
COPY pyproject.toml .
COPY core/ core/
COPY api.py config.py ingest.py ./

# Install production dependencies (streamlit and easyocr are now optional, not installed)
RUN pip install --no-cache-dir .

# Pre-download HuggingFace models so they are baked into the image
ENV HF_HOME=/app/models
RUN python -c "\
from sentence_transformers import CrossEncoder; \
from langchain_community.embeddings import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached.') \
"

# --- Final stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded models from builder
COPY --from=builder /app/models /app/models
ENV HF_HOME=/app/models

# Copy application source
COPY core/ core/
COPY api.py config.py ingest.py ./
COPY chat-widget-wordpress/widget/ chat-widget-wordpress/widget/

# Create data directories (will be overlaid by PVC mounts in production)
RUN mkdir -p /app/data/papers && \
    chgrp -R 0 /app/data && \
    chmod -R g=u /app/data

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
