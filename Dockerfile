FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface
    
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only PyTorch build explicitly before the remaining dependencies.
RUN python -m pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

COPY src ./src

# Cache the embedding model in the image so startup does not depend on Hugging Face.
RUN python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

RUN mkdir -p /app/data \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# A single worker avoids duplicating ML models and keeps the in-memory catalog consistent.
CMD ["python", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]