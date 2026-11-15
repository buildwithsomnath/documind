FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/backend:/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-cache HuggingFace Transformer models for instant container startup
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=False); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', token=False)"

# Copy application source code
COPY backend/ /app/backend/
COPY evaluation/ /app/evaluation/

# Ensure uploads directory exists
RUN mkdir -p /app/backend/uploads

WORKDIR /app/backend

EXPOSE 5000

# Run entrypoint: auto-initialize db tables if needed, then run gunicorn
CMD ["sh", "-c", "python manage.py init-db && gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 'app:app'"]
