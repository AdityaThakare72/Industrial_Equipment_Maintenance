# Multi-stage build for smaller final image

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ app/
COPY industrial_maintenance/ industrial_maintenance/

# Copy model artifacts (baked into image — no DVC in production)
COPY models/model.joblib models/model.joblib
COPY models/preprocessor.joblib models/preprocessor.joblib

# Copy env template (Render sets real values via env vars)
COPY .env.example .env

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/')" || exit 1

# Run with gunicorn + uvicorn workers for production
CMD ["sh", "-c", "gunicorn app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
