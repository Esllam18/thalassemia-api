# =============================================================
#  Thalassemia Prediction API — Dockerfile
#  Multi-stage build: keeps the final image small and secure.
# =============================================================

# ── Stage 1: Build stage ──────────────────────────────────────
# We install Python packages here, then copy only the results
# to the final image. This keeps build tools OUT of production.
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only requirements first (Docker cache optimization)
# If requirements.txt hasn't changed, this layer is cached
# and pip install is skipped on subsequent builds.
COPY requirements.txt .

# Install all Python packages into /install (not system-wide)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime stage ────────────────────────────────────
# This is the final image that runs in production.
# It is small because it doesn't have pip, compilers, or build tools.
FROM python:3.11-slim

# Install Tesseract OCR and the English language data pack.
# This replaces the Windows-only hardcoded path in the original code.
# Tesseract is now available as 'tesseract' on $PATH — pytesseract
# finds it automatically.
#
# libgl1 is required by OpenCV for image processing.
# --no-install-recommends keeps the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY main.py .

# Copy ML model artefacts
# These files must exist in the same directory as the Dockerfile
COPY thalassemia_expert_model.pkl .
COPY label_encoder.pkl .

# Create a non-root user for security.
# Running as root in a container is a security risk.
# If the process is compromised, it can't modify system files.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Tell Docker that this container listens on port 8000
EXPOSE 8000

# Health check: Docker pings /health every 30 seconds.
# If it fails 3 times in a row, the container is marked "unhealthy".
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production command:
# gunicorn manages worker processes
# uvicorn.workers.UvicornWorker makes each worker async (handles concurrent requests)
# --workers 2 = 2 parallel workers (adjust based on server CPU count)
# --timeout 60 = kill worker if it takes > 60 seconds (prevents hanging)
CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
