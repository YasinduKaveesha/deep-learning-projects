# Plant Disease Classifier — FastAPI inference server
#
# CPU-only build (~1.2 GB image).
# PyTorch is installed before requirements_api.txt so pip sees the
# requirement already satisfied and skips the default CUDA wheel (~3 GB).

FROM python:3.11-slim

WORKDIR /app

# ── system deps (curl for healthcheck) ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── CPU-only PyTorch (install first, saves ~3 GB vs CUDA default) ──────────
RUN pip install --no-cache-dir \
    "torch>=2.0.0" "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# ── remaining API dependencies ─────────────────────────────────────────────
# torch/torchvision are already installed above — pip will skip them here
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# ── application files ──────────────────────────────────────────────────────
COPY api/              ./api/
COPY data/processed/train.csv  ./data/processed/train.csv
COPY models/resnet50_best.pth  ./models/resnet50_best.pth

# ── runtime ────────────────────────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
