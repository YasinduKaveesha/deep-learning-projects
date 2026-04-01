FROM python:3.11-slim

WORKDIR /app

# System deps required by opencv-python-headless
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling 2 GB CUDA wheel)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs at runtime
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
