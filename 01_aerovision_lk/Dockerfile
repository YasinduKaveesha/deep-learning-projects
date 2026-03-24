FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements-docker.txt

COPY app/ app/
COPY weights/yolov8s_int8.onnx weights/yolov8s_int8.onnx

ENV INFERENCE_MODE=onnx

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
