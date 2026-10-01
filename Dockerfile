# RePulse Smart Campus — Secure PaaS Dockerfile
# FastAPI + OpenCV (headless) + Ultralytics YOLOv8

FROM python:3.10-slim

# --- Native Linux C++ dynamic libraries OpenCV needs at import time ---------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# --- Python dependencies -----------------------------------------------------
COPY requirements.txt .
# --extra-index-url pulls CPU-only torch/torchvision wheels (Lightweight version)
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://pytorch.org

# --- Production Environment Tuning -------------------------------------------
ENV TORCH_THREADS=1
ENV YOLO_IMGSZ=320
ENV YOLO_CONF=0.15

# --- Application code ---------------------------------------------------------
COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
