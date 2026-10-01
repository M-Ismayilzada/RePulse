# RePulse Smart Campus — Production Image for PaaS Deploy (Render/Railway)
# FastAPI + OpenCV (headless) + Ultralytics YOLOv8, defensively optimized.

FROM python:3.10-slim

# --- Native Linux C++ dynamic libraries OpenCV needs at import time ---------
# libgl1        -> libGL.so.1 (OpenGL), required by cv2 even headless (Debian 13 Trixie compliant)
# libglib2.0-0  -> libgthread-2.0.so.0, required by OpenCV's threading code
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# --- Python dependencies -----------------------------------------------------
COPY requirements.txt .
# --extra-index-url pulls CPU-only torch/torchvision wheels — smaller image, faster build.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://pytorch.org

# --- Pre-download YOLOv8n weights at build time ------------------------------
# Bakes yolov8n.pt into the image so the container never needs outbound
# internet access at runtime. Includes a defensive allowlist for PyTorch >=2.6.
RUN python -c "\
import torch; \
from ultralytics.nn.tasks import DetectionModel; \
torch.serialization.add_safe_globals([DetectionModel]); \
from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
print('YOLOv8n weights baked successfully:', model.names is not None)"

# --- Production Environment Tuning -------------------------------------------
# Pins PyTorch to a single thread to avoid thread contention on tight CPU cores.
ENV TORCH_THREADS=1
ENV YOLO_IMGSZ=320
ENV YOLO_CONF=0.15

# --- Application code ---------------------------------------------------------
COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
