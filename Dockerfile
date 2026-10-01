# RePulse Smart Campus — production image for Render/Railway
# FastAPI + OpenCV (headless) + Ultralytics YOLOv8, no Vercel size limits.

FROM python:3.10-slim

# --- Native Linux C++ dynamic libraries OpenCV needs at import time ---------
# libgl1-mesa-glx  -> libGL.so.1 (OpenGL), required by cv2 even headless
# libglib2.0-0     -> libgthread-2.0.so.0, required by OpenCV's threading code
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# --- Python dependencies -----------------------------------------------------
COPY requirements.txt .
# --extra-index-url pulls CPU-only torch/torchvision wheels (no CUDA needed
# for CPU inference on a small model) — smaller image, faster build.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# --- Pre-download YOLOv8n weights at build time ------------------------------
# Bakes yolov8n.pt into the image so the container never needs outbound
# internet access at runtime (no cold-download delay during a live demo).
#
# torch==2.4.1 (pinned above) predates PyTorch 2.6's change of torch.load's
# default from weights_only=False to weights_only=True, so this normally
# loads cleanly. The add_safe_globals call below is a defensive second layer:
# if this image is ever rebuilt against a newer torch that ignores the pin,
# it explicitly allow-lists the one Ultralytics class the checkpoint needs
# instead of silently failing. This is safe specifically because the
# checkpoint being loaded is the official asset Ultralytics publishes over
# HTTPS, fetched by us at build time — not a file supplied by an end user.
RUN python -c "\
import torch; \
from ultralytics.nn.tasks import DetectionModel; \
torch.serialization.add_safe_globals([DetectionModel]); \
from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
print('YOLOv8n weights baked successfully:', model.names is not None)"

# --- Application code ---------------------------------------------------------
COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
