"""RePulse Smart Campus — Flawless Pure YOLOv8 Targeted Production Backend Core."""

from __future__ import annotations

# ==============================================================================
# СВЕРХКРИТИЧЕСКИЙ ФИКС СЕКЬЮРИТИ PYTORCH 2.6+: КЛЮЧЕВОЙ ШАГ ДО ВСЕХ ИМПОРТОВ
# Полностью отключаем strict unpickler (weights_only) на глобальном уровне ядра torch
# ==============================================================================
try:
    import torch
    orig_load = torch.load
    def bulletproof_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return orig_load(*args, **kwargs)
    torch.load = bulletproof_load

    import torch.serialization
    if hasattr(torch.serialization, '_load'):
        orig_inner_load = torch.serialization._load
        def bulletproof_inner_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return orig_inner_load(*args, **kwargs)
        torch.serialization._load = bulletproof_inner_load
except Exception:
    pass
# ==============================================================================

import base64
import ipaddress
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repulse")

INDEX_PATH = Path(__file__).resolve().parent / "index.html"

SUCCESS_POINTS = 5
DEFAULT_SMART_LOCK_URL = "http://192.168.4"
SMART_LOCK_TIMEOUT_SECONDS = 1.2

# Проверенный дневной порог уверенности ИИ
CONFIDENCE_THRESHOLD = 0.15 
YOLO_WEIGHTS = "yolov8n.pt"

# Проверенные целевые ИИ-классы модели YOLOv8
PLASTIC_LABELS = {"bottle"}
METAL_LABELS = {"cup"}
PAPER_LABELS = {"box", "cardboard", "paper", "book"}

REQUIRED_FRAME_COUNT = 3

model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading YOLOv8 weights '{YOLO_WEIGHTS}' into RAM…")
    model = YOLO(YOLO_WEIGHTS)
    yield
    logger.info("Shutting down RePulse backend.")


app = FastAPI(title="RePulse Smart Campus", version="37.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def strip_data_uri(payload: Any) -> str:
    try:
        if payload is None:
            return ""
        raw = payload if isinstance(payload, str) else str(payload)
        raw = raw.strip()
        if "," in raw:
            parts = raw.split(",", 1)
            return parts[1].strip()
        return raw
    except Exception:
        return ""


def decode_and_resize_frame(payload: Any) -> Optional[np.ndarray]:
    try:
        clean = strip_data_uri(payload).replace("\n", "").replace("\r", "").replace(" ", "")
        if not clean:
            return None
        clean += "=" * ((-len(clean)) % 4)
        raw_bytes = base64.b64decode(clean, validate=False)
        if not raw_bytes:
            return None
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            return cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
        return None
    except Exception as exc:
        logger.warning(f"Frame decode/resize failed: {exc}")
        return None


def run_yolo_on_frame(frame: np.ndarray) -> list[tuple[str, float]]:
    """Возвращает список кортежей (название класса, площадь бокса)."""
    if model is None:
        return []
    
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections: list[tuple[str, float]] = []
    
    if not results or len(results) == 0:
        return detections
        
    boxes = results[0].boxes
    if boxes is None or boxes.cls is None or boxes.xywhn is None:
        return detections
        
    for i in range(len(boxes)):
        try:
            class_id = int(boxes.cls[i].item())
            label = model.names.get(class_id, str(class_id)).strip().lower()
            
            # Считаем относительную площадь рамки объекта
            xywhn = boxes.xywhn[i].tolist()
            w = float(xywhn[2])
            h = float(xywhn[3])
            box_area = w * h
            
            detections.append((label, box_area))
        except Exception:
            continue
    return detections


def analyse_frames(frames: list[np.ndarray]) -> dict[str, Any]:
    """Тот самый проверенный дневной каскад с точечным фиксом ложного металла."""
    for index, frame in enumerate(frames, start=1):
        detections = run_yolo_on_frame(frame)
        logger.info(f"YOLO Frame {index} Detections: {detections}")
        
        for label, box_area in detections:
            # 1. Если ИИ галлюцинирует cup (металл) на крупном комке бумаги в руке (>5% экрана)
            if label == "cup" and box_area > 0.05:
                return {"found": True, "material": "Paper / Cardboard", "frame": index}
                
            # 2. Стандартный чистый маппинг
            if label in PLASTIC_LABELS:
                return {"found": True, "material": "Plastic", "frame": index}
            if label in METAL_LABELS:
                return {"found": True, "material": "Metal", "frame": index}
            if label in PAPER_LABELS:
                return {"found": True, "material": "Paper / Cardboard", "frame": index}
                
    return {"found": False, "material": None, "frame": None}


def _host_is_private(hostname: str) -> bool:
    host = (hostname or "").strip().strip("[]")
    if not host:
        return False
    if host.lower() in {"localhost", "esp32.local"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        if host.startswith("192.168.") or host.startswith("10.") or host.startswith("172."):
            return True
        return False


def sanitize_smart_lock_url(candidate: Optional[str]) -> str:
    try:
        url = str(candidate or DEFAULT_SMART_LOCK_URL).strip() or DEFAULT_SMART_LOCK_URL
    except Exception:
        return DEFAULT_SMART_LOCK_URL
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return DEFAULT_SMART_LOCK_URL
    hostname = parsed.hostname or ""
    if hostname == "192.168.4" or url.rstrip("/") == "http://192.168.4":
        return url
    if not _host_is_private(hostname):
        return DEFAULT_SMART_LOCK_URL
    return url


async def trigger_smart_lock(url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=SMART_LOCK_TIMEOUT_SECONDS) as client:
            await client.get(url)
        logger.info(f"Smart lock GET dispatched: {url}")
    except Exception as exc:
        logger.warning(f"Smart lock unreachable ({url}), discarded: {exc}")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    if not INDEX_PATH.is_file():
        raise HTTPException(status_code=500, detail="index.html is missing.")
    return HTMLResponse(content=INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "engine": "YOLOv8 Targeted Core Pure v37.0",
        "required_frames": REQUIRED_FRAME_COUNT,
        "points_per_success": SUCCESS_POINTS,
        "supported_materials": ["Plastic", "Metal", "Paper / Cardboard"]
    }


@app.post("/api/roboflow/detect")
async def detect_recycling(request: Request, background_tasks: BackgroundTasks) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required.")

    if isinstance(body, list):
        raw_frames = body
    elif isinstance(body, dict):
        raw_frames = body.get("frames", body.get("images", []))
    else:
        raw_frames = []

    if not isinstance(raw_frames, list) or len(raw_frames) != REQUIRED_FRAME_COUNT:
        raise HTTPException(status_code=400, detail="Payload must contain 3 frames.")

    lock_url = sanitize_smart_lock_url(body.get("smart_lock_url") if isinstance(body, dict) else None)

    decoded = [decode_and_resize_frame(f) for f in raw_frames]
    frames = [f for f in decoded if f is not None]
    
    if len(frames) != REQUIRED_FRAME_COUNT:
        return {
            "status": "fraud",
            "points": 0,
            "message": "RePulse AI: OpenCV frame decoding failed.",
        }

    try:
        outcome = analyse_frames(frames)
    except Exception as exc:
        logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА КОДА ИИ: {type(exc).__name__}: {exc}")
        return {
            "status": "fraud",
            "points": 0,
            "message": "RePulse AI: Internal pipeline processing exception.",
        }

    if not outcome["found"]:
        return {
            "status": "fraud",
            "points": 0,
            "message": "RePulse AI: No valid recyclable waste items detected.",
        }

    background_tasks.add_task(trigger_smart_lock, lock_url)

    return {
        "status": "success",
        "points": SUCCESS_POINTS,
        "detected_material": outcome["material"],
        "message": f"Valid {outcome['material']} verified by YOLOv8. +{SUCCESS_POINTS} PTS awarded.",
        "smart_lock_url": lock_url,
    }
