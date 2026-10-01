"""RePulse Smart Campus — High-Performance Local YOLOv8 Backend Engine."""

from __future__ import annotations

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

# Путь к фронтенду index.html в той же папке
INDEX_PATH = Path(__file__).resolve().parent / "index.html"

SUCCESS_POINTS = 5
DEFAULT_SMART_LOCK_URL = "http://192.168.4"

# Оптимальный порог уверенности для честной работы (25%)
CONFIDENCE_THRESHOLD = 0.25 
YOLO_WEIGHTS = "yolov8n.pt"

REQUIRED_FRAME_COUNT = 3

model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"📡 Инициализация ИИ... Загрузка весов YOLOv8 '{YOLO_WEIGHTS}' в RAM ноутбука...")
    model = YOLO(YOLO_WEIGHTS)
    logger.info("✅ Локальное ИИ-ядро успешно запущено и готово к работе!")
    yield


app = FastAPI(title="RePulse Smart Campus (Local Core)", version="22.0.0", lifespan=lifespan)

# Разрешаем CORS, чтобы смартфон мог достучаться до ноутбука
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
            return raw.split(",", 1)[1].strip()
        return raw
    except Exception:
        return ""


def decode_frame_to_mat(payload: Any) -> Optional[np.ndarray]:
    """Декодирует кадр из Base64 напрямую в OpenCV матрицу без потери качества."""
    try:
        clean = strip_data_uri(payload).replace("\n", "").replace("\r", "").replace(" ", "")
        if not clean:
            return None
        clean += "=" * ((-len(clean)) % 4)
        raw_bytes = base64.b64decode(clean, validate=False)
        if not raw_bytes:
            return None
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning(f"Ошибка OpenCV при разборе кадра: {exc}")
        return None


def run_yolo_on_frame(frame: np.ndarray) -> list[str]:
    if model is None:
        return []
    
    # Запуск честного инференса на процессоре ноутбука
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    labels: list[str] = []
    
    if not results or len(results) == 0:
        return labels
        
    boxes = results[0].boxes
    if boxes is None or boxes.cls is None:
        return labels
        
    for cls_tensor in boxes.cls:
        cls_index = int(cls_tensor.item())
        label = model.names.get(cls_index, str(cls_index)) if isinstance(model.names, dict) else str(cls_index)
        labels.append(str(label).strip().lower())
    return labels


def analyse_frames(frames: list[np.ndarray]) -> dict[str, Any]:
    """Каскадный ИИ-анализ: жесткое разделение по официальным классам COCO."""
    all_detections = []
    for index, frame in enumerate(frames, start=1):
        labels = run_yolo_on_frame(frame)
        logger.info(f"Кадр {index} (YOLO Детекция): {labels}")
        all_detections.extend(labels)
        
    # Строго распределяем мусор по категориям
    if "bottle" in all_detections:
        return {"found": True, "material": "Plastic"}
        
    if "cup" in all_detections:
        return {"found": True, "material": "Metal"}
        
    if "box" in all_detections or "book" in all_detections:
        return {"found": True, "material": "Paper / Cardboard"}
                
    return {"found": False, "material": None}


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
        async with httpx.AsyncClient(timeout=1.2) as client:
            await client.get(url)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    if not INDEX_PATH.is_file():
        raise HTTPException(status_code=500, detail="index.html не найден в текущей папке.")
    return HTMLResponse(content=INDEX_PATH.read_text(encoding="utf-8"))


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

    # Декодируем оригинальные кадры через OpenCV
    decoded = [decode_frame_to_mat(f) for f in raw_frames]
    frames = [f for f in decoded if f is not None]
    
    if len(frames) != REQUIRED_FRAME_COUNT:
        raise HTTPException(status_code=400, detail="Ошибка декодирования OpenCV матриц.")

    # Честный прогон через YOLOv8 на вашем процессоре
    outcome = analyse_frames(frames)

    # Если ИИ ничего не нашел — жесткий и справедливый FRAUD
    if not outcome["found"]:
        return {
            "status": "fraud",
            "points": 0,
            "message": "RePulse AI: No targeted waste items detected.",
        }

    # Отправка сигнала на ESP32
    try:
        background_tasks.add_task(trigger_smart_lock, lock_url)
    except Exception:
        pass

    return {
        "status": "success",
        "points": SUCCESS_POINTS,
        "detected_material": outcome["material"],
        "message": f"Valid {outcome['material']} verified by local YOLOv8. +{SUCCESS_POINTS} PTS awarded.",
        "smart_lock_url": lock_url,
    }
