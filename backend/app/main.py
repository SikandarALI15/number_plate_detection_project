from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend.app.detector import NumberPlateDetector

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
LOGGER = logging.getLogger("platevision.api")

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BACKEND_DIR / "models" / "best.pt"))).expanduser()
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "12"))
MAX_IMAGE_BYTES = int(MAX_IMAGE_MB * 1024 * 1024)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if origin.strip()
]


def create_detector() -> NumberPlateDetector:
    return NumberPlateDetector(MODEL_PATH)


detector = create_detector()


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detection_count: int
    detections: list[DetectionBox]
    inference_ms: float
    device: str
    annotated_image_base64: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.is_file():
        try:
            await asyncio.to_thread(detector.load_model)
        except Exception:
            LOGGER.exception("Model could not be loaded during startup")
            raise
    else:
        LOGGER.error("Model file is missing: %s", MODEL_PATH)
    yield


app = FastAPI(
    title="PlateVision API",
    description="License plate detection API powered by a custom-trained YOLO11n model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    return {"name": "PlateVision API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health", tags=["System"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL_PATH.is_file() else "model_missing",
        "model_name": "YOLO11n License Plate Detector",
        "model_file_exists": MODEL_PATH.is_file(),
        "model_loaded": detector.is_loaded,
        "model_path": str(MODEL_PATH),
    }


@app.post("/api/detect", response_model=DetectionResponse, tags=["Inference"])
async def detect_license_plates(
    file: UploadFile = File(..., description="JPG, PNG or WEBP image"),
    confidence: float = Form(0.35, ge=0.01, le=0.99),
    iou: float = Form(0.45, ge=0.05, le=0.95),
    image_size: int = Form(640, ge=320, le=1280),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please select an image file.")

    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum upload size is {MAX_IMAGE_MB:g} MB.",
        )

    try:
        with Image.open(BytesIO(raw)) as source_image:
            if source_image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format. Please upload JPG, PNG or WEBP.",
                )
            source_image.load()
            image = source_image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="The image could not be opened.") from exc

    try:
        output = await run_in_threadpool(
            detector.predict,
            image,
            confidence,
            iou,
            image_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Detection failed. Check the backend logs.") from exc
    finally:
        await file.close()

    return {"filename": Path(file.filename).name, **output}


MAX_VIDEO_MB = float(os.getenv("MAX_VIDEO_MB", "200"))
MAX_VIDEO_BYTES = int(MAX_VIDEO_MB * 1024 * 1024)
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def _process_video(input_path: str, output_path: str, confidence: float, iou: float, image_size: int) -> dict[str, Any]:
    import cv2

    if not detector.is_loaded:
        detector.load_model()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened. Please try another video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video dimensions could not be read.")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video. Check available disk space.")
    frames_processed = 0
    frames_with_plates = 0
    total_detections = 0
    confidence_sum = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with detector._lock:
                result = detector.model.predict(source=frame, conf=confidence, iou=iou, imgsz=image_size, device=detector.device, max_det=100, verbose=False)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                count = len(result.boxes)
                frames_with_plates += 1
                total_detections += count
                confidence_sum += float(result.boxes.conf.detach().cpu().sum().item())
            annotated = result.plot()
            writer.write(annotated)
            frames_processed += 1
    finally:
        cap.release()
        writer.release()
    if frames_processed == 0:
        raise ValueError("No readable frames were found in this video.")
    return {"frames_processed": frames_processed, "total_frames": total_frames or frames_processed, "frames_with_plates": frames_with_plates, "total_detections": total_detections, "average_confidence": round(confidence_sum / total_detections, 4) if total_detections else 0.0, "fps": round(fps, 2), "width": width, "height": height}


@app.post("/api/detect-video", tags=["Inference"])
async def detect_video(
    file: UploadFile = File(..., description="MP4, AVI, MOV or MKV video"),
    confidence: float = Form(0.35, ge=0.01, le=0.99),
    iou: float = Form(0.45, ge=0.05, le=0.95),
    image_size: int = Form(640, ge=320, le=1280),
):
    if not file.filename or Path(file.filename).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload an MP4, AVI, MOV or MKV video.")
    raw = await file.read(MAX_VIDEO_BYTES + 1)
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    if len(raw) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail=f"Video is too large. Maximum size is {MAX_VIDEO_MB:g} MB.")
    temp_dir = Path(tempfile.mkdtemp(prefix="platevision_video_"))
    input_path = temp_dir / f"input{Path(file.filename).suffix.lower()}"
    output_path = temp_dir / "platevision_annotated.mp4"
    input_path.write_bytes(raw)
    try:
        stats = await run_in_threadpool(_process_video, str(input_path), str(output_path), confidence, iou, image_size)
    except ValueError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Video inference failed")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Video detection failed. Check backend logs and available disk space.") from exc
    return FileResponse(str(output_path), media_type="video/mp4", filename="platevision_annotated.mp4", headers={f"X-PlateVision-{key.replace('_', '-')}": str(value) for key, value in stats.items()}, background=BackgroundTask(shutil.rmtree, str(temp_dir), ignore_errors=True))
