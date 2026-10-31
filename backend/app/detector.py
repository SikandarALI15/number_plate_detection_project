from __future__ import annotations

import base64
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


class NumberPlateDetector:
    """Loads the trained YOLO model once and serializes inference calls."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: Any | None = None
        self.device = "cpu"
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file was not found: {self.model_path}. "
                "Place best.pt inside backend/models/ or set MODEL_PATH."
            )

        # Import here so API configuration and health checks stay lightweight.
        import torch
        from ultralytics import YOLO

        self.device = "0" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Loading YOLO model from %s on %s", self.model_path, self.device)
        self.model = YOLO(str(self.model_path))
        LOGGER.info("Model loaded. Classes: %s", getattr(self.model, "names", {}))

    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.35,
        iou: float = 0.45,
        image_size: int = 640,
    ) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        started = time.perf_counter()

        # A lock protects the shared model instance when multiple requests arrive.
        with self._lock:
            result = self.model.predict(
                source=rgb_image,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                device=self.device,
                max_det=100,
                verbose=False,
            )[0]

        boxes_out: list[dict[str, Any]] = []
        if result.boxes is not None:
            names = result.names or {}
            for box in result.boxes:
                xyxy = box.xyxy[0].detach().cpu().tolist()
                class_id = int(box.cls[0].detach().cpu().item())
                score = float(box.conf[0].detach().cpu().item())
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                boxes_out.append(
                    {
                        "x1": round(float(xyxy[0]), 2),
                        "y1": round(float(xyxy[1]), 2),
                        "x2": round(float(xyxy[2]), 2),
                        "y2": round(float(xyxy[3]), 2),
                        "confidence": round(score, 4),
                        "class_id": class_id,
                        "class_name": str(class_name),
                    }
                )

        # Ultralytics returns the plotted image in BGR format.
        plotted_bgr = result.plot()
        plotted_rgb = np.ascontiguousarray(plotted_bgr[:, :, ::-1])
        annotated = Image.fromarray(plotted_rgb)
        image_buffer = io.BytesIO()
        annotated.save(image_buffer, format="JPEG", quality=92, optimize=True)
        annotated_b64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "image_width": width,
            "image_height": height,
            "detections": boxes_out,
            "detection_count": len(boxes_out),
            "inference_ms": round(elapsed_ms, 2),
            "annotated_image_base64": annotated_b64,
            "device": "GPU" if self.device != "cpu" else "CPU",
        }
