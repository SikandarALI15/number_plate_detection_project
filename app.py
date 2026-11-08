from __future__ import annotations

import base64
import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="PlateVision | License Plate Detection",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_detector(model_path: str):
    LOGGER.info(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)
    return model

# Check multiple possible locations for the model file
MODEL_PATH = Path("best.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("backend/models/best.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("models/best.pt")

# --- Helper Functions ---
def predict_image(image: Image.Image, model, confidence: float = 0.35, iou: float = 0.45, image_size: int = 640):
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    started = time.perf_counter()

    result = model.predict(
        source=rgb_image, conf=confidence, iou=iou, imgsz=image_size, max_det=100, verbose=False
    )[0]

    boxes_out = []
    if result.boxes is not None:
        names = result.names or {}
        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            score = float(box.conf[0].detach().cpu().item())
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            boxes_out.append({
                "x1": round(float(xyxy[0]), 2), "y1": round(float(xyxy[1]), 2),
                "x2": round(float(xyxy[2]), 2), "y2": round(float(xyxy[3]), 2),
                "confidence": round(score, 4), "class_id": class_id, "class_name": str(class_name),
            })

    plotted_bgr = result.plot()
    plotted_rgb = np.ascontiguousarray(plotted_bgr[:, :, ::-1])
    annotated = Image.fromarray(plotted_rgb)
    image_buffer = io.BytesIO()
    annotated.save(image_buffer, format="JPEG", quality=92, optimize=True)
    annotated_b64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "image_width": width, "image_height": height, "detections": boxes_out,
        "detection_count": len(boxes_out), "inference_ms": round(elapsed_ms, 2),
        "annotated_image_base64": annotated_b64,
        "device": "GPU" if hasattr(model, 'device') and str(model.device) != 'cpu' else "CPU",
    }

def process_video(input_path: str, output_path: str, model, confidence: float, iou: float, image_size: int):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Video open nahi ho saki. Doosri video try karein.")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Output video create nahi ho saki.")
    
    frames_processed, frames_with_plates, total_detections, confidence_sum = 0, 0, 0, 0.0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            result = model.predict(source=frame, conf=confidence, iou=iou, imgsz=image_size, max_det=100, verbose=False)[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                count = len(result.boxes)
                frames_with_plates += 1
                total_detections += count
                confidence_sum += float(result.boxes.conf.detach().cpu().sum().item())
            
            writer.write(result.plot())
            frames_processed += 1
            
            if total_frames > 0:
                progress_bar.progress(min(frames_processed / total_frames, 1.0))
                status_text.text(f"Processing frame {frames_processed} / {total_frames}")
    finally:
        cap.release()
        writer.release()
        progress_bar.empty()
        status_text.empty()
        
    if frames_processed == 0:
        raise ValueError("Video mein koi readable frames nahi mile.")
    
    return {
        "frames_processed": frames_processed, "frames_with_plates": frames_with_plates, 
        "total_detections": total_detections, 
        "average_confidence": round(confidence_sum / total_detections, 4) if total_detections else 0.0, 
    }

# --- Streamlit UI ---
st.markdown(
    """
    <style>
    .hero { position:relative; overflow:hidden; padding:30px 34px; border-radius:22px;
      background:linear-gradient(120deg,#10233f 0%,#173a68 58%,#1b4c89 100%);
      box-shadow:0 18px 42px #19385b20; margin-bottom:25px; color: white; }
    .hero h1 { font-size:clamp(27px,3.1vw,39px); font-weight:800; margin:10px 0 9px; }
    .status-pill { display:inline-flex; align-items:center; gap:7px; padding:6px 10px; border-radius:999px; font-size:11px; font-weight:700; background:#e8f8ef; color:#187547; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:currentColor; }
    .panel { background:white; border:1px solid #e5eaf2; border-radius:17px; padding:17px 19px; box-shadow:0 5px 18px #172b4d08; }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### ⌖ PlateVision AI")
    st.markdown("<div style='color:#a9bad1;font-size:10px;font-weight:800;letter-spacing:1.5px;margin:22px 0 9px'>MODEL STATUS</div>", unsafe_allow_html=True)
    try:
        model = load_detector(str(MODEL_PATH))
        st.markdown('<div class="status-pill"><span class="status-dot"></span> MODEL LOADED SUCCESSFULLY</div>', unsafe_allow_html=True)
        st.caption(f"Path: {MODEL_PATH}")
    except Exception as e:
        st.markdown('<div class="status-pill" style="background:#fff0ef;color:#bd3b31"><span class="status-dot"></span> MODEL NOT FOUND</div>', unsafe_allow_html=True)
        st.error(f"Please ensure 'best.pt' is in the repository. Error: {e}")
        st.stop()

    st.markdown("<div style='color:#a9bad1;font-size:10px;font-weight:800;letter-spacing:1.5px;margin:27px 0 9px'>SETTINGS</div>", unsafe_allow_html=True)
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    image_size = st.select_slider("Inference image size", [320, 480, 640, 800, 1024], 640)

st.markdown('<div class="hero"><h1>License Plate Detection</h1><p>Upload a vehicle image or video and let the trained AI model locate license plates directly in the cloud.</p></div>', unsafe_allow_html=True)

# --- Image Detection ---
st.markdown("### 📸 Image Detection")
uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file:
    col1, col2 = st.columns([1, 1])
    with col1:
        original_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    with col2:
        if st.button("✦ Detect license plates", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    result_data = predict_image(original_image, model, confidence, iou, image_size)
                    result_data["filename"] = uploaded_file.name
                    
                    st.success(f"Detected {result_data['detection_count']} plate(s) in {result_data['inference_ms']} ms")
                    annotated_bytes = base64.b64decode(result_data["annotated_image_base64"])
                    st.image(Image.open(io.BytesIO(annotated_bytes)), caption="Annotated Output", use_container_width=True)
                    
                    st.download_button("⬇ Download Annotated Image", data=annotated_bytes, file_name=f"result_{uploaded_file.name}", mime="image/jpeg")
                    
                    if result_data["detections"]:
                        df = pd.DataFrame(result_data["detections"])
                        st.dataframe(df[["class_name", "confidence", "x1", "y1", "x2", "y2"]], use_container_width=True)
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

# --- Video Detection ---
st.markdown("---\n### 🎥 Video Detection")
st.caption("⚠️ Note: Free cloud resources have limits. Please use short video clips (under 15 seconds) for best results.")
uploaded_video = st.file_uploader("Upload short video", type=["mp4", "avi", "mov", "mkv"], key="vid")
if uploaded_video:
    st.video(uploaded_video.getvalue())
    if st.button("▶ Detect plates in video", type="primary"):
        with st.spinner("Processing video frame by frame. This may take a minute..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_in:
                    tmp_in.write(uploaded_video.getvalue())
                    input_path = tmp_in.name
                output_path = tempfile.mktemp(suffix=".mp4")
                
                stats = process_video(input_path, output_path, model, confidence, iou, image_size)
                
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                
                st.success("Video processing completed!")
                st.video(video_bytes)
                st.download_button("⬇ Download Annotated Video", data=video_bytes, file_name="annotated_result.mp4", mime="video/mp4")
                
                st.json(stats)
                
                Path(input_path).unlink(missing_ok=True)
                Path(output_path).unlink(missing_ok=True)
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
