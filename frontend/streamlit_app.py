from __future__ import annotations

import base64
import hashlib
import io
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PlateVision | License Plate Detection",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_DEFAULT = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    :root { --pv-ink:#14233b; --pv-muted:#69778c; --pv-blue:#2563eb; --pv-line:#e5eaf2; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#f6f8fc 0%,#f9fbfe 48%,#f5f7fb 100%); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#101d32; border-right:1px solid #1e304d; }
    [data-testid="stSidebar"] * { color:#eaf0fa !important; }
    [data-testid="stSidebar"] .stCaption { color:#a8b7cc !important; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#dce7f6 !important; }
    [data-testid="stSidebar"] input { color:#14233b !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color:#14233b !important; }
    .block-container { padding-top:2.1rem; padding-bottom:3rem; max-width:1480px; }
    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:26px; }
    .brand-mark { width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;
      background:linear-gradient(135deg,#3b82f6,#1d4ed8);color:white;font-size:22px;box-shadow:0 8px 18px #2563eb35; }
    .brand-name { color:#f8fbff;font-weight:800;letter-spacing:.2px;font-size:19px; }
    .brand-caption { color:#9aacc4;font-size:11px;letter-spacing:1.6px;text-transform:uppercase;margin-top:1px; }
    .hero { position:relative; overflow:hidden; padding:30px 34px; border-radius:22px;
      background:radial-gradient(circle at 90% 10%,#3262a9 0,transparent 32%),linear-gradient(120deg,#10233f 0%,#173a68 58%,#1b4c89 100%);
      box-shadow:0 18px 42px #19385b20; margin-bottom:25px; }
    .hero-kicker { color:#9ec5ff; font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
    .hero h1 { color:white;font-size:clamp(27px,3.1vw,39px);line-height:1.16;font-weight:800;margin:10px 0 9px; }
    .hero p { color:#c7d8ee;font-size:14px;max-width:660px;margin:0;line-height:1.65; }
    .hero-chip { position:absolute;right:30px;top:30px;border:1px solid #ffffff30;background:#ffffff12;color:#e8f2ff;
      padding:8px 12px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:1px; }
    .section-title { color:var(--pv-ink);font-size:17px;font-weight:800;margin:7px 0 3px; }
    .section-sub { color:var(--pv-muted);font-size:12px;margin-bottom:14px; }
    .panel { background:white;border:1px solid var(--pv-line);border-radius:17px;padding:17px 19px;box-shadow:0 5px 18px #172b4d08; }
    .panel-label { color:#78869a;font-size:10px;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:8px; }
    .panel-value { color:#172b49;font-size:23px;font-weight:800;line-height:1.2; }
    .panel-note { color:#8a97a9;font-size:11px;margin-top:5px; }
    .status-pill { display:inline-flex;align-items:center;gap:7px;padding:6px 10px;border-radius:999px;font-size:11px;font-weight:700; }
    .status-ok { background:#e8f8ef;color:#187547; }
    .status-down { background:#fff0ef;color:#bd3b31; }
    .status-dot { width:7px;height:7px;border-radius:50%;background:currentColor; }
    .empty-state { background:white;border:1px dashed #cbd5e1;border-radius:18px;padding:36px 22px;text-align:center; }
    .empty-icon { width:64px;height:64px;border-radius:18px;margin:0 auto 14px;background:#eef4ff;display:flex;align-items:center;justify-content:center;font-size:29px; }
    .empty-title { color:#20334f;font-size:17px;font-weight:800;margin-bottom:6px; }
    .empty-copy { color:#7b899b;font-size:12px;line-height:1.7;max-width:430px;margin:0 auto; }
    div.stButton > button[kind="primary"] { border:0;border-radius:11px;background:linear-gradient(110deg,#2563eb,#1d4ed8);font-weight:700;min-height:45px;box-shadow:0 7px 18px #2563eb30; }
    div.stButton > button[kind="primary"]:hover { background:linear-gradient(110deg,#1d4ed8,#1e40af);border:0; }
    .stDownloadButton > button { border-radius:10px;font-weight:700; }
    [data-testid="stFileUploader"] section { background:#f8faff;border:1px dashed #b8c8e0;border-radius:14px; }
    [data-testid="stDataFrame"] { border:1px solid #e7ecf3;border-radius:12px;overflow:hidden; }
    .footer-note { color:#98a4b4;font-size:10px;text-align:center;margin-top:35px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "image_key" not in st.session_state:
    st.session_state.image_key = None

with st.sidebar:
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">⌖</div><div><div class="brand-name">PlateVision</div><div class="brand-caption">AI Detection Studio</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='color:#a9bad1;font-size:10px;font-weight:800;letter-spacing:1.5px;margin:22px 0 9px'>CONNECTION</div>", unsafe_allow_html=True)
    api_url = st.text_input("Backend API URL", value=API_DEFAULT, help="The address where your FastAPI backend is running.").rstrip("/")
    try:
        health_response = requests.get(f"{api_url}/api/health", timeout=2.5)
        health_response.raise_for_status()
        health_data = health_response.json()
        if health_data.get("model_file_exists"):
            st.markdown('<div class="status-pill status-ok"><span class="status-dot"></span> API CONNECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill status-down"><span class="status-dot"></span> MODEL NOT FOUND</div>', unsafe_allow_html=True)
    except requests.RequestException:
        st.markdown('<div class="status-pill status-down"><span class="status-dot"></span> API OFFLINE</div>', unsafe_allow_html=True)

    st.markdown("<div style='color:#a9bad1;font-size:10px;font-weight:800;letter-spacing:1.5px;margin:27px 0 9px'>DETECTION SETTINGS</div>", unsafe_allow_html=True)
    confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.35, step=0.05, help="Higher values keep only more confident detections.")
    iou = st.slider("IoU threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05, help="Controls overlap filtering for duplicate boxes.")
    image_size = st.select_slider("Inference image size", options=[320, 480, 640, 800, 1024], value=640)
    st.markdown("<div style='border-top:1px solid #2b3d58;margin:24px 0 14px'></div><div style='color:#9aacc4;font-size:11px;line-height:1.7'>Model: <b>YOLO11n</b><br>Detection class: <b>License_Plate</b><br>Default image size: <b>640 × 640</b></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="hero-chip">YOLO11 · COMPUTER VISION</div>
      <div class="hero-kicker">Intelligent image analysis</div>
      <h1>License Plate Detection</h1>
      <p>Upload a vehicle image and let the trained AI model locate license plates. Review detected regions, confidence scores and annotated results in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.04, 0.96], gap="large")
with left:
    st.markdown('<div class="section-title">01 / Upload image</div><div class="section-sub">Choose a clear vehicle image to begin detection.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image_key = f"{uploaded_file.name}:{hashlib.sha256(image_bytes).hexdigest()}"
        if st.session_state.image_key != image_key:
            st.session_state.image_key = image_key
            st.session_state.result = None
        try:
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(original_image, caption=f"Original image · {original_image.width} × {original_image.height}", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please select another file.")
            uploaded_file = None
    else:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">🚘</div><div class="empty-title">Your image workspace</div><div class="empty-copy">Drag and drop a JPG, PNG or WEBP image above, or browse files from your device. For best results, use a sharp image where the vehicle plate is visible.</div></div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="section-title">02 / Run analysis</div><div class="section-sub">Adjust the model settings in the sidebar, then start inference.</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-label">ACTIVE MODEL</div><div class="panel-value">YOLO11n</div><div class="panel-note">Custom-trained · 1 detection class · License_Plate</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel"><div class="panel-label">CURRENT PARAMETERS</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Confidence", f"{confidence:.2f}")
    p2.metric("IoU", f"{iou:.2f}")
    p3.metric("Image size", f"{image_size}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    run_detection = st.button("✦  Detect license plates", type="primary", use_container_width=True, disabled=uploaded_file is None)
    if run_detection and uploaded_file is not None:
        with st.spinner("Analyzing image with YOLO11n…"):
            try:
                response = requests.post(
                    f"{api_url}/api/detect",
                    files={"file": (uploaded_file.name, image_bytes, uploaded_file.type or "application/octet-stream")},
                    data={"confidence": confidence, "iou": iou, "image_size": image_size},
                    timeout=180,
                )
                if response.ok:
                    st.session_state.result = response.json()
                    st.success("Image analysis completed.")
                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Backend error ({response.status_code}): {detail}")
            except requests.Timeout:
                st.error("The backend took too long to respond. Try a smaller image or check the backend logs.")
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")

result: dict[str, Any] | None = st.session_state.result
if result:
    st.markdown("---")
    st.markdown('<div class="section-title">03 / Detection results</div><div class="section-sub">Review the annotated image and model predictions.</div>', unsafe_allow_html=True)
    detections = result.get("detections", [])
    avg_conf = sum(float(item["confidence"]) for item in detections) / len(detections) if detections else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="panel"><div class="panel-label">PLATES DETECTED</div><div class="panel-value">{result.get("detection_count", len(detections))}</div><div class="panel-note">License plate regions</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="panel"><div class="panel-label">AVG. CONFIDENCE</div><div class="panel-value">{avg_conf:.1%}</div><div class="panel-note">Across detected regions</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="panel"><div class="panel-label">INFERENCE TIME</div><div class="panel-value">{result.get("inference_ms", 0):.0f}<span style="font-size:13px"> ms</span></div><div class="panel-note">Model processing time</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="panel"><div class="panel-label">PROCESSING DEVICE</div><div class="panel-value">{result.get("device", "CPU")}</div><div class="panel-note">Backend inference device</div></div>', unsafe_allow_html=True)
    st.write("")
    annotated_bytes = base64.b64decode(result["annotated_image_base64"])
    annotated_image = Image.open(io.BytesIO(annotated_bytes)).convert("RGB")
    image_col, detail_col = st.columns([1.15, 0.85], gap="large")
    with image_col:
        st.markdown("**Annotated output**")
        st.image(annotated_image, use_container_width=True)
        st.download_button(
            "⬇  Download annotated image",
            data=annotated_bytes,
            file_name=f"platevision_{result.get('filename', 'result').rsplit('.', 1)[0]}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
    with detail_col:
        st.markdown("**Detection details**")
        if detections:
            table_rows = [
                {
                    "#": index + 1,
                    "Class": item["class_name"],
                    "Confidence": f"{float(item['confidence']):.1%}",
                    "X1": round(item["x1"]),
                    "Y1": round(item["y1"]),
                    "X2": round(item["x2"]),
                    "Y2": round(item["y2"]),
                }
                for index, item in enumerate(detections)
            ]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            st.caption("Coordinates are measured in pixels from the top-left corner of the original image.")
            csv_data = pd.DataFrame(table_rows).to_csv(index=False).encode("utf-8")
            st.download_button("⬇  Export detection data (CSV)", data=csv_data, file_name="platevision_detections.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("No license plates were detected. Try lowering the confidence threshold or uploading a clearer image.")



# Video detection workspace
st.markdown("---")
st.markdown('<div class="section-title">04 / Video detection</div><div class="section-sub">Upload a video to detect license plates frame by frame and download the annotated MP4.</div>', unsafe_allow_html=True)
video_col, video_settings_col = st.columns([1.05, 0.95], gap="large")
with video_col:
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], key="video_upload", help="For smoother processing, start with a short video clip.")
    if uploaded_video:
        st.video(uploaded_video.getvalue())
with video_settings_col:
    st.markdown('<div class="panel"><div class="panel-label">VIDEO INFERENCE</div><div class="panel-value">Frame-by-frame</div><div class="panel-note">The same YOLO11n model analyzes each video frame.</div></div>', unsafe_allow_html=True)
    st.caption("Large or long videos take more time and temporary disk space. Try a short MP4 first.")
    run_video = st.button("▶  Detect plates in video", type="primary", use_container_width=True, disabled=uploaded_video is None)
    if run_video and uploaded_video is not None:
        with st.spinner("Processing video frame by frame. Please keep the backend running…"):
            try:
                video_response = requests.post(
                    f"{api_url}/api/detect-video",
                    files={"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type or "video/mp4")},
                    data={"confidence": confidence, "iou": iou, "image_size": image_size},
                    timeout=(30, 1800),
                )
                if video_response.ok:
                    st.session_state.video_result = video_response.content
                    st.session_state.video_stats = {k.replace("X-Platevision-", ""): v for k, v in video_response.headers.items() if k.lower().startswith("x-platevision-")}
                    st.success("Video detection completed.")
                else:
                    try:
                        detail = video_response.json().get("detail", video_response.text)
                    except ValueError:
                        detail = video_response.text
                    st.error(f"Backend error ({video_response.status_code}): {detail}")
            except requests.Timeout:
                st.error("Video processing timed out. Try a shorter or smaller video.")
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
if st.session_state.get("video_result"):
    video_bytes = st.session_state.video_result
    stats = st.session_state.get("video_stats", {})
    st.markdown("**Processed video**")
    vm1, vm2, vm3, vm4 = st.columns(4)
    vm1.metric("Frames processed", stats.get("frames-processed", "—"))
    vm2.metric("Frames with plates", stats.get("frames-with-plates", "—"))
    vm3.metric("Total detections", stats.get("total-detections", "—"))
    vm4.metric("Average confidence", f"{float(stats.get('average-confidence', 0)):.1%}")
    st.video(video_bytes)
    st.download_button("⬇  Download annotated video (MP4)", data=video_bytes, file_name="platevision_annotated.mp4", mime="video/mp4", type="primary")

st.markdown('<div class="footer-note">PLATEVISION · AI-POWERED LICENSE PLATE LOCALIZATION · DETECTION ONLY, NOT OCR</div>', unsafe_allow_html=True)
