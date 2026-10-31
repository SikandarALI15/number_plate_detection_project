# PlateVision — License Plate Detection

A professional Streamlit interface connected to a FastAPI inference service. It uses the supplied `best.pt` model to locate license-plate regions in vehicle images and return bounding boxes, confidence scores and an annotated image.

## 1. What is this model trained to do?

The uploaded checkpoint metadata identifies it as:

| Item | Checkpoint information |
|---|---|
| Model family | Ultralytics YOLO11 |
| Model size | YOLO11n (nano) |
| Task | Object detection |
| Detection class | `License_Plate` (one class) |
| Training image size | 640 × 640 |
| Configured training epochs | 50 |
| Optimizer | AdamW |
| Training data config path | `/content/License-Plate-Recognition-11/data.yaml` |
| Number of model parameters | 2,604,883 |

### Metrics stored in the checkpoint

| Validation metric | Saved value |
|---|---:|
| Precision | 98.22% |
| Recall | 94.67% |
| mAP@50 | 97.14% |
| mAP@50–95 | 69.52% |

These are the validation metrics saved in the checkpoint. They are not a guarantee of performance on every new camera, image quality, region or vehicle type. The full original dataset and its class distribution are not included in this project.

**Important:** this is a license-plate *detector*. It draws boxes around plate regions. It does not recognize or return the letters/numbers printed on a plate (OCR/ANPR text recognition would be a separate stage).

## 2. Project structure

```text
number_plate_detection_project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── main.py
│   ├── models/
│   │   └── best.pt
│   └── .env.example
├── frontend/
│   └── streamlit_app.py
├── requirements.txt
├── setup.bat
├── run_backend.bat
├── run_frontend.bat
└── README.md
```

## 3. Run on Windows

Use Python 3.11 (64-bit recommended).

1. Extract this project ZIP.
2. Double-click `setup.bat`. This creates `.venv` and installs the dependencies.
3. Double-click `run_backend.bat`. Wait until the API reports that the model has loaded.
4. Double-click `run_frontend.bat` in a second terminal/window.
5. Open the Streamlit URL shown in the terminal, normally `http://localhost:8501`.

The FastAPI interactive documentation is available at `http://127.0.0.1:8000/docs`.

### Manual run commands

```powershell
# From the project root
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Terminal 1 — backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
python -m streamlit run frontend/streamlit_app.py --server.port 8501
```

## 4. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome information |
| GET | `/api/health` | Checks model-file and load status |
| POST | `/api/detect` | Runs license-plate detection |
| GET | `/docs` | Swagger / interactive API documentation |

`POST /api/detect` accepts multipart form data:

- `file`: JPG, PNG or WEBP image
- `confidence`: 0.01–0.99 (default 0.35)
- `iou`: 0.05–0.95 (default 0.45)
- `image_size`: 320–1280 (default 640)

The JSON response includes detected boxes, class names, confidence scores, inference time, processing device and a base64-encoded annotated JPEG.

## 5. Configuration

The frontend defaults to `http://127.0.0.1:8000`. You can change the backend URL in the Streamlit sidebar or set `API_BASE_URL` before starting Streamlit.

The backend defaults to `backend/models/best.pt`. To use another checkpoint, set `MODEL_PATH` to its path. `MAX_IMAGE_MB` controls the upload limit (default 12 MB). `CORS_ORIGINS` is a comma-separated list of allowed browser origins.

## 6. CPU / GPU

The backend automatically selects CUDA when PyTorch detects a compatible NVIDIA GPU; otherwise it uses CPU. The first model load can take a little longer. CPU inference is supported but may be slower. For a dedicated CUDA setup, install a PyTorch build compatible with your NVIDIA driver and CUDA version before installing/running the app.

## 7. Troubleshooting

- **API OFFLINE in the sidebar:** start `run_backend.bat` first and check `http://127.0.0.1:8000/api/health`.
- **Model not found:** confirm `backend/models/best.pt` exists or set `MODEL_PATH`.
- **Import / version error:** use Python 3.11 in a fresh virtual environment and reinstall `requirements.txt`.
- **No plate found:** try a sharper, brighter image, or reduce the confidence threshold in the sidebar.
- **Slow prediction:** use a smaller image or run on a CUDA-enabled machine.

## License note

The checkpoint metadata records an Ultralytics AGPL-3.0 license. Review the applicable Ultralytics and model/dataset licensing terms before distributing or using this model in a proprietary/commercial product.
