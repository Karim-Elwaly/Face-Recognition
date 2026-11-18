"""
FastAPI Web Application for Face Recognition.
Provides drag-and-drop face registration, image recognition, and live stream.
Defaults to CPU execution.
"""
import base64
import io
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import DEFAULT_DEVICE, DEFAULT_CONFIDENCE, DEFAULT_SIMILARITY_THRESHOLD
from src.pipeline import FaceRecognitionPipeline

app = FastAPI(title="YOLO-Face + FaceNet Recognition System")

# Active pipeline instance (CPU first by default)
CURRENT_DEVICE = DEFAULT_DEVICE
pipeline = FaceRecognitionPipeline(device=CURRENT_DEVICE)

# Camera state for streaming
cam_capture: Optional[cv2.VideoCapture] = None


@app.get("/api/status")
def get_status():
    return {
        "device": CURRENT_DEVICE,
        "enrolled_count": len(pipeline.database.identities),
        "yolo_model": str(pipeline.detector.model_path.name)
    }


@app.get("/api/identities")
def list_identities():
    return pipeline.database.list_identities()


@app.delete("/api/identities/{name}")
def delete_identity(name: str):
    success = pipeline.database.remove(name)
    if success:
        pipeline.database.save()
        return {"success": True, "message": f"Deleted {name}"}
    raise HTTPException(status_code=404, detail="Identity not found")


@app.post("/api/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    res = pipeline.enroll_image(name=name, image_input=img_bgr, save_db=True)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res.get("error", "Enrollment failed."))

    return res


@app.post("/api/recognize")
async def recognize_uploaded_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    conf: Optional[float] = Form(None)
):
    effective_threshold = threshold if threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
    effective_conf = conf if conf is not None else DEFAULT_CONFIDENCE

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    annotated, detections = pipeline.process_frame(
        img_bgr,
        threshold=effective_threshold,
        conf=effective_conf,
        draw=True
    )

    # Encode annotated image to JPEG base64
    success, buffer = cv2.imencode(".jpg", annotated)
    b64_image = base64.b64encode(buffer).decode("utf-8") if success else ""

    return {
        "faces_detected": len(detections),
        "results": detections,
        "annotated_image": f"data:image/jpeg;base64,{b64_image}"
    }


def generate_frames():
    global cam_capture
    if cam_capture is None or not cam_capture.isOpened():
        cam_capture = cv2.VideoCapture(0)

    try:
        while True:
            if cam_capture is None or not cam_capture.isOpened():
                break
            ret, frame = cam_capture.read()
            if not ret or frame is None:
                break

            annotated, _ = pipeline.process_frame(frame, draw=True)
            ret_enc, buffer = cv2.imencode('.jpg', annotated)
            if not ret_enc:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        pass


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/switch_device")
def switch_device(device: str = Form(...)):
    global pipeline, CURRENT_DEVICE
    device = device.lower().strip()
    if device not in ["cpu", "cuda"]:
        raise HTTPException(status_code=400, detail="Device must be 'cpu' or 'cuda'")
    
    print(f"[App] Switching pipeline device to: {device}")
    CURRENT_DEVICE = device
    pipeline = FaceRecognitionPipeline(device=CURRENT_DEVICE)
    return {"success": True, "device": CURRENT_DEVICE}


# Serve index.html
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = static_dir / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Face Recognition API Running. static/index.html not found.</h1>"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device (default: cpu)")
    args = parser.parse_args()

    if args.device != CURRENT_DEVICE:
        CURRENT_DEVICE = args.device
        pipeline = FaceRecognitionPipeline(device=CURRENT_DEVICE)

    print(f"Starting server on http://{args.host}:{args.port} (Device: {CURRENT_DEVICE})")
    uvicorn.run(app, host=args.host, port=args.port)
