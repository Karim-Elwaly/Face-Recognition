# Real-Time Face Recognition System

A high-performance, modular face recognition system combining:
1. **Face Detection:** [akanametov/yolo-face](https://github.com/akanametov/yolo-face) (`yolo11n-face.pt`) for ultra-fast, robust face localization.
2. **Face Recognition:** **FaceNet** (`InceptionResnetV1` pretrained on VGGFace2) for extracting 512-dimensional facial identity embeddings.
3. **Identity Gallery:** Vectorized cosine similarity database with running-average embeddings per enrolled identity.

> **Execution Device:** Default is **`cpu`** across all tools. GPU execution can be toggled at any time by specifying `--device cuda`.

---

## Environment Setup

Using your virtual environment at `D:\cpc\.venv`:
```powershell
& "D:\cpc\.venv\Scripts\activate.ps1"
```

---

## 1. Registering (Enrolling) Faces

### A. Register a single person from a photo
```powershell
& "D:\cpc\.venv\Scripts\python.exe" register.py --name "Karim ElWaly" --image "path\to\photo.jpg" --device cpu
```

### B. Batch register a directory of photos
Organize photos either by file name (`alice.jpg`, `bob.jpg`) or subfolders (`data/known_faces/Karim/1.jpg`, `data/known_faces/Karim/2.jpg`):
```powershell
& "D:\cpc\.venv\Scripts\python.exe" register.py --scan-dir "data/known_faces" --device cpu
```

### C. List enrolled identities
```powershell
& "D:\cpc\.venv\Scripts\python.exe" register.py --list
```

### D. Remove an identity
```powershell
& "D:\cpc\.venv\Scripts\python.exe" register.py --remove "Karim ElWaly"
```

---

## 2. Real-Time Camera Recognition

Run live detection and identification on your webcam:
```powershell
# Run on CPU (default)
& "D:\cpc\.venv\Scripts\python.exe" live_cam.py --device cpu --cam 0

# Run on GPU (optional acceleration)
& "D:\cpc\.venv\Scripts\python.exe" live_cam.py --device cuda --cam 0
```

### In-Camera Keyboard Controls:
- **`q`** or **`ESC`**: Exit the camera stream.
- **`s`**: Capture the current face in frame and prompt to enroll them instantly.

---

## 3. Recognizing Faces in Static Images

Analyze any static photo, identify all faces, and optionally save the annotated result:
```powershell
& "D:\cpc\.venv\Scripts\python.exe" recognize_image.py --image "path\to\test.jpg" --device cpu --save "output.jpg"
```

---

## 4. Interactive Web Interface (FastAPI)

Launch the web application to register faces, upload images for testing, and view live camera feeds through your web browser:
```powershell
& "D:\cpc\.venv\Scripts\python.exe" app.py --port 8000 --device cpu
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 5. Running Automated Tests

Run the test suite on CPU:
```powershell
& "D:\cpc\.venv\Scripts\python.exe" tests/test_pipeline.py
```

---

## Architecture & Project Structure

```
d:\cpc\Face/
├── weights/
│   └── yolo11n-face.pt          # YOLO11-Face detector weights
├── data/
│   ├── known_faces/             # Enrolled face images
│   └── face_db.pt               # Precomputed 512-D embeddings
├── src/
│   ├── __init__.py
│   ├── config.py                # System settings (defaults: device='cpu', threshold=0.60)
│   ├── detector.py              # YOLO Face detector wrapper
│   ├── recognizer.py            # FaceNet InceptionResnetV1 embedder
│   ├── database.py              # Face database manager (cosine similarity matching)
│   └── pipeline.py              # End-to-end pipeline (detection + recognition + annotation)
├── register.py                  # CLI to enroll faces
├── live_cam.py                  # Real-time webcam recognition
├── recognize_image.py           # CLI for static image recognition
├── app.py                       # FastAPI web dashboard
├── static/
│   └── index.html               # Web UI (drag-and-drop registration & feed)
├── tests/
│   └── test_pipeline.py         # Automated test suite
├── requirements.txt             # Dependency reference
├── ARCHITECTURE.md              # Detailed technical architecture & math specs
└── README.md                    # Quick-start documentation and CLI guide
```

For complete technical details, mathematical formulations, and API specifications, see [ARCHITECTURE.md](file:///d:/cpc/Face/ARCHITECTURE.md).

