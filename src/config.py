"""
Configuration settings for the Face Recognition System.
Defaults to CPU execution as requested.
"""
from pathlib import Path
import os
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Set YOLO config directory to project workspace to ensure sandbox write access
YOLO_DIR = BASE_DIR / ".ultralytics"
try:
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_DIR)

WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
DB_PATH = DATA_DIR / "face_db.pt"

# External default weights path if not copied locally
FALLBACK_YOLO_WEIGHTS = Path("D:/cpc/yolo11n-face.pt")
LOCAL_YOLO_WEIGHTS = WEIGHTS_DIR / "yolo11n-face.pt"

def get_yolo_weights_path() -> Path:
    """Return local weights path if exists, otherwise fallback."""
    if LOCAL_YOLO_WEIGHTS.exists():
        return LOCAL_YOLO_WEIGHTS
    if FALLBACK_YOLO_WEIGHTS.exists():
        return FALLBACK_YOLO_WEIGHTS
    return LOCAL_YOLO_WEIGHTS

# Device: 'cpu' first by default
DEFAULT_DEVICE = "cpu"

# Detection parameters
DEFAULT_CONFIDENCE = 0.50
DEFAULT_IOU_THRESHOLD = 0.45

# Recognition parameters
FACE_INPUT_SIZE = (160, 160)
DEFAULT_SIMILARITY_THRESHOLD = 0.60
EMBEDDING_DIM = 512

