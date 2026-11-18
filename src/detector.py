"""
YOLO-Face Detector Module.
Loads YOLO-Face model (akanametov/yolo-face / yolo11n-face) and performs face detection.
Defaults to CPU device.
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from src.config import get_yolo_weights_path, DEFAULT_DEVICE, DEFAULT_CONFIDENCE, DEFAULT_IOU_THRESHOLD

import numpy as np
import cv2
from ultralytics import YOLO


class YOLOFaceDetector:
    def __init__(self, model_path: Optional[str or Path] = None, device: str = DEFAULT_DEVICE):
        self.device = str(device).lower()
        self.model_path = Path(model_path) if model_path else get_yolo_weights_path()
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO-Face model weights not found at: {self.model_path}. "
                f"Please ensure yolo11n-face.pt exists."
            )
            
        print(f"[Detector] Loading YOLO-Face model from {self.model_path} on device: {self.device}")
        self.model = YOLO(str(self.model_path))
        # Note: ultralytics supports device parameter during call or to(device)
        self.model.to(self.device)
        print(f"[Detector] YOLO-Face ready on {self.device}")

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU_THRESHOLD,
        margin: float = 0.08
    ) -> List[Dict]:
        """
        Detect faces in a BGR image frame.
        
        Args:
            frame_bgr: Input BGR image (NumPy array).
            conf: Minimum confidence threshold.
            iou: NMS IoU threshold.
            margin: Extra percentage padding around detected face crop.
            
        Returns:
            List of dicts: [
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float,
                    "crop": np.ndarray (BGR crop of face),
                },
                ...
            ]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Handle grayscale or BGRA (alpha channel) images
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

        h, w = frame_bgr.shape[:2]
        results = self.model(
            frame_bgr,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                confidence = float(box.conf[0].cpu().numpy())

                # Add margin around face for better recognition embedding
                box_w = x2 - x1
                box_h = y2 - y1
                pad_x = int(box_w * margin)
                pad_y = int(box_h * margin)

                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w, x2 + pad_x)
                crop_y2 = min(h, y2 + pad_y)

                face_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "crop_bbox": (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)),
                    "confidence": confidence,
                    "crop": face_crop
                })

        return detections

