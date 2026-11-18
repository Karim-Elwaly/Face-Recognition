"""
Unified Face Recognition Pipeline.
Coordinates Face Detection (YOLO-Face) + Face Recognition (FaceNet) + Gallery Database.
Runs on CPU by default.
"""
import re
import time
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import cv2

from src.config import (
    DEFAULT_DEVICE,
    DEFAULT_CONFIDENCE,
    DEFAULT_SIMILARITY_THRESHOLD,
    KNOWN_FACES_DIR,
    get_yolo_weights_path
)
from src.detector import YOLOFaceDetector
from src.recognizer import FaceRecognizer
from src.database import FaceDatabase


class FaceRecognitionPipeline:
    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        yolo_weights: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None
    ):
        self.device = str(device).lower()
        print(f"[Pipeline] Initializing pipeline on device: {self.device}")
        
        self.detector = YOLOFaceDetector(model_path=yolo_weights, device=self.device)
        self.recognizer = FaceRecognizer(device=self.device)
        self.database = FaceDatabase(db_path=db_path)
        print("[Pipeline] System fully initialized and ready.")

    def enroll_image(
        self,
        name: str,
        image_input: Union[str, Path, np.ndarray],
        conf: float = DEFAULT_CONFIDENCE,
        save_db: bool = True
    ) -> Dict:
        """
        Enroll a face from an image file path or BGR numpy array.
        
        Returns:
            Dict: status info with bounding box and success flag.
        """
        image_path_str = None
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            image_path_str = str(image_path)
            if not image_path.exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}
            frame_bgr = cv2.imread(str(image_path))
            if frame_bgr is None:
                return {"success": False, "error": f"Failed to read image: {image_path}"}
        elif isinstance(image_input, np.ndarray):
            frame_bgr = image_input
        else:
            return {"success": False, "error": "Invalid image input type."}

        # Detect faces in the image
        detections = self.detector.detect(frame_bgr, conf=conf)
        if not detections:
            return {
                "success": False,
                "error": "No face detected in the provided image. Please use a clearer photo."
            }

        # If multiple faces are detected, choose the one with the highest confidence / largest area
        best_det = max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) * d["confidence"]
        )

        # Extract embedding
        embedding = self.recognizer.encode(best_det["crop"])
        if np.linalg.norm(embedding) < 1e-6:
            return {"success": False, "error": "Failed to compute face embedding."}

        # Archive image file in data/known_faces/<person_name>/
        try:
            safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '_', name).strip()
            person_dir = KNOWN_FACES_DIR / safe_name
            person_dir.mkdir(parents=True, exist_ok=True)
            img_filename = f"{int(time.time() * 1000)}.jpg"
            saved_file_path = person_dir / img_filename
            cv2.imwrite(str(saved_file_path), frame_bgr)
            image_path_str = str(saved_file_path)
        except Exception as e:
            print(f"[Pipeline] Notice: could not archive image: {e}")

        # Add to database
        self.database.enroll(name=name, embedding=embedding, image_path=image_path_str)
        if save_db:
            self.database.save()

        return {
            "success": True,
            "name": name,
            "bbox": best_det["bbox"],
            "confidence": best_det["confidence"],
            "total_identities": len(self.database.identities)
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        conf: float = DEFAULT_CONFIDENCE,
        draw: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame: detect all faces, identify each, and annotate the frame.
        
        Args:
            frame_bgr: Input image in BGR format.
            threshold: Minimum cosine similarity to declare a match.
            conf: Face detection confidence threshold.
            draw: Whether to draw bounding boxes and name labels on frame.
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: (annotated_frame, list_of_detections)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, []

        detections = self.detector.detect(frame_bgr, conf=conf)
        if not detections:
            return frame_bgr, []

        # Extract embeddings for all detected faces in batch
        crops = [d["crop"] for d in detections]
        embeddings = self.recognizer.encode_batch(crops)

        # Match against database
        matches = self.database.match_batch(embeddings, threshold=threshold)

        results = []
        annotated_frame = frame_bgr.copy() if draw else frame_bgr

        for det, emb, (matched_name, similarity) in zip(detections, embeddings, matches):
            x1, y1, x2, y2 = det["bbox"]
            det_conf = det["confidence"]
            is_known = (matched_name != "Unknown")

            item = {
                "bbox": (x1, y1, x2, y2),
                "det_confidence": det_conf,
                "name": matched_name,
                "similarity": similarity,
                "is_known": is_known
            }
            results.append(item)

            if draw:
                # Color scheme: Green for recognized, Red/Orange for Unknown
                box_color = (0, 200, 0) if is_known else (0, 140, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                # Label text
                if is_known:
                    label = f"{matched_name} ({similarity:.2f})"
                else:
                    label = f"Unknown ({similarity:.2f})" if similarity > 0 else "Unknown"

                # Text box styling
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                label_y1 = max(0, y1 - text_h - 8)
                label_y2 = y1
                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y1),
                    (x1 + text_w + 10, label_y2),
                    box_color,
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        return annotated_frame, results

