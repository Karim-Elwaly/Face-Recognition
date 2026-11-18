"""
Unit & Integration Tests for Face Recognition System.
Validates CPU-first execution for Detector, Recognizer, Database, and Pipeline.
"""
import unittest
import tempfile
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import torch

from src.config import DEFAULT_DEVICE, EMBEDDING_DIM
from src.detector import YOLOFaceDetector
from src.recognizer import FaceRecognizer
from src.database import FaceDatabase
from src.pipeline import FaceRecognitionPipeline


class TestFaceRecognitionCPU(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_01_yolo_detector_cpu(self):
        """Test YOLO-Face detector initialization and inference on CPU."""
        detector = YOLOFaceDetector(device=self.device)
        self.assertEqual(detector.device, "cpu")

        # Test dummy frame inference
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 0)

    def test_02_facenet_recognizer_cpu(self):
        """Test FaceNet feature extractor on CPU."""
        recognizer = FaceRecognizer(device=self.device)
        self.assertEqual(recognizer.device, "cpu")

        # Single face crop embedding
        dummy_crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        emb = recognizer.encode(dummy_crop)
        self.assertEqual(emb.shape, (EMBEDDING_DIM,))
        self.assertEqual(emb.dtype, np.float32)

        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)

        # Batch crops embedding
        crops = [dummy_crop, dummy_crop.copy()]
        batch_embs = recognizer.encode_batch(crops)
        self.assertEqual(batch_embs.shape, (2, EMBEDDING_DIM))
        for row in batch_embs:
            self.assertAlmostEqual(np.linalg.norm(row), 1.0, places=4)

    def test_03_database_matching(self):
        """Test FaceDatabase enrollment, vector similarity matching, and persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_db.pt"
            db = FaceDatabase(db_path=db_file)

            # Generate synthetic unit embeddings
            emb_alice = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            emb_alice = emb_alice / np.linalg.norm(emb_alice)

            emb_bob = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            emb_bob = emb_bob / np.linalg.norm(emb_bob)

            # Enroll
            db.enroll("Alice", emb_alice)
            db.enroll("Bob", emb_bob)
            self.assertEqual(len(db.identities), 2)

            # Query with exact Alice embedding
            name, score = db.match(emb_alice, threshold=0.60)
            self.assertEqual(name, "Alice")
            self.assertGreaterEqual(score, 0.99)

            # Query with slightly perturbed Alice embedding (small noise in 512-D space)
            noise = np.random.normal(0, 0.01, EMBEDDING_DIM).astype(np.float32)
            perturbed = emb_alice + noise
            perturbed = perturbed / np.linalg.norm(perturbed)
            name, score = db.match(perturbed, threshold=0.60)
            self.assertEqual(name, "Alice")
            self.assertGreater(score, 0.85)

            # Query with an unknown orthogonal embedding
            orthogonal = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            orthogonal = orthogonal / np.linalg.norm(orthogonal)
            # Ensure similarity with Alice and Bob is low
            if np.dot(emb_alice, orthogonal) < 0.5 and np.dot(emb_bob, orthogonal) < 0.5:
                name, score = db.match(orthogonal, threshold=0.75)
                self.assertEqual(name, "Unknown")

            # Test persistence save and reload
            db.save()
            reloaded_db = FaceDatabase(db_path=db_file)
            self.assertEqual(len(reloaded_db.identities), 2)
            name_reloaded, _ = reloaded_db.match(emb_bob, threshold=0.60)
            self.assertEqual(name_reloaded, "Bob")

    def test_04_pipeline_cpu_execution(self):
        """Test full FaceRecognitionPipeline end-to-end on CPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "pipeline_db.pt"
            pipeline = FaceRecognitionPipeline(device=self.device, db_path=db_file)

            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            annotated, results = pipeline.process_frame(dummy_frame, draw=True)
            self.assertEqual(annotated.shape, dummy_frame.shape)
            self.assertEqual(len(results), 0)

    def test_05_channel_robustness(self):
        """Test channel handling: 1-channel grayscale and 4-channel BGRA."""
        recognizer = FaceRecognizer(device=self.device)
        
        # Grayscale (2D)
        gray_crop = np.random.randint(0, 255, (160, 160), dtype=np.uint8)
        emb_gray = recognizer.encode(gray_crop)
        self.assertEqual(emb_gray.shape, (EMBEDDING_DIM,))
        self.assertAlmostEqual(np.linalg.norm(emb_gray), 1.0, places=4)

        # BGRA (4-channel)
        bgra_crop = np.random.randint(0, 255, (160, 160, 4), dtype=np.uint8)
        emb_bgra = recognizer.encode(bgra_crop)
        self.assertEqual(emb_bgra.shape, (EMBEDDING_DIM,))
        self.assertAlmostEqual(np.linalg.norm(emb_bgra), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
