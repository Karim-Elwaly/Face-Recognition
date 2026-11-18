"""
FaceNet Feature Extraction & Recognition Module.
Uses InceptionResnetV1 (pretrained on VGGFace2) to extract 512-D identity embeddings.
Defaults to CPU device.
"""
from typing import List, Optional
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

from src.config import DEFAULT_DEVICE, FACE_INPUT_SIZE, EMBEDDING_DIM


class FaceRecognizer:
    def __init__(self, device: str = DEFAULT_DEVICE):
        self.device = str(device).lower()
        print(f"[Recognizer] Loading FaceNet (InceptionResnetV1, vggface2) on device: {self.device}")
        
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(FACE_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"[Recognizer] FaceNet ready on {self.device}")

    def _prepare_tensor(self, face_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Convert BGR face crop into preprocessed normalized torch Tensor."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        try:
            if face_bgr.ndim == 2:
                face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)
            elif face_bgr.ndim == 3 and face_bgr.shape[2] == 4:
                face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_BGRA2BGR)

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            return self.transform(pil_img)
        except Exception as e:
            print(f"[Recognizer] Warning: Preprocessing error: {e}")
            return None

    def encode(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extract 512-dimensional L2-normalized embedding for a face crop.
        
        Args:
            face_bgr: BGR face crop image.
            
        Returns:
            np.ndarray of shape (512,), float32.
        """
        tensor = self._prepare_tensor(face_bgr)
        if tensor is None:
            return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

        batch = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(batch)
            emb = emb.cpu().numpy().flatten().astype(np.float32)

        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb

    def encode_batch(self, face_crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a batch of face crops.
        
        Args:
            face_crops_bgr: List of BGR face crops.
            
        Returns:
            np.ndarray of shape (N, 512), float32.
        """
        if not face_crops_bgr:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        valid_tensors = []
        valid_indices = []

        for idx, crop in enumerate(face_crops_bgr):
            t = self._prepare_tensor(crop)
            if t is not None:
                valid_tensors.append(t)
                valid_indices.append(idx)

        total = len(face_crops_bgr)
        output = np.zeros((total, EMBEDDING_DIM), dtype=np.float32)

        if not valid_tensors:
            return output

        batch = torch.stack(valid_tensors).to(self.device)
        with torch.no_grad():
            embs = self.model(batch).cpu().numpy().astype(np.float32)

        # Normalize each row
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        embs = embs / norms

        for idx, mapped_emb in zip(valid_indices, embs):
            output[idx] = mapped_emb

        return output

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two unit vectors.
        Returns float between -1.0 and 1.0 (typically 0 to 1 for faces).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

