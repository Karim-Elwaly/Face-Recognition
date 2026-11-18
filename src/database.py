"""
Face Database Manager.
Stores, persists, and performs vector cosine similarity matching against enrolled faces.
"""
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import torch

from src.config import DB_PATH, DEFAULT_SIMILARITY_THRESHOLD, EMBEDDING_DIM


class FaceDatabase:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        # Mapping: name -> {"name": str, "embedding": np.ndarray (512,), "count": int, "images": list}
        self.identities: Dict[str, dict] = {}
        self._gallery_names: List[str] = []
        self._gallery_matrix: Optional[np.ndarray] = None  # (N, 512)
        
        self.load()

    def _rebuild_matrix(self):
        """Rebuild the fast matrix for vectorized cosine similarity."""
        self._gallery_names = list(self.identities.keys())
        if self._gallery_names:
            embs = [self.identities[name]["embedding"] for name in self._gallery_names]
            self._gallery_matrix = np.vstack(embs).astype(np.float32)
        else:
            self._gallery_matrix = None

    def enroll(self, name: str, embedding: np.ndarray, image_path: Optional[str] = None):
        """
        Enroll a face embedding for a given person name.
        If the person already exists, updates their embedding with a running average.
        """
        name = name.strip()
        if not name:
            raise ValueError("Identity name cannot be empty.")
            
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        else:
            raise ValueError("Invalid embedding (near-zero norm).")

        if name in self.identities:
            entry = self.identities[name]
            prev_emb = entry["embedding"]
            prev_count = entry["count"]
            # Weighted running average
            new_count = prev_count + 1
            new_emb = (prev_emb * prev_count + embedding) / new_count
            new_emb = new_emb / np.linalg.norm(new_emb)
            
            entry["embedding"] = new_emb.astype(np.float32)
            entry["count"] = new_count
            if image_path:
                entry["images"].append(str(image_path))
            print(f"[Database] Updated existing identity '{name}' (total photos: {new_count})")
        else:
            self.identities[name] = {
                "name": name,
                "embedding": embedding.astype(np.float32),
                "count": 1,
                "images": [str(image_path)] if image_path else []
            }
            print(f"[Database] Enrolled new identity '{name}'")

        self._rebuild_matrix()

    def remove(self, name: str) -> bool:
        """Remove a person from the database."""
        name = name.strip()
        if name in self.identities:
            del self.identities[name]
            self._rebuild_matrix()
            print(f"[Database] Removed identity '{name}'")
            return True
        return False

    def list_identities(self) -> List[dict]:
        """List all enrolled identities with summary info."""
        results = []
        for name, info in self.identities.items():
            results.append({
                "name": name,
                "count": info["count"],
                "images_count": len(info.get("images", []))
            })
        return results

    def match(
        self,
        query_embedding: np.ndarray,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> Tuple[str, float]:
        """
        Match a single query embedding against the gallery.
        
        Returns:
            Tuple[str, float]: (matched_name, similarity_score).
            If no match exceeds threshold or database is empty, returns ("Unknown", best_score).
        """
        if self._gallery_matrix is None or len(self._gallery_names) == 0:
            return "Unknown", 0.0

        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-6:
            return "Unknown", 0.0
        query_unit = (query_embedding / q_norm).astype(np.float32)

        # Dot product with all gallery embeddings
        scores = np.dot(self._gallery_matrix, query_unit)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return self._gallery_names[best_idx], best_score
        else:
            return "Unknown", best_score

    def match_batch(
        self,
        query_embeddings: np.ndarray,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> List[Tuple[str, float]]:
        """
        Vectorized match for multiple query embeddings (M, 512).
        """
        if query_embeddings is None or len(query_embeddings) == 0:
            return []

        if self._gallery_matrix is None or len(self._gallery_names) == 0:
            return [("Unknown", 0.0) for _ in range(len(query_embeddings))]

        # Normalize query embeddings row-wise
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        unit_queries = query_embeddings / norms

        # (M, 512) @ (512, N) -> (M, N)
        similarity_matrix = np.dot(unit_queries, self._gallery_matrix.T)

        results = []
        for row in similarity_matrix:
            best_idx = int(np.argmax(row))
            best_score = float(row[best_idx])
            if best_score >= threshold:
                results.append((self._gallery_names[best_idx], best_score))
            else:
                results.append(("Unknown", best_score))

        return results

    def save(self, path: Optional[Path] = None):
        """Save the database to disk."""
        target_path = Path(path) if path else self.db_path
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "identities": self.identities,
            "version": 1.0
        }
        torch.save(save_data, str(target_path))
        print(f"[Database] Saved {len(self.identities)} identities to {target_path}")

    def load(self, path: Optional[Path] = None):
        """Load the database from disk if it exists."""
        target_path = Path(path) if path else self.db_path
        if not target_path.exists():
            print(f"[Database] No existing database file found at {target_path}. Initialized empty.")
            return

        try:
            data = torch.load(str(target_path), map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "identities" in data:
                self.identities = data["identities"]
            elif isinstance(data, dict):
                self.identities = data
            self._rebuild_matrix()
            print(f"[Database] Loaded {len(self.identities)} identities from {target_path}")
        except Exception as e:
            print(f"[Database] Error loading database from {target_path}: {e}")
