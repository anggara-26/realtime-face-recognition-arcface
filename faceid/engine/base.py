"""Shared interface for pluggable face-recognition backends.

Mirrors the paper's "face engine" abstraction (BAB III.3.6 Modeling): the
pipeline (detect -> align -> embed -> cosine match) stays identical while the
backend that actually produces the embedding can be swapped between the
baseline (dlib, 128-D) and the optimized model (ArcFace, 512-D).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coordinates
    det_score: float
    embedding: np.ndarray  # L2-normalized, shape (embedding_dim,)


class FaceEngine:
    """Abstract base for a detect+embed backend."""

    name: str = "base"
    embedding_dim: int = 0

    def get_faces(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect every face in a BGR image and return normalized embeddings."""
        raise NotImplementedError

    def embed_single(self, image_bgr: np.ndarray) -> DetectedFace | None:
        """Convenience helper for enrollment: return the largest detected face."""
        faces = self.get_faces(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, assumed (or forced) L2-normalized."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)
