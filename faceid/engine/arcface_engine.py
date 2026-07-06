"""Optimized model: RetinaFace/SCRFD detection + ArcFace 512-D embedding.

Backed by InsightFace's "buffalo_l" model pack (SCRFD detector + ArcFace
recognition head, both ONNX), which is exactly the pairing the paper cites
as the optimized configuration (Deng dkk. 2022 [1]; Guo dkk. 2022 [2]).
"""
from __future__ import annotations

import numpy as np

from .base import DetectedFace, FaceEngine, l2_normalize

_APP = None  # lazy singleton; loading the model pack is expensive


def _get_app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _APP = app
    return _APP


class ArcFaceEngine(FaceEngine):
    name = "arcface"
    embedding_dim = 512

    def get_faces(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        app = _get_app()
        faces = app.get(image_bgr)
        out = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            emb = l2_normalize(np.asarray(f.embedding, dtype=np.float32))
            out.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(f.det_score),
                    embedding=emb,
                )
            )
        return out
