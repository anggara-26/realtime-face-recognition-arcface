"""Baseline model: dlib HOG detector + dlib ResNet embedding (128-D).

This mirrors the paper's baseline configuration (BAB II.2.2.1, BAB III.3.6).
Requires the optional `face_recognition` / `dlib` packages; these need a C++
build toolchain on most platforms, so this backend is optional and only
imported on demand (see engine/factory.py).
"""
from __future__ import annotations

import numpy as np

from .base import DetectedFace, FaceEngine, l2_normalize


class DlibEngine(FaceEngine):
    name = "dlib"
    embedding_dim = 128

    def __init__(self) -> None:
        try:
            import face_recognition  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Backend 'dlib' requires the 'face_recognition' package "
                "(pip install face_recognition dlib), which in turn needs a "
                "C++ compiler + CMake on most systems. Use the 'arcface' "
                "backend instead, or install the build toolchain."
            ) from exc
        self._fr = face_recognition

    def get_faces(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        # dlib's pybind11 bindings require a C-contiguous uint8 array; the
        # BGR->RGB channel-reversal slice below produces a negative-stride
        # view, which is *not* contiguous and fails compute_face_descriptor.
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        locations = self._fr.face_locations(rgb, model="hog")
        if not locations:
            return []
        encodings = self._fr.face_encodings(rgb, known_face_locations=locations)
        out = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            out.append(
                DetectedFace(
                    bbox=(int(left), int(top), int(right), int(bottom)),
                    det_score=1.0,  # dlib's HOG detector does not expose a score
                    embedding=l2_normalize(np.asarray(enc, dtype=np.float32)),
                )
            )
        return out
