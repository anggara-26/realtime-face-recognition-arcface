from __future__ import annotations

from .base import FaceEngine

_CACHE: dict[str, FaceEngine] = {}

ENGINE_CHOICES = ("arcface", "dlib")


def get_engine(name: str) -> FaceEngine:
    """Return a cached engine instance for the given backend name."""
    if name not in ENGINE_CHOICES:
        raise ValueError(f"Unknown engine '{name}', expected one of {ENGINE_CHOICES}")
    if name not in _CACHE:
        if name == "arcface":
            from .arcface_engine import ArcFaceEngine

            _CACHE[name] = ArcFaceEngine()
        elif name == "dlib":
            from .dlib_engine import DlibEngine

            _CACHE[name] = DlibEngine()
    return _CACHE[name]
