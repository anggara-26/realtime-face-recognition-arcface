"""Small image I/O and augmentation helpers, kept free of Django imports so
the BAB IV evaluation scripts (scripts/) can reuse them without bootstrapping
the web app.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_upload(raw_bytes: bytes) -> np.ndarray | None:
    """Decode raw image bytes (as read from an uploaded file) to BGR."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def decode_data_url(data_url: str) -> np.ndarray | None:
    """Decode a `data:image/...;base64,...` string (webcam snapshot) to BGR."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return decode_upload(raw)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return buf.tobytes()


# --- Test-time augmentation (BAB III.3.7 / BAB IV.4.4) ----------------------

def tta_variants(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Generate the augmented views used for one-shot test-time augmentation:
    horizontal flip, small rotations, and brightness changes."""
    variants = [image_bgr]
    variants.append(cv2.flip(image_bgr, 1))
    for angle in (-10, 10):
        variants.append(_rotate(image_bgr, angle))
    for factor in (0.8, 1.2):
        variants.append(_adjust_brightness(image_bgr, factor))
    return variants


def _rotate(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image_bgr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _adjust_brightness(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# --- CCTV-style degradations (BAB III.3.7 / BAB IV.4.2) ---------------------

def degrade_blur(image_bgr: np.ndarray, kernel: int) -> np.ndarray:
    kernel = max(1, kernel | 1)  # must be odd
    return cv2.GaussianBlur(image_bgr, (kernel, kernel), 0)


def degrade_resolution(image_bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    target_width = max(8, target_width)
    scale = target_width / w
    small = cv2.resize(image_bgr, (target_width, max(8, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_brightness(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    return _adjust_brightness(image_bgr, factor)


def degrade_rotation(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    return _rotate(image_bgr, angle_deg)


def degrade_occlusion(image_bgr: np.ndarray, bottom_fraction: float) -> np.ndarray:
    """Black out the bottom `bottom_fraction` of the image (mask/scarf-like occlusion)."""
    out = image_bgr.copy()
    h = out.shape[0]
    cut = int(h * (1 - bottom_fraction))
    out[cut:, :, :] = 0
    return out
