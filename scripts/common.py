"""Shared setup for the BAB IV evaluation scripts.

These scripts intentionally do NOT boot Django -- `faceid.engine` and
`faceid.imaging` are plain Python modules, so they can be exercised directly
to reproduce the paper's experiments (BAB III.3.7 / BAB IV) from the command
line.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "demo"
REPORTS_DIR = ROOT_DIR / "reports"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Kept in sync with config.settings.FACE_MATCH_THRESHOLD; duplicated here so
# these scripts stay Django-free.
ENGINE_THRESHOLDS = {
    "arcface": 0.38,
    "dlib": 0.84,
}

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from faceid.engine import get_engine  # noqa: E402


def load_identities(min_images: int = 2) -> dict[str, list[np.ndarray]]:
    """Load data/demo/<name>/*.jpg into {name: [bgr images]}."""
    identities: dict[str, list[np.ndarray]] = {}
    if not DATA_DIR.exists():
        raise SystemExit(
            f"No demo dataset at {DATA_DIR}. Run `python scripts/prepare_demo_data.py` first."
        )
    for person_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        images = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        if len(images) >= min_images:
            identities[person_dir.name] = images
    if not identities:
        raise SystemExit(
            f"No identity in {DATA_DIR} has >= {min_images} images. "
            "Run scripts/prepare_demo_data.py first."
        )
    return identities


def engine_from_argv(default: str = "arcface"):
    name = default
    for arg in sys.argv[1:]:
        if arg.startswith("--engine="):
            name = arg.split("=", 1)[1]
    print(f"[common] using face engine backend: {name}")
    return get_engine(name), name
