"""BAB IV.4.2 -- Robustness terhadap degradasi citra khas CCTV.

Usage: python scripts/eval_robustness.py [--engine=arcface|dlib]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faceid import imaging
from faceid.engine import cosine_similarity
from scripts.common import ENGINE_THRESHOLDS, REPORTS_DIR, engine_from_argv, load_identities

DEGRADATIONS = {
    "Blur (kernel px)": ("blur", [1, 3, 5, 7, 9, 11]),
    "Resolusi wajah (px lebar)": ("resolution", [160, 140, 120, 100, 80, 60]),
    "Kecerahan (x)": ("brightness", [1.0, 0.9, 0.8, 0.6, 0.4, 0.2]),
    "Rotasi (derajat)": ("rotation", [0, 5, 10, 15, 20, 30]),
    "Oklusi bawah (fraksi)": ("occlusion", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
}


def apply_degradation(kind: str, image, level):
    if kind == "blur":
        return imaging.degrade_blur(image, level)
    if kind == "resolution":
        return imaging.degrade_resolution(image, level)
    if kind == "brightness":
        return imaging.degrade_brightness(image, level)
    if kind == "rotation":
        return imaging.degrade_rotation(image, level)
    if kind == "occlusion":
        return imaging.degrade_occlusion(image, level)
    raise ValueError(kind)


def main():
    engine, engine_name = engine_from_argv()
    identities = load_identities(min_images=1)
    threshold = ENGINE_THRESHOLDS.get(engine_name, 0.4)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle(f"Robustness pengenalan terhadap degradasi citra ({engine_name})")
    axes_flat = axes.flatten()

    for ax_idx, (title, (kind, levels)) in enumerate(DEGRADATIONS.items()):
        ax = axes_flat[ax_idx]
        avg_per_level = []
        for level in levels:
            sims = []
            for name, images in identities.items():
                ref_face = engine.embed_single(images[0])
                if ref_face is None:
                    continue
                degraded = apply_degradation(kind, images[0], level)
                deg_face = engine.embed_single(degraded)
                if deg_face is None:
                    continue
                sims.append(cosine_similarity(ref_face.embedding, deg_face.embedding))
            avg_per_level.append(np.mean(sims) if sims else np.nan)
            print(f"{title} | level={level}: mean cosine={avg_per_level[-1]}")

        ax.plot(levels, avg_per_level, marker="o", color="#2f81f7", label="rata-rata")
        ax.axhline(threshold, linestyle="--", color="#e5534b", label=f"ambang ({threshold})")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("cosine similarity ke acuan")
        ax.legend(fontsize=7)

    axes_flat[-1].axis("off")
    fig.tight_layout()
    out_path = REPORTS_DIR / f"robustness_{engine_name}.png"
    fig.savefig(out_path, dpi=140)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
