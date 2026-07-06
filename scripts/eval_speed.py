"""BAB IV.4.3 -- Trade-off kecepatan (FPS) vs resolusi masukan.

Usage: python scripts/eval_speed.py [--engine=arcface|dlib] [--iters=15]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.common import REPORTS_DIR, engine_from_argv, load_identities

RESOLUTIONS = [(320, 240), (480, 360), (640, 480), (800, 600)]
REALTIME_FPS = 15


def main():
    engine, engine_name = engine_from_argv()
    iters = 15
    for arg in sys.argv[1:]:
        if arg.startswith("--iters="):
            iters = int(arg.split("=", 1)[1])

    identities = load_identities(min_images=1)
    sample_image = next(iter(identities.values()))[0]

    fps_values = []
    for w, h in RESOLUTIONS:
        resized = cv2.resize(sample_image, (w, h))
        engine.get_faces(resized)  # warm-up (model init, first-call overhead)

        start = time.perf_counter()
        for _ in range(iters):
            engine.get_faces(resized)
        elapsed = time.perf_counter() - start

        fps = iters / elapsed
        fps_values.append(fps)
        print(f"{w}x{h}: {elapsed / iters * 1000:.1f} ms/frame -> {fps:.1f} FPS")

    labels = [f"{w}x{h}" for w, h in RESOLUTIONS]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, fps_values, color="#2f81f7")
    ax.axhline(REALTIME_FPS, linestyle="--", color="#e5534b", label=f"ambang real-time ({REALTIME_FPS} FPS)")
    for bar, val in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("FPS")
    ax.set_xlabel("Resolusi input")
    ax.set_title(f"Kecepatan pipeline ({engine_name}, CPU) vs resolusi")
    ax.legend()

    fig.tight_layout()
    out_path = REPORTS_DIR / f"speed_{engine_name}.png"
    fig.savefig(out_path, dpi=140)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
