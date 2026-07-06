"""BAB IV.4.4 -- Studi one-shot: 1 foto vs 1 foto+TTA vs 3 foto (top-1 accuracy).

Usage: python scripts/eval_oneshot_tta.py [--engine=arcface|dlib]
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
from faceid.engine import l2_normalize
from scripts.common import REPORTS_DIR, engine_from_argv, load_identities


def enroll_one_shot(engine, image):
    face = engine.embed_single(image)
    return face.embedding if face else None


def enroll_tta(engine, image):
    embs = []
    for variant in imaging.tta_variants(image):
        face = engine.embed_single(variant)
        if face is not None:
            embs.append(face.embedding)
    return l2_normalize(np.mean(np.stack(embs), axis=0)) if embs else None


def enroll_multi(engine, images):
    embs = []
    for img in images:
        face = engine.embed_single(img)
        if face is not None:
            embs.append(face.embedding)
    return l2_normalize(np.mean(np.stack(embs), axis=0)) if embs else None


def top1_accuracy(engine, gallery: dict[str, np.ndarray], queries: dict[str, list]) -> float:
    names = list(gallery.keys())
    matrix = np.stack([gallery[n] for n in names])
    correct, total = 0, 0
    for true_name, images in queries.items():
        for img in images:
            face = engine.embed_single(img)
            if face is None:
                continue
            sims = matrix @ face.embedding
            pred = names[int(np.argmax(sims))]
            correct += int(pred == true_name)
            total += 1
    return correct / total if total else 0.0


def main():
    engine, engine_name = engine_from_argv()
    identities = load_identities(min_images=4)
    print(f"Loaded {len(identities)} identities with >=4 photos each.")

    gallery_one, gallery_tta, gallery_multi, queries = {}, {}, {}, {}
    for name, images in identities.items():
        n_source = min(3, len(images) - 1)  # keep >=1 photo for querying
        source, query_images = images[:n_source], images[n_source:]
        if not query_images:
            continue

        emb_one = enroll_one_shot(engine, source[0])
        emb_tta = enroll_tta(engine, source[0])
        emb_multi = enroll_multi(engine, source)
        if emb_one is None or emb_tta is None or emb_multi is None:
            continue

        gallery_one[name] = emb_one
        gallery_tta[name] = emb_tta
        gallery_multi[name] = emb_multi
        queries[name] = query_images

    acc_one = top1_accuracy(engine, gallery_one, queries)
    acc_tta = top1_accuracy(engine, gallery_tta, queries)
    acc_multi = top1_accuracy(engine, gallery_multi, queries)

    labels = ["1 foto\n(one-shot)", "1 foto + TTA", "3 foto\n(multi)"]
    values = [acc_one * 100, acc_tta * 100, acc_multi * 100]
    print(f"Accuracy -- 1 foto: {values[0]:.1f}%  1 foto+TTA: {values[1]:.1f}%  multi-foto: {values[2]:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#e57373", "#f6c343", "#2ea043"])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Pengaruh strategi enrollment ({engine_name} / data=demo)")

    fig.tight_layout()
    out_path = REPORTS_DIR / f"oneshot_tta_{engine_name}.png"
    fig.savefig(out_path, dpi=140)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
