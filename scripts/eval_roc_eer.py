"""BAB IV.4.1 -- Evaluasi verifikasi: distribusi skor genuine/impostor, ROC, EER.

Usage: python scripts/eval_roc_eer.py [--engine=arcface|dlib]
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.common import REPORTS_DIR, engine_from_argv, load_identities
from faceid.engine import cosine_similarity


def compute_embeddings(engine, identities: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    embeddings: dict[str, list[np.ndarray]] = {}
    for name, images in identities.items():
        embs = []
        for img in images:
            face = engine.embed_single(img)
            if face is not None:
                embs.append(face.embedding)
        if len(embs) >= 2:
            embeddings[name] = embs
    return embeddings


def build_pairs(embeddings: dict[str, list[np.ndarray]]):
    genuine, impostor = [], []
    names = list(embeddings.keys())
    for name in names:
        for a, b in combinations(embeddings[name], 2):
            genuine.append(cosine_similarity(a, b))
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            for a in embeddings[name_a]:
                for b in embeddings[name_b]:
                    impostor.append(cosine_similarity(a, b))
    return np.array(genuine), np.array(impostor)


def roc_and_eer(genuine: np.ndarray, impostor: np.ndarray):
    thresholds = np.linspace(min(impostor.min(), genuine.min()), max(impostor.max(), genuine.max()), 500)
    far, frr = [], []
    for t in thresholds:
        far.append(np.mean(impostor >= t))  # False Acceptance Rate
        frr.append(np.mean(genuine < t))  # False Rejection Rate
    far, frr = np.array(far), np.array(frr)
    tar = 1 - frr  # True Accept Rate

    eer_idx = np.argmin(np.abs(far - frr))
    eer = (far[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    order = np.argsort(far)
    auc = np.trapezoid(tar[order], far[order])
    return far, tar, eer, eer_threshold, auc


def main():
    engine, engine_name = engine_from_argv()
    identities = load_identities(min_images=2)
    print(f"Loaded {len(identities)} identities: {list(identities.keys())}")

    embeddings = compute_embeddings(engine, identities)
    genuine, impostor = build_pairs(embeddings)
    print(f"Genuine pairs: {len(genuine)}, Impostor pairs: {len(impostor)}")

    far, tar, eer, eer_threshold, auc = roc_and_eer(genuine, impostor)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Evaluasi verifikasi wajah -- {engine_name} / data=demo")

    ax = axes[0]
    bins = np.linspace(0, 1, 40)
    ax.hist(genuine, bins=bins, alpha=0.6, label="genuine (sama)", color="#2ea043")
    ax.hist(impostor, bins=bins, alpha=0.6, label="impostor (beda)", color="#e5534b")
    ax.axvline(eer_threshold, linestyle="--", color="#2f81f7", label=f"ambang EER={eer_threshold:.2f}")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("frekuensi")
    ax.set_title("Distribusi skor genuine vs impostor")
    ax.legend(fontsize=8)

    ax = axes[1]
    order = np.argsort(far)
    ax.plot(far[order], tar[order], label=f"ROC (AUC={auc:.3f})", color="#2f81f7")
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER={eer * 100:.2f}%")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title("Kurva ROC")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = REPORTS_DIR / f"roc_eer_{engine_name}.png"
    fig.savefig(out_path, dpi=140)
    print(f"AUC={auc:.4f}  EER={eer * 100:.2f}%  (ambang={eer_threshold:.3f})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
