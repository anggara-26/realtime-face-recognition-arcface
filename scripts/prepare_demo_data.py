"""Populate data/demo/<name>/*.jpg from a small subset of LFW (BAB III.3.1).

The paper uses LFW (via scikit-learn) as its public/secondary dataset. This
script downloads it once (cached by scikit-learn under ~/scikit_learn_data),
picks a handful of identities that have enough photos, and writes them out
as plain JPGs so the other eval scripts don't need scikit-learn at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.common import DATA_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identities", type=int, default=6, help="How many people to export.")
    parser.add_argument("--per-identity", type=int, default=8, help="Photos per person.")
    parser.add_argument(
        "--min-faces-per-person",
        type=int,
        default=20,
        help="LFW filter: only consider people with at least this many photos available.",
    )
    args = parser.parse_args()

    from sklearn.datasets import fetch_lfw_people

    print("Downloading/loading LFW subset via scikit-learn (cached after first run)...")
    # fetch_lfw_people's default slice_ crops tightly around the face
    # (~94x125px, almost no background), which is too tight for face
    # *detectors* (SCRFD/HOG expect some context around the face). Use the
    # full 250x250 deep-funneled image instead.
    lfw = fetch_lfw_people(
        min_faces_per_person=args.min_faces_per_person,
        color=True,
        resize=1.0,
        slice_=(slice(0, 250), slice(0, 250)),
    )
    names = lfw.target_names
    images = lfw.images  # (n, h, w, 3) float in [0, 255], RGB
    targets = lfw.target

    counts = np.bincount(targets)
    order = np.argsort(-counts)  # most-photographed people first -> more data to sample from
    chosen = order[: args.identities]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for person_idx in chosen:
        name = names[person_idx].replace(" ", "_")
        person_dir = DATA_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        idxs = np.where(targets == person_idx)[0][: args.per_identity]
        for i, img_idx in enumerate(idxs):
            # fetch_lfw_people returns float32 pixels normalized to [0, 1].
            rgb = np.clip(images[img_idx] * 255.0, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(person_dir / f"{i:02d}.jpg"), bgr)
        print(f"  {name}: {len(idxs)} photos -> {person_dir}")

    print(f"\nDone. Demo dataset ready at {DATA_DIR}")


if __name__ == "__main__":
    main()
