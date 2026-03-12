"""Pre-extract bubbles and boundaries from MSD graphs to speed up training."""
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.bubble_extractor import extract_bubbles, extract_boundary


def preprocess_split(msd_path: str, split: str, out_dir: str):
    graph_dir = os.path.join(msd_path, "modified-swiss-dwellings-v2", split, "graph_out")
    struct_dir = os.path.join(msd_path, "modified-swiss-dwellings-v2", split, "struct_in")

    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    graph_files = sorted(glob.glob(os.path.join(graph_dir, "*.pickle")))
    print(f"{split}: {len(graph_files)} files")

    skipped = 0
    for i, gf in enumerate(graph_files):
        base = os.path.splitext(os.path.basename(gf))[0]
        out_path = os.path.join(split_out, base + ".npz")

        # Skip if already processed
        if os.path.exists(out_path):
            continue

        with open(gf, "rb") as f:
            G = pickle.load(f)

        bubbles = extract_bubbles(G)
        if len(bubbles) == 0:
            skipped += 1
            continue

        # Extract boundary
        struct_path = os.path.join(struct_dir, base + ".npy")
        if os.path.exists(struct_path):
            struct_in = np.load(struct_path)
            boundary = extract_boundary(struct_in)
        else:
            boundary = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)

        # Store as npz: bubbles as separate arrays for efficiency
        cx = np.array([b[0] for b in bubbles], dtype=np.float32)
        cy = np.array([b[1] for b in bubbles], dtype=np.float32)
        radii = np.array([b[2] for b in bubbles], dtype=np.float32)
        types = np.array([b[3] for b in bubbles], dtype=np.int32)

        np.savez(
            out_path,
            cx=cx, cy=cy, radii=radii, types=types,
            boundary=boundary.astype(np.float32),
        )

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(graph_files)} done")

    print(f"  Done. Skipped {skipped} empty graphs.")


if __name__ == "__main__":
    msd_path = "/Data/amine.chraibi/msd"
    out_dir = "/Data/amine.chraibi/msd_preprocessed"
    os.makedirs(out_dir, exist_ok=True)

    preprocess_split(msd_path, "train", out_dir)
    preprocess_split(msd_path, "test", out_dir)

    # Print size
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_dir)
        for f in fns
    )
    print(f"\nTotal cache size: {total / 1024 / 1024:.1f} MB")
