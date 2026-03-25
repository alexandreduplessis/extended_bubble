"""Simplified dataset: boundary + bubbles only (no constraints)."""

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_ROOM_TYPES = 9
SLOT_DIM = 3 + NUM_ROOM_TYPES + 1  # 13


class BoundaryBubbleDataset(Dataset):
    def __init__(self, cache_dir, split="train", n_max=135, n_boundary_max=64, augment=True):
        self.n_max = n_max
        self.n_boundary_max = n_boundary_max
        self.augment = augment and (split == "train")

        cache_split = os.path.join(cache_dir, split)
        self.files = sorted(glob.glob(os.path.join(cache_split, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {cache_split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        cx, cy, radii, types = data["cx"], data["cy"], data["radii"], data["types"]
        boundary = data["boundary"]

        bubbles = [(float(x), float(y), float(r), int(t))
                   for x, y, r, t in zip(cx, cy, radii, types)]

        # Normalize to [0,1]
        bubbles, boundary = self._normalize(bubbles, boundary)

        if self.augment:
            bubbles, boundary = self._augment(bubbles, boundary)

        # Build tensors
        n_bubbles = min(len(bubbles), self.n_max)
        bubble_tensor = torch.zeros(self.n_max, SLOT_DIM)
        bubble_mask = torch.zeros(self.n_max, dtype=torch.bool)

        for i in range(n_bubbles):
            cx_i, cy_i, r_i, rt = bubbles[i]
            bubble_tensor[i, 0] = float(cx_i)
            bubble_tensor[i, 1] = float(cy_i)
            bubble_tensor[i, 2] = float(r_i)
            if 0 <= rt < NUM_ROOM_TYPES:
                bubble_tensor[i, 3 + rt] = 1.0
            else:
                bubble_tensor[i, 3 + NUM_ROOM_TYPES] = 1.0
            bubble_mask[i] = True

        # Empty slots get "empty" one-hot
        for i in range(n_bubbles, self.n_max):
            bubble_tensor[i, 3 + NUM_ROOM_TYPES] = 1.0

        n_bnd = min(len(boundary), self.n_boundary_max)
        boundary_tensor = torch.zeros(self.n_boundary_max, 2)
        boundary_mask = torch.zeros(self.n_boundary_max, dtype=torch.bool)
        if n_bnd > 0:
            boundary_tensor[:n_bnd] = torch.from_numpy(boundary[:n_bnd].astype(np.float32))
            boundary_mask[:n_bnd] = True

        return {
            "bubbles": bubble_tensor,
            "bubble_mask": bubble_mask,
            "boundary": boundary_tensor,
            "boundary_mask": boundary_mask,
        }

    @staticmethod
    def _normalize(bubbles, boundary):
        if len(boundary) == 0:
            return bubbles, boundary
        bmin = boundary.min(axis=0)
        bmax = boundary.max(axis=0)
        span = bmax - bmin
        scale = max(span[0], span[1])
        if scale < 1e-9:
            return bubbles, boundary
        boundary = (boundary - bmin) / scale
        normalized = []
        for cx, cy, r, rt in bubbles:
            normalized.append(((cx - bmin[0]) / scale, (cy - bmin[1]) / scale, r / scale, rt))
        return normalized, boundary

    @staticmethod
    def _augment(bubbles, boundary):
        k = np.random.randint(0, 4)
        flip = np.random.random() > 0.5

        def _t(x, y):
            for _ in range(k):
                x, y = 1.0 - y, x
            if flip:
                x = 1.0 - x
            return x, y

        aug = [(lambda cx, cy, r, rt: (*_t(cx, cy), r, rt))(*b) for b in bubbles]
        if len(boundary) > 0:
            new_bnd = np.array([_t(float(b[0]), float(b[1])) for b in boundary])
            boundary = new_bnd
        return aug, boundary


def collate_fn(batch):
    B = len(batch)
    keys = ["bubbles", "bubble_mask", "boundary", "boundary_mask"]
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out
