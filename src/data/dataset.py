"""PyTorch dataset for bubble diffusion training."""

import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.bubble_extractor import extract_boundary, extract_bubbles
from src.data.constraints import (
    CONSTRAINT_DIM,
    NUM_ROOM_TYPES,
    encode_constraint,
    generate_constraints_from_bubbles,
)

# 9 room types + 1 empty padding slot
_TYPE_ONEHOT_DIM = NUM_ROOM_TYPES + 1  # 10
SLOT_DIM = 3 + _TYPE_ONEHOT_DIM  # 13


class BubbleDataset(Dataset):
    """Dataset of (boundary, constraints, bubbles) extracted from MSD.

    Parameters
    ----------
    msd_path : str
        Root path, e.g. ``/Data/amine.chraibi/msd``.
    split : str
        ``"train"`` or ``"test"``.
    n_max : int
        Maximum number of bubble slots per sample.
    n_boundary_max : int
        Maximum number of boundary vertices per sample.
    min_constraints : int
        Minimum number of constraints to generate.
    max_constraints : int
        Maximum number of constraints to generate.
    augment : bool
        Whether to apply random rotation and flip (only when split is train).
    """

    def __init__(
        self,
        msd_path: str,
        split: str = "train",
        n_max: int = 200,
        n_boundary_max: int = 64,
        min_constraints: int = 10,
        max_constraints: int = 200,
        augment: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.msd_path = msd_path
        self.split = split
        self.n_max = n_max
        self.n_boundary_max = n_boundary_max
        self.min_constraints = min_constraints
        self.max_constraints = max_constraints
        self.augment = augment and (split == "train")

        # Check for preprocessed cache
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(msd_path), "msd_preprocessed")
        cache_split = os.path.join(cache_dir, split)
        self.cache_files = sorted(glob.glob(os.path.join(cache_split, "*.npz")))
        self.use_cache = len(self.cache_files) > 0

        if self.use_cache:
            # Use preprocessed files
            self.graph_files = self.cache_files  # reuse for __len__
        else:
            graph_dir = os.path.join(
                msd_path, "modified-swiss-dwellings-v2", split, "graph_out"
            )
            self.struct_dir = os.path.join(
                msd_path, "modified-swiss-dwellings-v2", split, "struct_in"
            )

            self.graph_files = sorted(glob.glob(os.path.join(graph_dir, "*.pickle")))
            if len(self.graph_files) == 0:
                raise FileNotFoundError(
                    f"No pickle files found in {graph_dir}"
                )

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_cache:
            return self._getitem_cached(idx)
        return self._getitem_raw(idx)

    def _getitem_cached(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.cache_files[idx])
        cx, cy, radii, types = data["cx"], data["cy"], data["radii"], data["types"]
        boundary = data["boundary"]

        bubbles = [
            (float(x), float(y), float(r), int(t))
            for x, y, r, t in zip(cx, cy, radii, types)
        ]
        return self._build_tensors(bubbles, boundary, normalize=True)

    def _getitem_raw(self, idx: int) -> Dict[str, torch.Tensor]:
        graph_path = self.graph_files[idx]
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        bubbles = extract_bubbles(G)

        base_name = os.path.splitext(os.path.basename(graph_path))[0]
        struct_path = os.path.join(self.struct_dir, base_name + ".npy")
        if os.path.exists(struct_path):
            struct_in = np.load(struct_path)
            boundary = extract_boundary(struct_in)
        else:
            boundary = self._boundary_from_rooms(G)

        return self._build_tensors(bubbles, boundary, normalize=True)

    def _build_tensors(
        self,
        bubbles: List[Tuple[float, float, float, int]],
        boundary: np.ndarray,
        normalize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Shared logic: normalize, augment, generate constraints, build tensors."""
        if normalize:
            bubbles, boundary = self._normalize(bubbles, boundary)

        if self.augment:
            bubbles, boundary = self._augment(bubbles, boundary)

        # Generate constraints (random subset each time)
        n_constraints = np.random.randint(
            self.min_constraints, self.max_constraints + 1
        )
        constraints_list = generate_constraints_from_bubbles(
            bubbles, boundary, n_sample=n_constraints
        )

        # Build bubble tensor (n_max, 13)
        n_bubbles = min(len(bubbles), self.n_max)
        bubble_tensor = torch.zeros(self.n_max, SLOT_DIM, dtype=torch.float32)
        bubble_mask = torch.zeros(self.n_max, dtype=torch.bool)

        for i in range(n_bubbles):
            cx, cy, r, rt = bubbles[i]
            bubble_tensor[i, 0] = float(cx)
            bubble_tensor[i, 1] = float(cy)
            bubble_tensor[i, 2] = float(r)
            rt = int(rt)
            if 0 <= rt < NUM_ROOM_TYPES:
                bubble_tensor[i, 3 + rt] = 1.0
            else:
                bubble_tensor[i, 3 + NUM_ROOM_TYPES] = 1.0  # empty slot
            bubble_mask[i] = True

        # Build boundary tensor (n_boundary_max, 2)
        n_bnd = min(len(boundary), self.n_boundary_max)
        boundary_tensor = torch.zeros(
            self.n_boundary_max, 2, dtype=torch.float32
        )
        boundary_mask = torch.zeros(self.n_boundary_max, dtype=torch.bool)

        if n_bnd > 0:
            boundary_tensor[:n_bnd] = torch.from_numpy(
                boundary[:n_bnd].astype(np.float32)
            )
            boundary_mask[:n_bnd] = True

        # Build constraint tensor (max_constraints, CONSTRAINT_DIM)
        n_con = min(len(constraints_list), self.max_constraints)
        constraint_tensor = torch.zeros(
            self.max_constraints, CONSTRAINT_DIM, dtype=torch.float32
        )
        constraint_mask = torch.zeros(self.max_constraints, dtype=torch.bool)

        for i in range(n_con):
            constraint_tensor[i] = torch.from_numpy(
                encode_constraint(constraints_list[i])
            )
            constraint_mask[i] = True

        return {
            "bubbles": bubble_tensor,
            "bubble_mask": bubble_mask,
            "boundary": boundary_tensor,
            "boundary_mask": boundary_mask,
            "constraints": constraint_tensor,
            "constraint_mask": constraint_mask,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        bubbles: List[Tuple[float, float, float, int]],
        boundary: np.ndarray,
    ) -> Tuple[List[Tuple[float, float, float, int]], np.ndarray]:
        """Normalize coordinates to [0, 1] based on boundary bounding box."""
        if len(boundary) == 0:
            return bubbles, boundary

        bmin = boundary.min(axis=0)
        bmax = boundary.max(axis=0)
        span = bmax - bmin
        scale = max(span[0], span[1])
        if scale < 1e-9:
            return bubbles, boundary

        boundary = (boundary - bmin) / scale

        normalized: List[Tuple[float, float, float, int]] = []
        for cx, cy, r, rt in bubbles:
            nx_ = (cx - bmin[0]) / scale
            ny_ = (cy - bmin[1]) / scale
            nr = r / scale
            normalized.append((nx_, ny_, nr, rt))

        return normalized, boundary

    @staticmethod
    def _augment(
        bubbles: List[Tuple[float, float, float, int]],
        boundary: np.ndarray,
    ) -> Tuple[List[Tuple[float, float, float, int]], np.ndarray]:
        """Random rotation (0/90/180/270) and horizontal flip."""
        k = np.random.randint(0, 4)
        flip = np.random.random() > 0.5

        def _transform(x: float, y: float) -> Tuple[float, float]:
            for _ in range(k):
                x, y = 1.0 - y, x
            if flip:
                x = 1.0 - x
            return x, y

        augmented_bubbles: List[Tuple[float, float, float, int]] = []
        for cx, cy, r, rt in bubbles:
            cx, cy = _transform(cx, cy)
            augmented_bubbles.append((cx, cy, r, rt))

        if len(boundary) > 0:
            new_boundary = np.empty_like(boundary)
            for i in range(len(boundary)):
                bx, by = _transform(
                    float(boundary[i, 0]), float(boundary[i, 1])
                )
                new_boundary[i] = [bx, by]
            boundary = new_boundary

        return augmented_bubbles, boundary

    @staticmethod
    def _boundary_from_rooms(G) -> np.ndarray:
        """Fallback: compute boundary as convex hull of all room geometries."""
        from shapely.geometry import MultiPoint
        from shapely.ops import unary_union

        geoms = [
            d["geometry"]
            for _, d in G.nodes(data=True)
            if "geometry" in d and d.get("room_type", -1) in range(10)
        ]
        if not geoms:
            return np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64
            )
        union = unary_union(geoms)
        hull = union.convex_hull
        if hasattr(hull, "exterior"):
            coords = np.array(hull.exterior.coords)
            return coords[:-1]
        return np.array(hull.coords)


def collate_bubbles(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Custom collate: pad variable-length tensors to batch maximum.

    Returns
    -------
    dict with keys:
        bubbles      : (B, N_max, 13)
        bubble_mask  : (B, N_max)
        boundary     : (B, Nb_max, 2)
        boundary_mask: (B, Nb_max)
        constraints  : (B, Nc_max, 30)
        constraint_mask: (B, Nc_max)
    """
    B = len(batch)

    n_max = max(s["bubbles"].shape[0] for s in batch)
    nb_max = max(s["boundary"].shape[0] for s in batch)
    nc_max = max(s["constraints"].shape[0] for s in batch)

    bubbles = torch.zeros(B, n_max, SLOT_DIM, dtype=torch.float32)
    bubble_mask = torch.zeros(B, n_max, dtype=torch.bool)
    boundary = torch.zeros(B, nb_max, 2, dtype=torch.float32)
    boundary_mask = torch.zeros(B, nb_max, dtype=torch.bool)
    constraints = torch.zeros(B, nc_max, CONSTRAINT_DIM, dtype=torch.float32)
    constraint_mask = torch.zeros(B, nc_max, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = sample["bubbles"].shape[0]
        bubbles[i, :n] = sample["bubbles"]
        bubble_mask[i, :n] = sample["bubble_mask"]

        nb = sample["boundary"].shape[0]
        boundary[i, :nb] = sample["boundary"]
        boundary_mask[i, :nb] = sample["boundary_mask"]

        nc = sample["constraints"].shape[0]
        constraints[i, :nc] = sample["constraints"]
        constraint_mask[i, :nc] = sample["constraint_mask"]

    return {
        "bubbles": bubbles,
        "bubble_mask": bubble_mask,
        "boundary": boundary,
        "boundary_mask": boundary_mask,
        "constraints": constraints,
        "constraint_mask": constraint_mask,
    }
