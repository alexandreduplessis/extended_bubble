"""Constraint encoding and procedural generation for floor plan conditioning."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enum & dataclass
# ---------------------------------------------------------------------------

class ConstraintType(IntEnum):
    MIN_AREA = 0
    MAX_AREA = 1
    MIN_COUNT = 2
    MAX_COUNT = 3
    MIN_DISTANCE = 4
    MAX_DISTANCE = 5
    MUST_EXIST = 6
    FORBIDDEN_ADJACENCY = 7
    REQUIRED_ADJACENCY = 8
    BOUNDARY_CONTACT = 9
    MIN_ROOM_RADIUS = 10


@dataclass
class Constraint:
    type: ConstraintType
    room_type_a: int = -1
    room_type_b: int = -1
    value: float = 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CONSTRAINT_TYPES: int = 11
NUM_ROOM_TYPES: int = 9
CONSTRAINT_DIM: int = 30  # 11 + 9 + 9 + 1


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_constraint(c: Constraint) -> np.ndarray:
    """Encode a single constraint as a fixed-size vector of shape (30,).

    Layout: [type_onehot(11), room_type_a_onehot(9), room_type_b_onehot(9), value(1)]
    One-hot slots are all-zero when the corresponding field is -1 (unused).
    """
    vec = np.zeros(CONSTRAINT_DIM, dtype=np.float32)

    # type one-hot (positions 0..10)
    vec[int(c.type)] = 1.0

    # room_type_a one-hot (positions 11..19)
    if 0 <= c.room_type_a < NUM_ROOM_TYPES:
        vec[NUM_CONSTRAINT_TYPES + c.room_type_a] = 1.0

    # room_type_b one-hot (positions 20..28)
    if 0 <= c.room_type_b < NUM_ROOM_TYPES:
        vec[NUM_CONSTRAINT_TYPES + NUM_ROOM_TYPES + c.room_type_b] = 1.0

    # value (position 29)
    vec[-1] = c.value

    return vec


# ---------------------------------------------------------------------------
# Procedural generation
# ---------------------------------------------------------------------------

def generate_constraints_from_bubbles(
    bubbles: List[Tuple[float, float, float, int]],
    boundary: np.ndarray,
    n_sample: int = 100,
    seed: Optional[int] = None,
    relax_factor: float = 0.1,
) -> List[Constraint]:
    """Generate candidate constraints that a given bubble layout satisfies.

    Parameters
    ----------
    bubbles : list of (cx, cy, radius, room_type)
        Each bubble representing a room.
    boundary : ndarray of shape (N, 2)
        Boundary polygon vertices.
    n_sample : int
        Maximum number of constraints to return (sampled if more candidates).
    seed : int or None
        Random seed for reproducible sampling.
    relax_factor : float
        Fraction by which to relax area / radius limits.

    Returns
    -------
    list of Constraint
    """
    rng = np.random.RandomState(seed)
    candidates: List[Constraint] = []

    if len(bubbles) == 0:
        return candidates

    # Parse bubbles
    cxs = np.array([b[0] for b in bubbles])
    cys = np.array([b[1] for b in bubbles])
    radii = np.array([b[2] for b in bubbles])
    types = np.array([b[3] for b in bubbles])

    # Pre-compute per-type info
    type_counts: Counter = Counter(types.tolist())
    unique_types = sorted(type_counts.keys())

    areas_by_type: dict[int, List[float]] = {}
    radii_by_type: dict[int, List[float]] = {}
    for cx, cy, r, rt in bubbles:
        areas_by_type.setdefault(rt, []).append(np.pi * r * r)
        radii_by_type.setdefault(rt, []).append(r)

    # --- MUST_EXIST per present type ---
    for rt in unique_types:
        candidates.append(Constraint(
            type=ConstraintType.MUST_EXIST,
            room_type_a=rt,
        ))

    # --- MIN_COUNT / MAX_COUNT per type ---
    for rt in unique_types:
        cnt = type_counts[rt]
        candidates.append(Constraint(
            type=ConstraintType.MIN_COUNT,
            room_type_a=rt,
            value=float(cnt),
        ))
        # relax max count by 0-2
        relax = rng.randint(0, 3)
        candidates.append(Constraint(
            type=ConstraintType.MAX_COUNT,
            room_type_a=rt,
            value=float(cnt + relax),
        ))

    # --- MIN_AREA / MAX_AREA per type (relaxed) ---
    for rt in unique_types:
        a_list = areas_by_type[rt]
        min_a = min(a_list) * (1.0 - relax_factor)
        max_a = max(a_list) * (1.0 + relax_factor)
        candidates.append(Constraint(
            type=ConstraintType.MIN_AREA,
            room_type_a=rt,
            value=float(min_a),
        ))
        candidates.append(Constraint(
            type=ConstraintType.MAX_AREA,
            room_type_a=rt,
            value=float(max_a),
        ))

    # --- MIN_ROOM_RADIUS per type ---
    for rt in unique_types:
        min_r = min(radii_by_type[rt]) * (1.0 - relax_factor)
        candidates.append(Constraint(
            type=ConstraintType.MIN_ROOM_RADIUS,
            room_type_a=rt,
            value=float(min_r),
        ))

    # --- Pairwise distances: MAX_DISTANCE and adjacency ---
    n = len(bubbles)
    all_pairs = list(itertools.combinations(range(n), 2))

    # Sample up to 50 pairs for distance constraints
    if len(all_pairs) > 50:
        pair_indices = rng.choice(len(all_pairs), size=50, replace=False)
        sampled_pairs = [all_pairs[i] for i in pair_indices]
    else:
        sampled_pairs = all_pairs

    adjacent_type_pairs: set[Tuple[int, int]] = set()

    for i, j in sampled_pairs:
        cx_i, cy_i, r_i, rt_i = bubbles[i]
        cx_j, cy_j, r_j, rt_j = bubbles[j]
        dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)

        # MAX_DISTANCE
        candidates.append(Constraint(
            type=ConstraintType.MAX_DISTANCE,
            room_type_a=rt_i,
            room_type_b=rt_j,
            value=float(dist),
        ))

        # Check adjacency (overlap or nearly touching)
        if dist <= (r_i + r_j) * 1.2:
            tp = (min(rt_i, rt_j), max(rt_i, rt_j))
            adjacent_type_pairs.add(tp)

    # Also check all pairs for adjacency (not just sampled)
    for i, j in all_pairs:
        cx_i, cy_i, r_i, rt_i = bubbles[i]
        cx_j, cy_j, r_j, rt_j = bubbles[j]
        dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
        if dist <= (r_i + r_j) * 1.2:
            tp = (min(rt_i, rt_j), max(rt_i, rt_j))
            adjacent_type_pairs.add(tp)

    # --- REQUIRED_ADJACENCY ---
    for rt_a, rt_b in adjacent_type_pairs:
        candidates.append(Constraint(
            type=ConstraintType.REQUIRED_ADJACENCY,
            room_type_a=rt_a,
            room_type_b=rt_b,
        ))

    # --- FORBIDDEN_ADJACENCY for type pairs that are NOT adjacent ---
    all_type_pairs = set()
    for rt_a in unique_types:
        for rt_b in unique_types:
            tp = (min(rt_a, rt_b), max(rt_a, rt_b))
            all_type_pairs.add(tp)

    non_adjacent = all_type_pairs - adjacent_type_pairs
    for rt_a, rt_b in non_adjacent:
        candidates.append(Constraint(
            type=ConstraintType.FORBIDDEN_ADJACENCY,
            room_type_a=rt_a,
            room_type_b=rt_b,
        ))

    # --- BOUNDARY_CONTACT ---
    if boundary is not None and len(boundary) > 0:
        bmin = boundary.min(axis=0)
        bmax = boundary.max(axis=0)
        for cx, cy, r, rt in bubbles:
            near_left = abs(cx - r - bmin[0]) < r * 0.3
            near_right = abs(cx + r - bmax[0]) < r * 0.3
            near_bottom = abs(cy - r - bmin[1]) < r * 0.3
            near_top = abs(cy + r - bmax[1]) < r * 0.3
            if near_left or near_right or near_bottom or near_top:
                candidates.append(Constraint(
                    type=ConstraintType.BOUNDARY_CONTACT,
                    room_type_a=rt,
                ))

    # --- Sample ---
    if len(candidates) <= n_sample:
        return candidates

    indices = rng.choice(len(candidates), size=n_sample, replace=False)
    return [candidates[i] for i in indices]
