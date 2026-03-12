"""Evaluation metrics for generated bubble layouts."""

from __future__ import annotations

import itertools
import math
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

from src.data.constraints import Constraint, ConstraintType


def constraint_satisfaction_rate(
    bubbles: List[Tuple[float, float, float, int]],
    constraints: List[Constraint],
    adjacency_threshold: float = 0.15,
) -> float:
    """Compute the fraction of constraints satisfied by the given bubble layout.

    Parameters
    ----------
    bubbles : list of (cx, cy, radius, room_type)
    constraints : list of Constraint
    adjacency_threshold : float
        Extra distance tolerance for adjacency checks.

    Returns
    -------
    float
        Fraction of constraints satisfied (0.0 to 1.0).
    """
    if not constraints:
        return 1.0

    satisfied = 0
    for c in constraints:
        if _check_constraint(bubbles, c, adjacency_threshold):
            satisfied += 1

    return satisfied / len(constraints)


def _check_constraint(
    bubbles: List[Tuple[float, float, float, int]],
    c: Constraint,
    adjacency_threshold: float,
) -> bool:
    """Check whether a single constraint is satisfied."""
    types = [b[3] for b in bubbles]

    if c.type == ConstraintType.MUST_EXIST:
        return c.room_type_a in types

    elif c.type == ConstraintType.MIN_COUNT:
        count = sum(1 for t in types if t == c.room_type_a)
        return count >= c.value

    elif c.type == ConstraintType.MAX_COUNT:
        count = sum(1 for t in types if t == c.room_type_a)
        return count <= c.value

    elif c.type == ConstraintType.MIN_AREA:
        for cx, cy, r, rt in bubbles:
            if rt == c.room_type_a:
                area = math.pi * r * r
                if area < c.value:
                    return False
        # If no rooms of this type, treat as vacuously true
        return True

    elif c.type == ConstraintType.MAX_AREA:
        for cx, cy, r, rt in bubbles:
            if rt == c.room_type_a:
                area = math.pi * r * r
                if area > c.value:
                    return False
        return True

    elif c.type == ConstraintType.MIN_DISTANCE:
        rooms_a = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_a]
        rooms_b = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_b]
        if c.room_type_a == c.room_type_b:
            for (ax, ay, _), (bx, by, __) in itertools.combinations(rooms_a, 2):
                dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                if dist < c.value:
                    return False
        else:
            for ax, ay, _ in rooms_a:
                for bx, by, __ in rooms_b:
                    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                    if dist < c.value:
                        return False
        return True

    elif c.type == ConstraintType.MAX_DISTANCE:
        rooms_a = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_a]
        rooms_b = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_b]
        if c.room_type_a == c.room_type_b:
            for (ax, ay, _), (bx, by, __) in itertools.combinations(rooms_a, 2):
                dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                if dist > c.value:
                    return False
        else:
            for ax, ay, _ in rooms_a:
                for bx, by, __ in rooms_b:
                    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                    if dist > c.value:
                        return False
        return True

    elif c.type == ConstraintType.REQUIRED_ADJACENCY:
        rooms_a = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_a]
        rooms_b = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_b]
        if c.room_type_a == c.room_type_b:
            for (ax, ay, ar), (bx, by, br) in itertools.combinations(rooms_a, 2):
                dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                if dist <= ar + br + adjacency_threshold:
                    return True
        else:
            for ax, ay, ar in rooms_a:
                for bx, by, br in rooms_b:
                    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                    if dist <= ar + br + adjacency_threshold:
                        return True
        return False

    elif c.type == ConstraintType.FORBIDDEN_ADJACENCY:
        rooms_a = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_a]
        rooms_b = [(cx, cy, r) for cx, cy, r, rt in bubbles if rt == c.room_type_b]
        if c.room_type_a == c.room_type_b:
            for (ax, ay, ar), (bx, by, br) in itertools.combinations(rooms_a, 2):
                dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                if dist <= ar + br + adjacency_threshold:
                    return False
        else:
            for ax, ay, ar in rooms_a:
                for bx, by, br in rooms_b:
                    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                    if dist <= ar + br + adjacency_threshold:
                        return False
        return True

    elif c.type == ConstraintType.BOUNDARY_CONTACT:
        # Room must be near the edge of the [0, 1] bounding box
        for cx, cy, r, rt in bubbles:
            if rt == c.room_type_a:
                near_left = abs(cx - r) < r * 0.3
                near_right = abs(cx + r - 1.0) < r * 0.3
                near_bottom = abs(cy - r) < r * 0.3
                near_top = abs(cy + r - 1.0) < r * 0.3
                if near_left or near_right or near_bottom or near_top:
                    return True
        return False

    elif c.type == ConstraintType.MIN_ROOM_RADIUS:
        for cx, cy, r, rt in bubbles:
            if rt == c.room_type_a:
                if r < c.value:
                    return False
        return True

    # Unknown constraint type -- treat as satisfied
    return True


def boundary_coverage(
    bubbles: List[Tuple[float, float, float, int]],
    boundary: np.ndarray,
) -> float:
    """Compute the ratio of circle area inside the boundary polygon to boundary area.

    Parameters
    ----------
    bubbles : list of (cx, cy, radius, room_type)
    boundary : ndarray of shape (N, 2)
        Boundary polygon vertices.

    Returns
    -------
    float
        Coverage ratio (total circle area inside boundary / boundary area).
    """
    if len(bubbles) == 0 or boundary is None or len(boundary) < 3:
        return 0.0

    poly = Polygon(boundary)
    if not poly.is_valid or poly.area < 1e-12:
        return 0.0

    total_inside = 0.0
    for cx, cy, r, _rt in bubbles:
        circle = Point(cx, cy).buffer(r, resolution=64)
        intersection = poly.intersection(circle)
        total_inside += intersection.area

    return total_inside / poly.area
