"""Extract bubble representations (center, radius, type) from MSD floor plan graphs."""

import math
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPoint, Polygon

# Room types to keep (0..8). Exclude Structure=9 and Background=13.
_VALID_ROOM_TYPES = set(range(9))


def extract_bubbles(
    graph: nx.Graph,
) -> List[Tuple[float, float, float, int]]:
    """Extract bubble representations from an MSD NetworkX graph.

    For every node whose ``room_type`` is in 0..8 (i.e. excluding
    Structure=9 and Background=13), compute a bubble defined by the
    centroid of its Shapely geometry and an equivalent radius
    ``r = sqrt(area / pi)``.

    Parameters
    ----------
    graph : nx.Graph
        An MSD graph where each node carries ``room_type`` (int),
        ``centroid`` (tuple), and ``geometry`` (shapely Polygon) attributes.

    Returns
    -------
    list of (centroid_x, centroid_y, radius, room_type)
    """
    bubbles: List[Tuple[float, float, float, int]] = []
    for _, data in graph.nodes(data=True):
        room_type: int = data["room_type"]
        if room_type not in _VALID_ROOM_TYPES:
            continue
        geom: Polygon = data["geometry"]
        centroid = geom.centroid
        area = geom.area
        radius = math.sqrt(area / math.pi)
        bubbles.append((centroid.x, centroid.y, radius, room_type))
    return bubbles


def extract_boundary(
    struct_in: np.ndarray,
    simplify_tolerance: float = 0.1,
) -> np.ndarray:
    """Extract the outer boundary polygon of a structure from a struct_in array.

    Parameters
    ----------
    struct_in : np.ndarray
        Array of shape ``(512, 512, 3)`` where channel 0 is a binary mask
        (0 = structure, 1 = non-structure), channel 1 is the x-coordinate
        in metres, and channel 2 is the y-coordinate in metres.
    simplify_tolerance : float, optional
        Tolerance passed to ``shapely.geometry.Polygon.simplify``.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` array of boundary vertices in real-world metres.
    """
    mask = struct_in[:, :, 0] < 0.5  # True where structure exists

    # Fill holes so we get a single solid region.
    filled = ndimage.binary_fill_holes(mask)

    # Collect real-world coordinates of all structure pixels.
    ys, xs = np.where(filled)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float64)

    real_x = struct_in[ys, xs, 1]
    real_y = struct_in[ys, xs, 2]

    points = np.column_stack([real_x, real_y])

    # Build convex hull, then simplify.
    mp = MultiPoint(points)
    hull = mp.convex_hull

    if isinstance(hull, Polygon):
        simplified = hull.simplify(simplify_tolerance, preserve_topology=True)
        coords = np.array(simplified.exterior.coords)
        # exterior.coords includes the closing vertex (same as first); drop it.
        return coords[:-1]

    # Degenerate case (line or point): return whatever coordinates we have.
    return np.array(hull.coords)
