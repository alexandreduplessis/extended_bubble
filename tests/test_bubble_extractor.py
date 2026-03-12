"""Tests for src.data.bubble_extractor."""

import math

import networkx as nx
import numpy as np
from shapely.geometry import Polygon, box

from src.data.bubble_extractor import extract_boundary, extract_bubbles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(nodes):
    """Build a NetworkX graph from a list of (room_type, geometry) pairs."""
    g = nx.Graph()
    for i, (rt, geom) in enumerate(nodes):
        g.add_node(
            i,
            room_type=rt,
            centroid=(geom.centroid.x, geom.centroid.y),
            geometry=geom,
        )
    return g


# ---------------------------------------------------------------------------
# extract_bubbles tests
# ---------------------------------------------------------------------------

class TestExtractBubbles:
    def test_filters_structure_and_background(self):
        """Nodes with room_type 9 (Structure) and 13 (Background) are excluded."""
        geom = box(0, 0, 2, 2)  # 2x2 square
        nodes = [
            (0, geom),   # Bedroom  -> keep
            (9, geom),   # Structure -> exclude
            (13, geom),  # Background -> exclude
            (4, geom),   # Corridor  -> keep
        ]
        bubbles = extract_bubbles(_make_graph(nodes))
        types = [b[3] for b in bubbles]
        assert types == [0, 4]

    def test_keeps_all_valid_types(self):
        """All room types 0..8 are kept."""
        geom = box(0, 0, 1, 1)
        nodes = [(rt, geom) for rt in range(9)]
        bubbles = extract_bubbles(_make_graph(nodes))
        assert len(bubbles) == 9

    def test_radius_computation(self):
        """Radius should equal sqrt(area / pi)."""
        side = 4.0
        geom = box(0, 0, side, side)
        area = side * side
        expected_radius = math.sqrt(area / math.pi)

        bubbles = extract_bubbles(_make_graph([(1, geom)]))
        assert len(bubbles) == 1
        _, _, radius, _ = bubbles[0]
        assert math.isclose(radius, expected_radius, rel_tol=1e-9)

    def test_centroid_values(self):
        """Centroid should match the shapely geometry centroid."""
        geom = box(1, 2, 5, 6)
        bubbles = extract_bubbles(_make_graph([(2, geom)]))
        cx, cy, _, _ = bubbles[0]
        assert math.isclose(cx, 3.0)
        assert math.isclose(cy, 4.0)

    def test_empty_graph(self):
        """An empty graph yields no bubbles."""
        assert extract_bubbles(nx.Graph()) == []


# ---------------------------------------------------------------------------
# extract_boundary tests
# ---------------------------------------------------------------------------

class TestExtractBoundary:
    @staticmethod
    def _make_struct_in(x_min, x_max, y_min, y_max, size=512):
        """Create a synthetic struct_in array with a rectangular structure."""
        arr = np.ones((size, size, 3), dtype=np.float64)

        # Mark a rectangular block as structure (channel 0 = 0).
        r0, r1 = size // 4, 3 * size // 4
        c0, c1 = size // 4, 3 * size // 4
        arr[r0:r1, c0:c1, 0] = 0.0

        # Fill channels 1 and 2 with linearly-spaced coordinates.
        xs = np.linspace(x_min, x_max, size)
        ys = np.linspace(y_min, y_max, size)
        xv, yv = np.meshgrid(xs, ys)
        arr[:, :, 1] = xv
        arr[:, :, 2] = yv

        return arr

    def test_returns_at_least_4_vertices(self):
        """A rectangular structure should produce >= 4 boundary vertices."""
        struct_in = self._make_struct_in(0, 10, 0, 10)
        boundary = extract_boundary(struct_in)
        assert boundary.shape[1] == 2
        assert boundary.shape[0] >= 4

    def test_boundary_within_coordinate_range(self):
        """All boundary vertices should lie within the real-world coordinate range."""
        struct_in = self._make_struct_in(0, 20, 0, 20)
        boundary = extract_boundary(struct_in)
        assert np.all(boundary[:, 0] >= 0)
        assert np.all(boundary[:, 0] <= 20)
        assert np.all(boundary[:, 1] >= 0)
        assert np.all(boundary[:, 1] <= 20)

    def test_empty_structure(self):
        """If there are no structure pixels, return an empty array."""
        arr = np.ones((512, 512, 3), dtype=np.float64)  # all non-structure
        boundary = extract_boundary(arr)
        assert boundary.shape == (0, 2)
