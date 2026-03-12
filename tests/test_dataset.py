"""Tests for src.data.dataset."""

import torch

from src.data.dataset import SLOT_DIM, collate_bubbles
from src.data.constraints import CONSTRAINT_DIM


def _make_sample(n_bubbles: int, n_boundary: int, n_constraints: int):
    """Create a fake dataset sample with the given sizes."""
    return {
        "bubbles": torch.randn(n_bubbles, SLOT_DIM),
        "bubble_mask": torch.ones(n_bubbles, dtype=torch.bool),
        "boundary": torch.randn(n_boundary, 2),
        "boundary_mask": torch.ones(n_boundary, dtype=torch.bool),
        "constraints": torch.randn(n_constraints, CONSTRAINT_DIM),
        "constraint_mask": torch.ones(n_constraints, dtype=torch.bool),
    }


class TestCollateShapes:
    def test_collate_shapes(self):
        """Collation pads to batch maximum and produces correct shapes."""
        s1 = _make_sample(n_bubbles=5, n_boundary=8, n_constraints=3)
        s2 = _make_sample(n_bubbles=10, n_boundary=4, n_constraints=7)
        batch = collate_bubbles([s1, s2])

        assert batch["bubbles"].shape == (2, 10, SLOT_DIM)
        assert batch["bubble_mask"].shape == (2, 10)
        assert batch["boundary"].shape == (2, 8, 2)
        assert batch["boundary_mask"].shape == (2, 8)
        assert batch["constraints"].shape == (2, 7, CONSTRAINT_DIM)
        assert batch["constraint_mask"].shape == (2, 7)

    def test_collate_masks_correct(self):
        """Masks are True for real entries and False for padding."""
        s1 = _make_sample(n_bubbles=3, n_boundary=2, n_constraints=1)
        s2 = _make_sample(n_bubbles=6, n_boundary=5, n_constraints=4)
        batch = collate_bubbles([s1, s2])

        # Sample 1 has 3 bubbles, padded to 6
        assert batch["bubble_mask"][0, :3].all()
        assert not batch["bubble_mask"][0, 3:].any()

        # Sample 2 has 6 bubbles, no padding
        assert batch["bubble_mask"][1, :6].all()

        # Boundary: sample 1 has 2, padded to 5
        assert batch["boundary_mask"][0, :2].all()
        assert not batch["boundary_mask"][0, 2:].any()

    def test_collate_preserves_values(self):
        """Original tensor values are preserved after collation."""
        s1 = _make_sample(n_bubbles=4, n_boundary=3, n_constraints=2)
        s2 = _make_sample(n_bubbles=2, n_boundary=6, n_constraints=5)
        batch = collate_bubbles([s1, s2])

        # Check that original values of s1 bubbles are preserved
        assert torch.allclose(batch["bubbles"][0, :4], s1["bubbles"])
        # Padding should be zeros
        assert torch.allclose(
            batch["bubbles"][1, 2:], torch.zeros(2, SLOT_DIM)
        )

    def test_collate_dtypes(self):
        """Collated tensors have the correct dtypes."""
        s1 = _make_sample(n_bubbles=2, n_boundary=3, n_constraints=1)
        batch = collate_bubbles([s1])

        assert batch["bubbles"].dtype == torch.float32
        assert batch["bubble_mask"].dtype == torch.bool
        assert batch["boundary"].dtype == torch.float32
        assert batch["boundary_mask"].dtype == torch.bool
        assert batch["constraints"].dtype == torch.float32
        assert batch["constraint_mask"].dtype == torch.bool

    def test_slot_dim_is_13(self):
        """SLOT_DIM should be 13 (3 coords + 10 type one-hot)."""
        assert SLOT_DIM == 13

    def test_constraint_dim_is_30(self):
        """CONSTRAINT_DIM should be 30."""
        assert CONSTRAINT_DIM == 30
