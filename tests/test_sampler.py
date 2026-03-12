"""Tests for BubbleSampler."""

import torch

from src.model.transformer import BubbleDenoiser
from src.model.diffusion import GaussianDiffusion
from src.inference.sampler import BubbleSampler


def test_sample_shapes():
    """Create small model and diffusion, sample with n_slots=50.
    Verify output is list of (cx, cy, r, type) tuples.
    type should be int, all values should be 4-tuples."""
    torch.manual_seed(42)

    model = BubbleDenoiser(
        slot_dim=13,
        d_model=32,
        n_layers=2,
        n_heads=4,
        constraint_dim=30,
    )
    diffusion = GaussianDiffusion(
        num_timesteps=100,
        geom_dims=3,
        type_dims=10,
    )

    sampler = BubbleSampler(model, diffusion, num_room_types=9)

    # Dummy inputs (batch=1)
    boundary = torch.randn(1, 10, 2)
    boundary_mask = torch.ones(1, 10)
    constraints = torch.randn(1, 5, 30)
    constraint_mask = torch.ones(1, 5)

    results = sampler.sample(
        boundary=boundary,
        boundary_mask=boundary_mask,
        constraints=constraints,
        constraint_mask=constraint_mask,
        n_slots=50,
        num_steps=10,
        cfg_scale=3.0,
    )

    # Output should be a list
    assert isinstance(results, list)

    # Each element should be a 4-tuple
    for item in results:
        assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
        assert len(item) == 4, f"Expected 4-tuple, got {len(item)}-tuple"

        cx, cy, r, rtype = item

        # Type should be int
        assert isinstance(rtype, int), f"Expected int type, got {type(rtype)}"

        # Room type should be in valid range (0 to 8, not 9 which is empty)
        assert 0 <= rtype < 9, f"Room type {rtype} out of range [0, 8]"

        # Geometry should be in valid ranges
        assert 0.0 <= cx <= 1.0, f"cx={cx} out of range [0, 1]"
        assert 0.0 <= cy <= 1.0, f"cy={cy} out of range [0, 1]"
        assert r >= 0.01 - 1e-6 and r <= 0.5 + 1e-6, f"r={r} out of range [0.01, 0.5]"
