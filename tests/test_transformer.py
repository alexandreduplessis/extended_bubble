"""Tests for the transformer denoiser."""

import torch
from src.model.transformer import BubbleDenoiser


def test_forward_shapes():
    """Model with small dims — verify output shape matches input."""
    B, N, D = 2, 50, 13
    Nb, Nc = 8, 20
    constraint_dim = 30

    model = BubbleDenoiser(
        slot_dim=D,
        d_model=64,
        n_layers=2,
        n_heads=4,
        constraint_dim=constraint_dim,
        ffn_ratio=4.0,
        dropout=0.0,
    )

    xt = torch.randn(B, N, D)
    t = torch.randint(0, 1000, (B,))
    boundary = torch.randn(B, Nb, 2)
    boundary_mask = torch.ones(B, Nb)
    constraints = torch.randn(B, Nc, constraint_dim)
    constraint_mask = torch.ones(B, Nc)
    bubble_mask = torch.ones(B, N)

    pred = model(xt, t, boundary, boundary_mask, constraints, constraint_mask, bubble_mask)
    assert pred.shape == (B, N, D), f"Expected {(B, N, D)}, got {pred.shape}"


def test_output_masked():
    """Padded slots (bubble_mask=0) should have zero output."""
    B, N, D = 2, 50, 13
    Nb, Nc = 8, 20
    constraint_dim = 30

    model = BubbleDenoiser(
        slot_dim=D,
        d_model=64,
        n_layers=2,
        n_heads=4,
        constraint_dim=constraint_dim,
        ffn_ratio=4.0,
        dropout=0.0,
    )

    xt = torch.randn(B, N, D)
    t = torch.randint(0, 1000, (B,))
    boundary = torch.randn(B, Nb, 2)
    boundary_mask = torch.ones(B, Nb)
    constraints = torch.randn(B, Nc, constraint_dim)
    constraint_mask = torch.ones(B, Nc)
    bubble_mask = torch.ones(B, N)

    # Mask out slots 10: for the first sample
    bubble_mask[0, 10:] = 0

    pred = model(xt, t, boundary, boundary_mask, constraints, constraint_mask, bubble_mask)
    assert pred[0, 10:].abs().max() < 1e-6, "Padded slots should be zeroed out"
