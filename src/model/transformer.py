"""Transformer-based denoiser for bubble diagram diffusion."""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift, broadcasting over the seq dim."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional / timestep embedding a la *Attention Is All You Need*."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) long or int tensor of timesteps.
        Returns:
            (B, dim) float tensor.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:  # odd dim: pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# Transformer block with AdaLN + cross-attention
# ---------------------------------------------------------------------------

class AdaLNCrossAttentionBlock(nn.Module):
    """One transformer layer with AdaLN-modulated self-attn, two cross-attn
    sub-layers (boundary & constraint), and an AdaLN-modulated FFN.

    4 sub-layers x 3 AdaLN params (shift, scale, gate) = 12 modulation params.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        # --- Self-attention ---
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # --- Cross-attention to boundary tokens ---
        self.norm_ca_bnd = nn.LayerNorm(d_model)
        self.cross_attn_bnd = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # --- Cross-attention to constraint tokens ---
        self.norm_ca_cst = nn.LayerNorm(d_model)
        self.cross_attn_cst = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # --- FFN ---
        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

        # --- AdaLN modulation: 12 params (shift, scale, gate) x 4 sub-layers ---
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 12 * d_model),
        )

        # Init final linear of adaLN to zeros
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        boundary: torch.Tensor,
        boundary_mask: torch.Tensor,
        constraints: torch.Tensor,
        constraint_mask: torch.Tensor,
        bubble_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) bubble slot features.
            c: (B, D) conditioning vector (timestep embedding).
            boundary: (B, Nb, D) projected boundary tokens.
            boundary_mask: (B, Nb) 1=valid, 0=pad (or None).
            constraints: (B, Nc, D) projected constraint tokens.
            constraint_mask: (B, Nc) 1=valid, 0=pad (or None).
            bubble_mask: (B, N) 1=valid, 0=pad (or None).
        """
        # Compute all 12 modulation parameters at once
        mod = self.adaLN_modulation(c)  # (B, 12*D)
        (
            shift_sa, scale_sa, gate_sa,
            shift_ca_bnd, scale_ca_bnd, gate_ca_bnd,
            shift_ca_cst, scale_ca_cst, gate_ca_cst,
            shift_ffn, scale_ffn, gate_ffn,
        ) = mod.chunk(12, dim=-1)

        # Helper: convert 1=valid masks to True=ignore for MHA key_padding_mask
        def invert_mask(mask):
            if mask is None:
                return None
            return mask == 0  # True where padded → ignored

        bubble_kpm = invert_mask(bubble_mask)
        boundary_kpm = invert_mask(boundary_mask)
        constraint_kpm = invert_mask(constraint_mask)

        # 1. Self-attention
        h = modulate(self.norm_sa(x), shift_sa, scale_sa)
        h, _ = self.self_attn(h, h, h, key_padding_mask=bubble_kpm)
        x = x + gate_sa.unsqueeze(1) * h

        # 2. Cross-attention to boundary
        h = modulate(self.norm_ca_bnd(x), shift_ca_bnd, scale_ca_bnd)
        h, _ = self.cross_attn_bnd(h, boundary, boundary, key_padding_mask=boundary_kpm)
        x = x + gate_ca_bnd.unsqueeze(1) * h

        # 3. Cross-attention to constraints
        h = modulate(self.norm_ca_cst(x), shift_ca_cst, scale_ca_cst)
        h, _ = self.cross_attn_cst(h, constraints, constraints, key_padding_mask=constraint_kpm)
        x = x + gate_ca_cst.unsqueeze(1) * h

        # 4. FFN
        h = modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
        h = self.ffn(h)
        x = x + gate_ffn.unsqueeze(1) * h

        return x


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

class BubbleDenoiser(nn.Module):
    """Transformer denoiser for bubble-diagram diffusion.

    Predicts noise given noisy slots, timestep, boundary, and constraints.
    """

    def __init__(
        self,
        slot_dim: int = 13,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        constraint_dim: int = 30,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Input projections ---
        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.boundary_proj = nn.Linear(2, d_model)
        self.constraint_proj = nn.Sequential(
            nn.Linear(constraint_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Timestep embedding ---
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            AdaLNCrossAttentionBlock(d_model, n_heads, ffn_ratio, dropout)
            for _ in range(n_layers)
        ])

        # --- Final head ---
        self.final_norm = nn.LayerNorm(d_model)
        # AdaLN for final layer: 2 params (shift, scale)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.final_linear = nn.Linear(d_model, slot_dim)

        # Zero-init final layers for training stability
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        boundary: torch.Tensor,
        boundary_mask: torch.Tensor,
        constraints: torch.Tensor,
        constraint_mask: torch.Tensor,
        bubble_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xt: (B, N, slot_dim) noisy bubble slots.
            t: (B,) integer timesteps.
            boundary: (B, Nb, 2) boundary vertex coords.
            boundary_mask: (B, Nb) 1=valid, 0=pad.
            constraints: (B, Nc, constraint_dim) constraint features.
            constraint_mask: (B, Nc) 1=valid, 0=pad.
            bubble_mask: (B, N) 1=valid, 0=pad.
        Returns:
            (B, N, slot_dim) predicted noise, zero at padded positions.
        """
        # Project inputs
        x = self.slot_proj(xt)                   # (B, N, D)
        bnd = self.boundary_proj(boundary)       # (B, Nb, D)
        cst = self.constraint_proj(constraints)  # (B, Nc, D)

        # Timestep conditioning
        c = self.time_embed(t)                   # (B, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, bnd, boundary_mask, cst, constraint_mask, bubble_mask)

        # Final head with AdaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        output = self.final_linear(x)            # (B, N, slot_dim)

        # Zero out padded positions
        output = output * bubble_mask.unsqueeze(-1)
        return output
