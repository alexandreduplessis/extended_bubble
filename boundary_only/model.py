"""Simplified transformer denoiser: boundary-only conditioning (no constraints)."""

import math
import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


def safe_kpm(mask):
    """Convert 1=valid mask to True=ignore key_padding_mask, safely handling all-masked."""
    if mask is None:
        return None
    kpm = mask == 0
    all_masked = kpm.all(dim=-1)
    if all_masked.any():
        kpm = kpm.clone()
        kpm[all_masked, 0] = False
    return kpm


class BoundaryOnlyBlock(nn.Module):
    """Transformer block: self-attn + boundary cross-attn + FFN, all with AdaLN-Zero."""

    def __init__(self, d_model, n_heads, ffn_ratio=4.0, dropout=0.0):
        super().__init__()
        # Self-attention
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention to boundary
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

        # AdaLN: 3 sub-layers x 3 params = 9
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c, boundary, boundary_kpm, bubble_kpm):
        mod = self.adaLN(c)
        s_sa, sc_sa, g_sa, s_ca, sc_ca, g_ca, s_ff, sc_ff, g_ff = mod.chunk(9, dim=-1)

        # Self-attention
        h = modulate(self.norm_sa(x), s_sa, sc_sa)
        h, _ = self.self_attn(h, h, h, key_padding_mask=bubble_kpm)
        x = x + g_sa.unsqueeze(1) * h

        # Cross-attention to boundary
        h = modulate(self.norm_ca(x), s_ca, sc_ca)
        h, _ = self.cross_attn(h, boundary, boundary, key_padding_mask=boundary_kpm)
        x = x + g_ca.unsqueeze(1) * h

        # FFN
        h = modulate(self.norm_ffn(x), s_ff, sc_ff)
        h = self.ffn(h)
        x = x + g_ff.unsqueeze(1) * h

        return x


class BoundaryDenoiser(nn.Module):
    """Boundary-only denoiser. No constraint conditioning."""

    def __init__(self, slot_dim=13, d_model=256, n_layers=8, n_heads=8,
                 ffn_ratio=4.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.boundary_proj = nn.Linear(2, d_model)

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            BoundaryOnlyBlock(d_model, n_heads, ffn_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.final_linear = nn.Linear(d_model, slot_dim)

        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, xt, t, boundary, boundary_mask, bubble_mask):
        x = self.slot_proj(xt)
        bnd = self.boundary_proj(boundary)
        c = self.time_embed(t)

        boundary_kpm = safe_kpm(boundary_mask)
        bubble_kpm = safe_kpm(bubble_mask)

        for block in self.blocks:
            x = block(x, c, bnd, boundary_kpm, bubble_kpm)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        output = self.final_linear(x)
        return output * bubble_mask.unsqueeze(-1)
