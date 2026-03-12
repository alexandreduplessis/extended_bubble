"""Inference sampler for bubble diagram generation using DDIM with CFG."""

from typing import Callable, List, Optional, Tuple

import torch

from src.model.transformer import BubbleDenoiser
from src.model.diffusion import GaussianDiffusion, _extract


class BubbleSampler:
    """Generates bubble diagrams via DDIM sampling with classifier-free guidance."""

    def __init__(
        self,
        model: BubbleDenoiser,
        diffusion: GaussianDiffusion,
        num_room_types: int = 9,
    ):
        self.model = model
        self.diffusion = diffusion
        self.device = next(model.parameters()).device
        self.num_room_types = num_room_types
        # type one-hot has num_room_types + 1 dims (last is "empty")
        self.num_type_dims = num_room_types + 1

    @torch.no_grad()
    def sample(
        self,
        boundary: torch.Tensor,
        boundary_mask: torch.Tensor,
        constraints: torch.Tensor,
        constraint_mask: torch.Tensor,
        n_slots: int = 200,
        num_steps: int = 100,
        cfg_scale: float = 3.0,
        energy_fns: Optional[List[Callable]] = None,
        energy_lambda: float = 0.01,
    ) -> List[Tuple[float, float, float, int]]:
        """Sample bubble diagrams using DDIM with classifier-free guidance.

        Args:
            boundary: (1, Nb, 2) boundary vertex coordinates.
            boundary_mask: (1, Nb) mask for boundary vertices.
            constraints: (1, Nc, constraint_dim) constraint features.
            constraint_mask: (1, Nc) mask for constraints.
            n_slots: number of bubble slots to generate.
            num_steps: number of DDIM denoising steps.
            cfg_scale: classifier-free guidance scale.
            energy_fns: optional list of energy functions for guidance.
            energy_lambda: weight for energy guidance.

        Returns:
            List of (cx, cy, radius, room_type) tuples for non-empty bubbles.
        """
        self.model.eval()
        geom_dims = self.diffusion.geom_dims
        type_dims = self.diffusion.type_dims
        slot_dim = geom_dims + type_dims
        B = 1

        # Move inputs to device
        boundary = boundary.to(self.device)
        boundary_mask = boundary_mask.to(self.device)
        constraints = constraints.to(self.device)
        constraint_mask = constraint_mask.to(self.device)

        # All slots active during generation
        bubble_mask = torch.ones(B, n_slots, device=self.device)

        # Null kwargs for unconditional pass (zero out constraint_mask)
        null_constraint_mask = torch.zeros_like(constraint_mask)

        # Build timestep subsequence (descending)
        step_size = self.diffusion.num_timesteps // num_steps
        timesteps = list(range(0, self.diffusion.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        # Start from pure noise
        xt = torch.randn(B, n_slots, slot_dim, device=self.device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)

            # Conditional prediction
            pred_noise_cond = self.model(
                xt, t, boundary, boundary_mask, constraints, constraint_mask, bubble_mask
            )

            # Unconditional prediction (zeroed constraint_mask)
            pred_noise_uncond = self.model(
                xt, t, boundary, boundary_mask, constraints, null_constraint_mask, bubble_mask
            )

            # CFG combination
            pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)

            # Predict x0
            x0_pred = self.diffusion.predict_x0_from_noise(xt, t, pred_noise)

            # Optional energy guidance
            if energy_fns:
                x0_guided = x0_pred.detach().clone().requires_grad_(True)
                total_energy = sum(fn(x0_guided) for fn in energy_fns)
                grad = torch.autograd.grad(total_energy, x0_guided)[0]
                x0_pred = x0_pred - energy_lambda * grad

            # Previous timestep
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
            else:
                t_prev_val = 0

            t_prev = torch.full((B,), t_prev_val, device=self.device, dtype=torch.long)

            # --- Geometry DDIM update (deterministic, eta=0) ---
            x0_geom = x0_pred[..., :geom_dims]
            noise_geom = pred_noise[..., :geom_dims]

            abar_prev_g = _extract(self.diffusion.alphas_cumprod, t_prev, x0_geom.shape)
            sqrt_abar_prev_g = torch.sqrt(abar_prev_g)
            dir_xt_g = torch.sqrt((1.0 - abar_prev_g).clamp(min=0))

            xt_prev_geom = sqrt_abar_prev_g * x0_geom + dir_xt_g * noise_geom

            # --- Type DDIM update (deterministic, eta=0) ---
            x0_type = x0_pred[..., geom_dims:]
            noise_type = pred_noise[..., geom_dims:]

            abar_prev_tp = _extract(self.diffusion.alphas_cumprod_type, t_prev, x0_type.shape)
            sqrt_abar_prev_tp = torch.sqrt(abar_prev_tp)
            dir_xt_tp = torch.sqrt((1.0 - abar_prev_tp).clamp(min=0))

            xt_prev_type = sqrt_abar_prev_tp * x0_type + dir_xt_tp * noise_type

            xt = torch.cat([xt_prev_geom, xt_prev_type], dim=-1)

        # Final x0 prediction from the last xt
        x0_final = xt  # At t=0, xt is the final prediction

        # Post-process: replace NaN (can occur with untrained models)
        x0_final = torch.nan_to_num(x0_final, nan=0.0)
        x0_geom = x0_final[0, :, :geom_dims]  # (n_slots, 3)
        x0_type = x0_final[0, :, geom_dims:]   # (n_slots, type_dims)

        # Snap type dims to nearest room type (argmax on one-hot)
        type_indices = torch.argmax(x0_type, dim=-1)  # (n_slots,)

        # Clamp geometry to valid ranges
        cx = x0_geom[:, 0].clamp(0.0, 1.0)
        cy = x0_geom[:, 1].clamp(0.0, 1.0)
        radius = x0_geom[:, 2].clamp(0.01, 0.5)

        # Filter out empty slots (last type index = num_room_types)
        empty_idx = self.num_room_types  # index 9
        results = []
        for j in range(n_slots):
            rtype = type_indices[j].item()
            if rtype != empty_idx:
                results.append((
                    cx[j].item(),
                    cy[j].item(),
                    radius[j].item(),
                    int(rtype),
                ))

        return results
