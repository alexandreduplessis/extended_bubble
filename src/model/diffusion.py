"""Diffusion process with separate schedules for geometry and type dimensions."""

import math
import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Standard cosine schedule from Nichol & Dhariwal.

    Returns betas tensor of shape (timesteps,).
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999).float()


def shifted_cosine_schedule(
    timesteps: int, shift: float = 0.3, s: float = 0.008
) -> torch.Tensor:
    """Faster cosine schedule for type dimensions.

    Compresses the cosine schedule into the first `shift` fraction of timesteps,
    so types are nearly clean by t = shift * T.
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    t_effective = (steps / timesteps / shift).clamp(max=1.0)
    f = torch.cos((t_effective + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999).float()


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract values from `a` at indices `t`, reshaped for broadcasting with x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.long())
    # Reshape to (B, 1, 1, ...) for broadcasting
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion with separate noise schedules for geometry and type dims."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        type_schedule_shift: float = 0.3,
        geom_dims: int = 3,
        type_dims: int = 10,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.geom_dims = geom_dims
        self.type_dims = type_dims

        # --- Geometry schedule ---
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # Posterior coefficients for DDPM reverse: q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # --- Type schedule (shifted / faster) ---
        betas_type = shifted_cosine_schedule(
            num_timesteps, shift=type_schedule_shift
        )
        alphas_type = 1.0 - betas_type
        alphas_cumprod_type = torch.cumprod(alphas_type, dim=0)
        alphas_cumprod_type_prev = torch.cat(
            [torch.ones(1), alphas_cumprod_type[:-1]]
        )

        self.register_buffer("betas_type", betas_type)
        self.register_buffer("alphas_cumprod_type", alphas_cumprod_type)
        self.register_buffer(
            "sqrt_alphas_cumprod_type", torch.sqrt(alphas_cumprod_type)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod_type",
            torch.sqrt(1.0 - alphas_cumprod_type),
        )

        # Type posterior
        posterior_variance_type = (
            betas_type
            * (1.0 - alphas_cumprod_type_prev)
            / (1.0 - alphas_cumprod_type)
        )
        self.register_buffer("posterior_variance_type", posterior_variance_type)
        self.register_buffer(
            "posterior_mean_coef1_type",
            betas_type
            * torch.sqrt(alphas_cumprod_type_prev)
            / (1.0 - alphas_cumprod_type),
        )
        self.register_buffer(
            "posterior_mean_coef2_type",
            (1.0 - alphas_cumprod_type_prev)
            * torch.sqrt(alphas_type)
            / (1.0 - alphas_cumprod_type),
        )

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion q(x_t | x_0).

        Args:
            x0: (B, N, D) clean data where D = geom_dims + type_dims
            t:  (B,) integer timesteps

        Returns:
            (xt, noise) both of shape (B, N, D)
        """
        noise = torch.randn_like(x0)

        x0_geom = x0[..., : self.geom_dims]
        x0_type = x0[..., self.geom_dims :]
        noise_geom = noise[..., : self.geom_dims]
        noise_type = noise[..., self.geom_dims :]

        # Geometry
        sqrt_abar_g = _extract(self.sqrt_alphas_cumprod, t, x0_geom.shape)
        sqrt_1m_abar_g = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0_geom.shape
        )
        xt_geom = sqrt_abar_g * x0_geom + sqrt_1m_abar_g * noise_geom

        # Type
        sqrt_abar_t = _extract(self.sqrt_alphas_cumprod_type, t, x0_type.shape)
        sqrt_1m_abar_t = _extract(
            self.sqrt_one_minus_alphas_cumprod_type, t, x0_type.shape
        )
        xt_type = sqrt_abar_t * x0_type + sqrt_1m_abar_t * noise_type

        xt = torch.cat([xt_geom, xt_type], dim=-1)
        return xt, noise

    # ------------------------------------------------------------------
    # Reverse helpers
    # ------------------------------------------------------------------

    def predict_x0_from_noise(
        self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Recover x0 from xt and predicted noise, using split schedules."""
        xt_geom = xt[..., : self.geom_dims]
        xt_type = xt[..., self.geom_dims :]
        noise_geom = noise[..., : self.geom_dims]
        noise_type = noise[..., self.geom_dims :]

        # Geometry
        sqrt_abar_g = _extract(self.sqrt_alphas_cumprod, t, xt_geom.shape)
        sqrt_1m_abar_g = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt_geom.shape
        )
        x0_geom = (xt_geom - sqrt_1m_abar_g * noise_geom) / sqrt_abar_g.clamp(
            min=1e-8
        )

        # Type
        sqrt_abar_t = _extract(self.sqrt_alphas_cumprod_type, t, xt_type.shape)
        sqrt_1m_abar_t = _extract(
            self.sqrt_one_minus_alphas_cumprod_type, t, xt_type.shape
        )
        x0_type = (xt_type - sqrt_1m_abar_t * noise_type) / sqrt_abar_t.clamp(
            min=1e-8
        )

        x0 = torch.cat([x0_geom, x0_type], dim=-1)
        return x0.clamp(-3, 3)

    def p_sample(
        self,
        model_fn,
        xt: torch.Tensor,
        t: torch.Tensor,
        **model_kwargs,
    ) -> torch.Tensor:
        """Single DDPM reverse step."""
        predicted_noise = model_fn(xt, t, **model_kwargs)
        x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)

        # Posterior mean for geometry
        xt_geom = xt[..., : self.geom_dims]
        x0_geom = x0_pred[..., : self.geom_dims]
        coef1_g = _extract(self.posterior_mean_coef1, t, xt_geom.shape)
        coef2_g = _extract(self.posterior_mean_coef2, t, xt_geom.shape)
        mean_geom = coef1_g * x0_geom + coef2_g * xt_geom

        # Posterior mean for type
        xt_type = xt[..., self.geom_dims :]
        x0_type = x0_pred[..., self.geom_dims :]
        coef1_t = _extract(self.posterior_mean_coef1_type, t, xt_type.shape)
        coef2_t = _extract(self.posterior_mean_coef2_type, t, xt_type.shape)
        mean_type = coef1_t * x0_type + coef2_t * xt_type

        mean = torch.cat([mean_geom, mean_type], dim=-1)

        # Add noise except at t == 0
        var_g = _extract(self.posterior_variance, t, xt_geom.shape)
        var_t = _extract(self.posterior_variance_type, t, xt_type.shape)
        variance = torch.cat(
            [var_g.expand_as(xt_geom), var_t.expand_as(xt_type)], dim=-1
        )

        noise = torch.randn_like(xt)
        nonzero_mask = (t > 0).float().view(-1, *((1,) * (xt.dim() - 1)))
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,
        shape: tuple,
        num_steps: int = 100,
        cfg_scale: float = 0.0,
        eta: float = 0.0,
        **model_kwargs,
    ) -> torch.Tensor:
        """Full DDIM sampling loop.

        Args:
            model_fn: noise prediction network  (xt, t, **kwargs) -> noise
            shape: (B, N, D)
            num_steps: number of DDIM steps (< num_timesteps)
            cfg_scale: classifier-free guidance scale (0 = no guidance)
            eta: DDIM stochasticity (0 = deterministic)
            **model_kwargs: extra kwargs passed to model_fn;
                if cfg_scale > 0, must contain 'null_kwargs' dict for unconditional pass
        """
        device = self.betas.device
        B = shape[0]

        # Subsequence of timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))  # descending

        xt = torch.randn(shape, device=device)

        null_kwargs = model_kwargs.pop("null_kwargs", {})

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Model prediction (with optional CFG)
            pred_noise = model_fn(xt, t, **model_kwargs)
            if cfg_scale > 0.0:
                pred_noise_uncond = model_fn(xt, t, **null_kwargs)
                pred_noise = (
                    pred_noise_uncond
                    + cfg_scale * (pred_noise - pred_noise_uncond)
                )

            # Predict x0
            x0_pred = self.predict_x0_from_noise(xt, t, pred_noise)

            # Previous timestep
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
            else:
                t_prev_val = 0

            t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            # --- Geometry DDIM update ---
            x0_geom = x0_pred[..., : self.geom_dims]
            xt_geom = xt[..., : self.geom_dims]
            noise_geom = pred_noise[..., : self.geom_dims]

            abar_t_g = _extract(self.alphas_cumprod, t, xt_geom.shape)
            abar_prev_g = _extract(self.alphas_cumprod, t_prev, xt_geom.shape)

            sqrt_abar_prev_g = torch.sqrt(abar_prev_g)
            sqrt_1m_abar_prev_g = torch.sqrt(1.0 - abar_prev_g)

            # sigma for stochasticity
            sigma_g = (
                eta
                * torch.sqrt(
                    (1 - abar_prev_g)
                    / (1 - abar_t_g).clamp(min=1e-8)
                    * (1 - abar_t_g / abar_prev_g.clamp(min=1e-8))
                )
            )

            dir_xt_g = torch.sqrt((1.0 - abar_prev_g - sigma_g**2).clamp(min=0))
            xt_prev_geom = (
                sqrt_abar_prev_g * x0_geom
                + dir_xt_g * noise_geom
                + sigma_g * torch.randn_like(xt_geom)
            )

            # --- Type DDIM update ---
            x0_type = x0_pred[..., self.geom_dims :]
            xt_type = xt[..., self.geom_dims :]
            noise_type = pred_noise[..., self.geom_dims :]

            abar_t_tp = _extract(self.alphas_cumprod_type, t, xt_type.shape)
            abar_prev_tp = _extract(self.alphas_cumprod_type, t_prev, xt_type.shape)

            sqrt_abar_prev_tp = torch.sqrt(abar_prev_tp)

            sigma_tp = (
                eta
                * torch.sqrt(
                    (1 - abar_prev_tp)
                    / (1 - abar_t_tp).clamp(min=1e-8)
                    * (1 - abar_t_tp / abar_prev_tp.clamp(min=1e-8))
                )
            )

            dir_xt_tp = torch.sqrt((1.0 - abar_prev_tp - sigma_tp**2).clamp(min=0))
            xt_prev_type = (
                sqrt_abar_prev_tp * x0_type
                + dir_xt_tp * noise_type
                + sigma_tp * torch.randn_like(xt_type)
            )

            xt = torch.cat([xt_prev_geom, xt_prev_type], dim=-1)

        return xt
