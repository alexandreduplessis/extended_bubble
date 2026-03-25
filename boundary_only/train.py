"""Training script for boundary-only bubble diffusion."""

import sys
import math
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from boundary_only.model import BoundaryDenoiser
from boundary_only.dataset import BoundaryBubbleDataset, collate_fn, SLOT_DIM, NUM_ROOM_TYPES
from src.model.diffusion import GaussianDiffusion
from src.evaluation.visualize import plot_bubbles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.global_step = 0

        dc = config["data"]
        mc = config["model"]
        tc = config["training"]
        diff_c = config["diffusion"]

        self.slot_dim = SLOT_DIM  # 13

        self.model = BoundaryDenoiser(
            slot_dim=self.slot_dim,
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            n_heads=mc["n_heads"],
            ffn_ratio=mc["ffn_ratio"],
            dropout=mc["dropout"],
        ).to(device)

        self.diffusion = GaussianDiffusion(
            num_timesteps=diff_c["num_timesteps"],
            schedule=diff_c["schedule"],
            type_schedule_shift=diff_c["type_schedule_shift"],
            geom_dims=3,
            type_dims=NUM_ROOM_TYPES + 1,
        ).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tc["lr"])
        self.warmup_steps = tc["warmup_steps"]
        self.max_steps = tc["max_steps"]
        self.cfg_drop_boundary = tc["cfg_drop_boundary"]
        self.type_loss_weight = tc["type_loss_weight"]
        self.grad_clip = tc["grad_clip"]

        self.train_dataset = BoundaryBubbleDataset(
            cache_dir=dc["cache_dir"], split="train",
            n_max=dc["n_max"], n_boundary_max=dc["n_boundary_max"], augment=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=tc["batch_size"], shuffle=True,
            num_workers=8, collate_fn=collate_fn, pin_memory=True,
            drop_last=True, persistent_workers=True, prefetch_factor=4,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

    def _get_lr(self):
        if self.global_step < self.warmup_steps:
            return self.config["training"]["lr"] * self.global_step / self.warmup_steps
        progress = (self.global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.config["training"]["lr"] * 0.5 * (1 + math.cos(progress * math.pi))

    def train_step(self, batch):
        self.model.train()
        bubbles = batch["bubbles"].to(self.device)
        bubble_mask = batch["bubble_mask"].to(self.device).float()
        boundary = batch["boundary"].to(self.device)
        boundary_mask = batch["boundary_mask"].to(self.device).float()
        B = bubbles.shape[0]

        # CFG dropout: drop boundary for some samples
        drop_bnd = torch.rand(B, device=self.device) < self.cfg_drop_boundary
        boundary_mask = boundary_mask * (~drop_bnd).float().unsqueeze(-1)

        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
        xt, noise = self.diffusion.q_sample(bubbles, t)

        pred_noise = self.model(xt, t, boundary, boundary_mask, bubble_mask)

        # Geometry loss (masked to real rooms only)
        geom_error = (pred_noise[..., :3] - noise[..., :3]) ** 2
        geom_loss = (geom_error * bubble_mask.unsqueeze(-1)).sum() / bubble_mask.sum().clamp(min=1) / 3

        # Type loss (all slots, but only at low noise t < T/2)
        type_error = (pred_noise[..., 3:] - noise[..., 3:]) ** 2
        n_type_dims = self.slot_dim - 3
        N = pred_noise.shape[1]
        low_noise = (t < self.diffusion.num_timesteps // 2).float()
        type_per_sample = type_error.sum(dim=(1, 2)) / (N * n_type_dims)
        type_loss = (type_per_sample * low_noise).sum() / low_noise.sum().clamp(min=1)

        loss = geom_loss + self.type_loss_weight * type_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.optimizer.step()
        self.global_step += 1

        return {"loss": loss.item(), "geom_loss": geom_loss.item(),
                "type_loss": type_loss.item(), "lr": lr}

    @torch.no_grad()
    def generate_samples(self, step, save_dir, n_samples=4):
        self.model.eval()
        batch = next(iter(self.train_loader))
        sample_dir = save_dir / "samples"
        sample_dir.mkdir(exist_ok=True)

        images = []
        dc = self.config["data"]
        ic = self.config["inference"]

        for i in range(min(n_samples, batch["boundary"].shape[0])):
            boundary = batch["boundary"][i:i+1].to(self.device)
            boundary_mask = batch["boundary_mask"][i:i+1].to(self.device).float()
            n_real = batch["bubble_mask"][i].sum().item()

            bubbles = self._sample_one(boundary, boundary_mask,
                                        n_slots=dc["n_max"], num_steps=50,
                                        cfg_scale=ic["cfg_scale"], max_rooms=int(n_real))

            bnd_np = boundary[0].cpu().numpy()
            valid = boundary_mask[0].cpu().numpy() > 0.5
            bnd_np = bnd_np[valid]

            img_path = str(sample_dir / f"step_{step}_sample_{i}.png")
            plot_bubbles(bubbles, boundary=bnd_np,
                         title=f"Step {step} — {len(bubbles)} rooms", save_path=img_path)
            images.append(wandb.Image(img_path, caption=f"Sample {i} ({len(bubbles)} rooms)"))

        if images:
            wandb.log({"samples": images}, step=step)
        self.model.train()

    def _sample_one(self, boundary, boundary_mask, n_slots, num_steps, cfg_scale, max_rooms=None):
        """DDIM sampling with boundary-only CFG."""
        from src.model.diffusion import _extract

        geom_dims = self.diffusion.geom_dims
        type_dims = self.diffusion.type_dims
        slot_dim = geom_dims + type_dims

        bubble_mask = torch.ones(1, n_slots, device=self.device)
        null_bnd_mask = torch.zeros_like(boundary_mask)

        step_size = self.diffusion.num_timesteps // num_steps
        timesteps = list(reversed(range(0, self.diffusion.num_timesteps, step_size)))

        xt = torch.randn(1, n_slots, slot_dim, device=self.device)

        empty_onehot = torch.zeros(type_dims, device=self.device)
        empty_onehot[-1] = 1.0
        active_mask = None
        selection_step = max(1, len(timesteps) // 10)

        for i, t_val in enumerate(timesteps):
            t = torch.full((1,), t_val, device=self.device, dtype=torch.long)

            with torch.no_grad():
                pred_cond = self.model(xt, t, boundary, boundary_mask, bubble_mask)
                pred_uncond = self.model(xt, t, boundary, null_bnd_mask, bubble_mask)
                pred_noise = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                x0_pred = self.diffusion.predict_x0_from_noise(xt, t, pred_noise)

            # Slot selection
            if max_rooms is not None and i == selection_step and active_mask is None:
                x0_type = x0_pred[0, :, geom_dims:]
                score = x0_type[:, :-1].max(dim=-1).values - x0_type[:, -1]
                _, top_idx = torch.topk(score, min(max_rooms, n_slots))
                active_mask = torch.zeros(n_slots, device=self.device, dtype=torch.bool)
                active_mask[top_idx] = True

            if active_mask is not None:
                x0_pred[0, ~active_mask, geom_dims:] = empty_onehot

            # DDIM step
            t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t_prev = torch.full((1,), t_prev_val, device=self.device, dtype=torch.long)

            x0_g = x0_pred[..., :geom_dims]
            ng = pred_noise[..., :geom_dims]
            ap_g = _extract(self.diffusion.alphas_cumprod, t_prev, x0_g.shape)
            xt_g = torch.sqrt(ap_g) * x0_g + torch.sqrt((1 - ap_g).clamp(min=0)) * ng

            x0_t = x0_pred[..., geom_dims:]
            nt = pred_noise[..., geom_dims:]
            ap_t = _extract(self.diffusion.alphas_cumprod_type, t_prev, x0_t.shape)
            xt_t = torch.sqrt(ap_t) * x0_t + torch.sqrt((1 - ap_t).clamp(min=0)) * nt

            if active_mask is not None:
                abar_val = ap_t[0, 0, 0].item()
                xt_t[0, ~active_mask] = (
                    math.sqrt(abar_val) * empty_onehot
                    + math.sqrt(1 - abar_val) * torch.randn_like(xt_t[0, ~active_mask])
                )

            xt = torch.cat([xt_g, xt_t], dim=-1)

        x0_final = torch.nan_to_num(xt, nan=0.0)
        x0_geom = x0_final[0, :, :geom_dims]
        x0_type = x0_final[0, :, geom_dims:]
        type_idx = torch.argmax(x0_type, dim=-1)

        cx = x0_geom[:, 0].clamp(0, 1)
        cy = x0_geom[:, 1].clamp(0, 1)
        radius = x0_geom[:, 2].clamp(0.01, 0.5)

        empty_idx = NUM_ROOM_TYPES
        results = []
        for j in range(n_slots):
            rt = type_idx[j].item()
            if rt != empty_idx:
                conf = x0_type[j, :empty_idx].max().item() - x0_type[j, empty_idx].item()
                results.append((cx[j].item(), cy[j].item(), radius[j].item(), rt, conf))

        results.sort(key=lambda x: x[4], reverse=True)
        if max_rooms and len(results) > max_rooms:
            results = results[:max_rooms]
        return [(c, y, r, t) for c, y, r, t, _ in results]

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config,
        }, path)

    def train(self):
        tc = self.config["training"]
        save_dir = Path(tc["checkpoint_dir"])
        save_dir.mkdir(exist_ok=True)

        wandb.init(project="extendedbubble", config=self.config,
                   name=f"boundary-only-{self.max_steps}steps")
        wandb.watch(self.model, log="gradients", log_freq=500)

        sample_every = tc.get("sample_every", 2000)
        data_iter = iter(self.train_loader)
        step_start = time.time()

        for step in range(self.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)
            dt = time.time() - step_start
            metrics["step_time"] = dt
            wandb.log(metrics, step=step)

            logger.info(
                f"Step {step}: loss={metrics['loss']:.4f} "
                f"geom={metrics['geom_loss']:.4f} type={metrics['type_loss']:.4f} "
                f"lr={metrics['lr']:.6f} dt={dt:.2f}s"
            )
            step_start = time.time()

            if step > 0 and step % tc["save_every"] == 0:
                self.save(str(save_dir / f"checkpoint_{step}.pt"))

            if step > 0 and step % sample_every == 0:
                self.generate_samples(step, save_dir)

        self.save(str(save_dir / "checkpoint_final.pt"))
        wandb.finish()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(config, device)
    trainer.train()


if __name__ == "__main__":
    main()
