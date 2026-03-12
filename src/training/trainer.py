"""Training loop for bubble diffusion."""
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

import wandb

from src.model.diffusion import GaussianDiffusion
from src.model.transformer import BubbleDenoiser
from src.data.dataset import BubbleDataset, collate_bubbles
from src.inference.sampler import BubbleSampler
from src.evaluation.visualize import plot_bubbles

logger = logging.getLogger(__name__)


class BubbleTrainer:
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.global_step = 0

        dc = config["data"]
        mc = config["model"]
        tc = config["training"]
        diff_c = config["diffusion"]

        # Slot dim = 3 (geom) + num_room_types + 1 (empty)
        self.slot_dim = 3 + dc["num_room_types"] + 1  # 13

        # Model
        self.model = BubbleDenoiser(
            slot_dim=self.slot_dim,
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            n_heads=mc["n_heads"],
            constraint_dim=dc["constraint_dim"],
            ffn_ratio=mc["ffn_ratio"],
            dropout=mc["dropout"],
        ).to(device)

        # Diffusion
        self.diffusion = GaussianDiffusion(
            num_timesteps=diff_c["num_timesteps"],
            schedule=diff_c["schedule"],
            type_schedule_shift=diff_c["type_schedule_shift"],
            geom_dims=3,
            type_dims=dc["num_room_types"] + 1,
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tc["lr"], weight_decay=0.0
        )

        # LR scheduler
        self.warmup_steps = tc["warmup_steps"]
        self.max_steps = tc["max_steps"]

        # Training params
        self.cfg_drop_single = tc["cfg_drop_single"]
        self.cfg_drop_all = tc["cfg_drop_all"]
        self.cfg_drop_boundary = tc["cfg_drop_boundary"]
        self.type_loss_weight = tc["type_loss_weight"]
        self.grad_clip = tc["grad_clip"]

        # Dataset
        self.train_dataset = BubbleDataset(
            msd_path=dc["msd_path"],
            split="train",
            n_max=dc["n_max"],
            n_boundary_max=dc["n_boundary_max"],
            min_constraints=dc["min_constraints"],
            max_constraints=dc["max_constraints"],
            augment=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=tc["batch_size"],
            shuffle=True,
            num_workers=8,
            collate_fn=collate_bubbles,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

    def _get_lr(self) -> float:
        """Linear warmup then cosine decay."""
        if self.global_step < self.warmup_steps:
            return self.config["training"]["lr"] * self.global_step / self.warmup_steps
        # Cosine decay
        progress = (self.global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.config["training"]["lr"] * 0.5 * (1 + math.cos(progress * math.pi))

    def _apply_cfg_dropout(self, constraints, constraint_mask, boundary, boundary_mask):
        """Apply classifier-free guidance dropout.

        - Drop all constraints with p=cfg_drop_all (zero out constraint_mask for whole batch items)
        - Drop individual constraints with p=cfg_drop_single
        - Drop boundary with p=cfg_drop_boundary
        """
        B = constraints.shape[0]

        # Drop all constraints for some samples
        drop_all = torch.rand(B, device=constraints.device) < self.cfg_drop_all
        constraint_mask = constraint_mask * (~drop_all).float().unsqueeze(-1)

        # Drop individual constraints
        if self.cfg_drop_single > 0:
            drop_single = torch.rand_like(constraint_mask) < self.cfg_drop_single
            constraint_mask = constraint_mask * (~drop_single).float()

        # Drop boundary for some samples
        drop_boundary = torch.rand(B, device=boundary.device) < self.cfg_drop_boundary
        boundary_mask = boundary_mask * (~drop_boundary).float().unsqueeze(-1)

        return constraints, constraint_mask, boundary, boundary_mask

    def train_step(self, batch: dict) -> dict:
        """Run a single training step.

        Returns a dict of metrics: loss, geom_loss, type_loss, lr.
        """
        self.model.train()

        # Move to device and convert bool masks to float
        bubbles = batch["bubbles"].to(self.device)                        # (B, N, slot_dim)
        bubble_mask = batch["bubble_mask"].to(self.device).float()        # (B, N)
        boundary = batch["boundary"].to(self.device)                      # (B, Nb, 2)
        boundary_mask = batch["boundary_mask"].to(self.device).float()
        constraints = batch["constraints"].to(self.device)                # (B, Nc, 30)
        constraint_mask = batch["constraint_mask"].to(self.device).float()

        B = bubbles.shape[0]

        # CFG dropout
        constraints, constraint_mask, boundary, boundary_mask = \
            self._apply_cfg_dropout(constraints, constraint_mask, boundary, boundary_mask)

        # Sample timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        # Forward diffusion
        xt, noise = self.diffusion.q_sample(bubbles, t)

        # Predict noise
        pred_noise = self.model(
            xt, t, boundary, boundary_mask, constraints, constraint_mask, bubble_mask
        )

        # Loss: MSE on noise, with masking
        # Geometry dims: masked for empty slots (only real rooms)
        geom_error = (pred_noise[..., :3] - noise[..., :3]) ** 2
        geom_loss = (geom_error * bubble_mask.unsqueeze(-1)).sum() / bubble_mask.sum().clamp(min=1) / 3

        # Type dims: always active (model must predict empty type noise too)
        type_error = (pred_noise[..., 3:] - noise[..., 3:]) ** 2
        n_type_dims = self.slot_dim - 3
        type_loss = type_error.sum() / (B * pred_noise.shape[1] * n_type_dims)

        loss = geom_loss + self.type_loss_weight * type_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # LR schedule
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.step()
        self.global_step += 1

        return {
            "loss": loss.item(),
            "geom_loss": geom_loss.item(),
            "type_loss": type_loss.item(),
            "lr": lr,
        }

    @torch.no_grad()
    def _generate_samples(self, step: int, save_dir: Path, n_samples: int = 4):
        """Generate and save sample bubble diagrams for visual monitoring."""
        self.model.eval()
        sampler = BubbleSampler(
            self.model, self.diffusion,
            num_room_types=self.config["data"]["num_room_types"],
        )

        # Grab a real batch for conditioning context
        try:
            batch = next(iter(self.train_loader))
        except StopIteration:
            return

        sample_dir = save_dir / "samples"
        sample_dir.mkdir(exist_ok=True)

        images = []
        for i in range(min(n_samples, batch["boundary"].shape[0])):
            boundary = batch["boundary"][i:i+1].to(self.device)
            boundary_mask = batch["boundary_mask"][i:i+1].to(self.device).float()
            constraints = batch["constraints"][i:i+1].to(self.device)
            constraint_mask = batch["constraint_mask"][i:i+1].to(self.device).float()

            bubbles = sampler.sample(
                boundary, boundary_mask, constraints, constraint_mask,
                n_slots=self.config["data"]["n_max"],
                num_steps=50,  # fewer steps for speed
                cfg_scale=self.config["inference"]["cfg_scale"],
            )

            bnd_np = boundary[0].cpu().numpy()
            valid = boundary_mask[0].cpu().numpy() > 0.5
            bnd_np = bnd_np[valid]

            img_path = str(sample_dir / f"step_{step}_sample_{i}.png")
            plot_bubbles(
                bubbles, boundary=bnd_np,
                title=f"Step {step} — {len(bubbles)} rooms",
                save_path=img_path,
            )
            images.append(wandb.Image(img_path, caption=f"Sample {i} ({len(bubbles)} rooms)"))

        if images:
            wandb.log({"samples": images}, step=step)

        self.model.train()

    def train(self):
        """Main training loop up to max_steps."""
        logger.info(f"Starting training for {self.max_steps} steps")
        tc = self.config["training"]
        save_dir = Path("/Data/amine.chraibi/checkpoints")
        save_dir.mkdir(exist_ok=True)

        # Wandb init
        wandb.init(
            project="extendedbubble",
            config=self.config,
            name=f"run-{self.max_steps}steps",
        )
        wandb.watch(self.model, log="gradients", log_freq=500)

        sample_every = tc.get("sample_every", 5000)
        data_iter = iter(self.train_loader)
        step_start = time.time()

        for step in range(self.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)

            # Log every step to wandb, print every step to console
            step_time = time.time() - step_start
            metrics["step_time"] = step_time
            wandb.log(metrics, step=step)

            logger.info(
                f"Step {step}: loss={metrics['loss']:.4f} "
                f"geom={metrics['geom_loss']:.4f} "
                f"type={metrics['type_loss']:.4f} "
                f"lr={metrics['lr']:.6f} "
                f"dt={step_time:.2f}s"
            )
            step_start = time.time()

            if step > 0 and step % tc["save_every"] == 0:
                ckpt_path = save_dir / f"checkpoint_{step}.pt"
                self.save(str(ckpt_path))
                logger.info(f"Saved checkpoint at step {step}")

            if step > 0 and step % sample_every == 0:
                logger.info(f"Generating samples at step {step}...")
                self._generate_samples(step, save_dir)

        # Final save and samples
        self.save(str(save_dir / "checkpoint_final.pt"))
        self._generate_samples(self.max_steps, save_dir)
        wandb.finish()

    def save(self, path: str):
        """Save checkpoint to disk."""
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config,
        }, path)

    def load(self, path: str):
        """Load checkpoint from disk."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["step"]
