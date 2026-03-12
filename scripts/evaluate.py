"""Evaluation script: load checkpoint, sample layouts, compute metrics, save visualizations."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.constraints import Constraint, ConstraintType, encode_constraint
from src.evaluation.metrics import boundary_coverage, constraint_satisfaction_rate
from src.evaluation.visualize import plot_bubbles
from src.inference.sampler import BubbleSampler
from src.model.diffusion import GaussianDiffusion
from src.model.transformer import BubbleDenoiser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load a checkpoint and return the sampler and config."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    dc = config["data"]
    mc = config["model"]
    diff_c = config["diffusion"]

    slot_dim = 3 + dc["num_room_types"] + 1

    model = BubbleDenoiser(
        slot_dim=slot_dim,
        d_model=mc["d_model"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        constraint_dim=dc["constraint_dim"],
        ffn_ratio=mc["ffn_ratio"],
        dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=diff_c["num_timesteps"],
        schedule=diff_c["schedule"],
        type_schedule_shift=diff_c["type_schedule_shift"],
        geom_dims=3,
        type_dims=dc["num_room_types"] + 1,
    ).to(device)

    sampler = BubbleSampler(model, diffusion, num_room_types=dc["num_room_types"])
    return sampler, config


def make_dummy_inputs(config: dict, device: str = "cpu"):
    """Create simple dummy boundary and constraint tensors for sampling."""
    dc = config["data"]

    # Simple square boundary [0,1]^2
    boundary_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    n_boundary_max = dc["n_boundary_max"]
    boundary = torch.zeros(1, n_boundary_max, 2)
    boundary[0, :4] = torch.from_numpy(boundary_pts)
    boundary_mask = torch.zeros(1, n_boundary_max)
    boundary_mask[0, :4] = 1.0

    # Minimal constraints: MUST_EXIST for a few room types
    constraint_list = [
        Constraint(type=ConstraintType.MUST_EXIST, room_type_a=0),  # Bedroom
        Constraint(type=ConstraintType.MUST_EXIST, room_type_a=1),  # Living
        Constraint(type=ConstraintType.MUST_EXIST, room_type_a=7),  # Bathroom
    ]
    encoded = [encode_constraint(c) for c in constraint_list]

    n_constraints_max = dc.get("n_constraints_max", 300)
    constraints = torch.zeros(1, n_constraints_max, dc["constraint_dim"])
    constraint_mask = torch.zeros(1, n_constraints_max)
    for i, enc in enumerate(encoded):
        constraints[0, i] = torch.from_numpy(enc)
        constraint_mask[0, i] = 1.0

    boundary = boundary.to(device)
    boundary_mask = boundary_mask.to(device)
    constraints = constraints.to(device)
    constraint_mask = constraint_mask.to(device)

    return boundary, boundary_mask, constraints, constraint_mask, boundary_pts, constraint_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate bubble diagram generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of layouts to sample")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="Directory for output")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--num_steps", type=int, default=100, help="DDIM denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="CFG guidance scale")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    sampler, config = load_checkpoint(args.checkpoint, device=args.device)

    boundary, boundary_mask, constraints, constraint_mask, boundary_np, constraint_list = \
        make_dummy_inputs(config, device=args.device)

    all_csr = []
    all_coverage = []

    for i in range(args.num_samples):
        logger.info(f"Sampling layout {i + 1}/{args.num_samples}")

        bubbles = sampler.sample(
            boundary=boundary,
            boundary_mask=boundary_mask,
            constraints=constraints,
            constraint_mask=constraint_mask,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )

        # Compute metrics
        csr = constraint_satisfaction_rate(bubbles, constraint_list)
        cov = boundary_coverage(bubbles, boundary_np)
        all_csr.append(csr)
        all_coverage.append(cov)

        logger.info(
            f"  Layout {i + 1}: {len(bubbles)} rooms, "
            f"CSR={csr:.3f}, coverage={cov:.3f}"
        )

        # Save visualization
        save_path = str(output_dir / f"layout_{i:03d}.png")
        plot_bubbles(
            bubbles,
            boundary=boundary_np,
            title=f"Sample {i + 1} | CSR={csr:.2f} | Cov={cov:.2f}",
            save_path=save_path,
        )

    # Summary
    logger.info("=" * 50)
    logger.info(f"Mean CSR:      {np.mean(all_csr):.4f} +/- {np.std(all_csr):.4f}")
    logger.info(f"Mean Coverage: {np.mean(all_coverage):.4f} +/- {np.std(all_coverage):.4f}")
    logger.info(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
