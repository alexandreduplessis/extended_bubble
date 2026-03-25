"""Generate samples from a boundary-only checkpoint."""

import sys
import argparse
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from boundary_only.model import BoundaryDenoiser
from boundary_only.dataset import BoundaryBubbleDataset, collate_fn, SLOT_DIM, NUM_ROOM_TYPES
from boundary_only.train import Trainer
from src.model.diffusion import GaussianDiffusion
from src.evaluation.visualize import plot_bubbles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--max_rooms", type=int, default=None,
                        help="Max rooms per sample. If None, uses ground truth count.")
    parser.add_argument("--out_dir", default="samples_boundary_only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    dc = config["data"]
    mc = config["model"]
    diff_c = config["diffusion"]

    model = BoundaryDenoiser(
        slot_dim=SLOT_DIM, d_model=mc["d_model"], n_layers=mc["n_layers"],
        n_heads=mc["n_heads"], ffn_ratio=mc["ffn_ratio"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=diff_c["num_timesteps"], schedule=diff_c["schedule"],
        type_schedule_shift=diff_c["type_schedule_shift"],
        geom_dims=3, type_dims=NUM_ROOM_TYPES + 1,
    ).to(device)

    # Build a minimal trainer just for sampling
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.diffusion = diffusion
    trainer.device = device
    trainer.config = config

    dataset = BoundaryBubbleDataset(
        cache_dir=dc["cache_dir"], split="test",
        n_max=dc["n_max"], n_boundary_max=dc["n_boundary_max"], augment=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    step = ckpt["step"]
    print(f"Loaded step {step}, generating {args.n_samples} samples...")

    for i in range(args.n_samples):
        sample = dataset[i % len(dataset)]
        boundary = sample["boundary"].unsqueeze(0).to(device)
        boundary_mask = sample["boundary_mask"].unsqueeze(0).to(device).float()
        n_real = sample["bubble_mask"].sum().item()
        max_rooms = args.max_rooms or int(n_real)

        bubbles = trainer._sample_one(
            boundary, boundary_mask, n_slots=dc["n_max"],
            num_steps=config["inference"]["num_steps"],
            cfg_scale=config["inference"]["cfg_scale"],
            max_rooms=max_rooms,
        )

        bnd_np = boundary[0].cpu().numpy()
        valid = boundary_mask[0].cpu().numpy() > 0.5
        bnd_np = bnd_np[valid]

        # Also plot ground truth
        gt_bubbles = []
        bt = sample["bubbles"]
        for j in range(int(n_real)):
            cx, cy, r = bt[j, 0].item(), bt[j, 1].item(), bt[j, 2].item()
            rt = torch.argmax(bt[j, 3:]).item()
            gt_bubbles.append((cx, cy, r, rt))

        plot_bubbles(gt_bubbles, boundary=bnd_np,
                     title=f"Ground Truth — {len(gt_bubbles)} rooms",
                     save_path=str(out_dir / f"gt_{i}.png"))
        plot_bubbles(bubbles, boundary=bnd_np,
                     title=f"Generated (step {step}) — {len(bubbles)} rooms",
                     save_path=str(out_dir / f"gen_{i}_step{step}.png"))
        print(f"  [{i+1}] GT={len(gt_bubbles)} rooms, Gen={len(bubbles)} rooms")

    print("Done!")


if __name__ == "__main__":
    main()
