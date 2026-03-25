"""Generate sample bubble diagrams from a checkpoint."""
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.diffusion import GaussianDiffusion
from src.model.transformer import BubbleDenoiser
from src.inference.sampler import BubbleSampler
from src.inference.energy import build_energy_fns
from src.data.dataset import BubbleDataset, collate_bubbles
from src.evaluation.visualize import plot_bubbles


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/Data/amine.chraibi/checkpoints/checkpoint_128200.pt")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--max_rooms", type=int, default=10)
    parser.add_argument("--out_dir", default="/Data/amine.chraibi/samples")
    parser.add_argument("--energy_lambda", type=float, default=0.0)
    args = parser.parse_args()

    ckpt_path = args.ckpt
    n_samples = args.n_samples
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        dropout=0.0,
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

    # Load real samples for conditioning
    dataset = BubbleDataset(
        msd_path=dc["msd_path"],
        split="test",
        n_max=dc["n_max"],
        n_boundary_max=dc["n_boundary_max"],
        min_constraints=dc["min_constraints"],
        max_constraints=dc["max_constraints"],
        augment=False,
    )

    energy_lambda = args.energy_lambda
    max_rooms = args.max_rooms
    step = ckpt["step"]
    print(f"Loaded checkpoint from step {step}")
    print(f"Generating {n_samples} samples (max_rooms={max_rooms}, energy_lambda={energy_lambda})...")

    for i in range(n_samples):
        sample = dataset[i % len(dataset)]
        boundary = sample["boundary"].unsqueeze(0).to(device)
        boundary_mask = sample["boundary_mask"].unsqueeze(0).to(device).float()
        constraints = sample["constraints"].unsqueeze(0).to(device)
        constraint_mask = sample["constraint_mask"].unsqueeze(0).to(device).float()

        # Build energy functions for this sample
        energy_fns = build_energy_fns(
            boundary, boundary_mask, constraints, constraint_mask,
            num_room_types=dc["num_room_types"],
        )

        bubbles = sampler.sample(
            boundary, boundary_mask, constraints, constraint_mask,
            n_slots=dc["n_max"],
            num_steps=config["inference"]["num_steps"],
            cfg_scale=config["inference"]["cfg_scale"],
            energy_fns=energy_fns,
            energy_lambda=energy_lambda,
            max_rooms=max_rooms,
        )

        bnd_np = boundary[0].cpu().numpy()
        valid = boundary_mask[0].cpu().numpy() > 0.5
        bnd_np = bnd_np[valid]

        img_path = str(out_dir / f"sample_{i}_step{step}.png")
        plot_bubbles(
            bubbles, boundary=bnd_np,
            title=f"Step {step} — {len(bubbles)} rooms",
            save_path=img_path,
        )
        print(f"  [{i+1}/{n_samples}] {len(bubbles)} rooms -> {img_path}")

    print("Done!")


if __name__ == "__main__":
    main()
