"""Load a trained 1D DDPM model and generate samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion_1d.data import sample_mixture_of_gaussians
from diffusion_1d.diffusion import DDPM1D
from diffusion_1d.model import DenoiseMLP
from diffusion_1d.visualize import plot_histogram_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples from trained 1D DDPM.")
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default="outputs/ddpm_1d.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-npy",
        type=str,
        default="outputs/generated_samples.npy",
        help="Where to save generated samples (.npy).",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/generated_vs_real.png",
        help="Where to save histogram comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    hidden_dim = int(checkpoint["hidden_dim"])
    timesteps = int(checkpoint["timesteps"])
    time_dim = int(checkpoint.get("time_dim", 64))
    layers = int(checkpoint.get("layers", 4))
    means = tuple(checkpoint.get("means", (-2.0, 0.0, 2.0)))
    stds = tuple(checkpoint.get("stds", (0.25, 0.25, 0.25)))
    weights = tuple(checkpoint.get("weights", (0.4, 0.2, 0.4)))

    model = DenoiseMLP(hidden_dim=hidden_dim, time_dim=time_dim, num_hidden_layers=layers)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = DDPM1D(model=model, timesteps=timesteps, device="cpu")
    generated = diffusion.p_sample_loop((args.num_samples, 1)).cpu().numpy().squeeze(-1)
    real = sample_mixture_of_gaussians(
        args.num_samples,
        means=means,
        stds=stds,
        weights=weights,
        seed=args.seed,
    ).numpy().squeeze(-1)

    save_npy = Path(args.save_npy)
    save_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_npy, generated)
    plot_histogram_comparison(real, generated, save_path=args.save_plot)

    print(f"Saved generated samples to: {save_npy.resolve()}")
    print(f"Saved histogram plot to: {Path(args.save_plot).resolve()}")


if __name__ == "__main__":
    main()
