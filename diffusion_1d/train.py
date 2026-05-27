"""Train a simple 1D DDPM on a mixture of Gaussians."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from diffusion_1d.data import sample_mixture_of_gaussians
from diffusion_1d.diffusion import DDPM1D
from diffusion_1d.model import DenoiseMLP
from diffusion_1d.visualize import (
    plot_forward_noising,
    plot_histogram_comparison,
    plot_loss_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D DDPM (CPU-friendly).")
    parser.add_argument("--steps", type=int, default=10_000, help="Total optimization steps.")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion steps T.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden size.")
    parser.add_argument("--time-dim", type=int, default=64, help="Sinusoidal timestep embedding size.")
    parser.add_argument("--layers", type=int, default=4, help="Number of hidden layers in the MLP.")
    parser.add_argument("--num-gen", type=int, default=5000, help="Generated samples for histogram.")
    parser.add_argument("--log-every", type=int, default=100, help="Log loss every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/ddpm_1d.pt",
        help="Path for model checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    means = (-2.0, 0.0, 2.0)
    stds = (0.25, 0.25, 0.25)
    weights = (0.4, 0.2, 0.4)

    model = DenoiseMLP(
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        num_hidden_layers=args.layers,
    ).to(device)
    diffusion = DDPM1D(model=model, timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    losses: list[float] = []
    progress = tqdm(range(1, args.steps + 1), desc="Training", leave=True)
    for step in progress:
        # Re-sample a fresh batch each step from the toy data distribution.
        x0 = sample_mixture_of_gaussians(
            n_samples=args.batch_size,
            means=means,
            stds=stds,
            weights=weights,
        ).to(device)

        t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device).long()
        noise = torch.randn_like(x0)

        # Sample x_t from q(x_t | x_0), then train model to recover the injected noise.
        x_t = diffusion.q_sample(x_start=x0, t=t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)

        if step % args.log_every == 0 or step == 1:
            recent = losses[max(0, len(losses) - args.log_every) :]
            avg_recent_loss = float(np.mean(recent))
            progress.set_postfix(step=step, loss=f"{avg_recent_loss:.4f}")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timesteps": args.timesteps,
            "hidden_dim": args.hidden_dim,
            "time_dim": args.time_dim,
            "layers": args.layers,
            "means": means,
            "stds": stds,
            "weights": weights,
            "seed": args.seed,
        },
        checkpoint_path,
    )

    # Create evaluation plots after training.
    model.eval()
    generated = diffusion.p_sample_loop((args.num_gen, 1)).cpu().numpy().squeeze(-1)
    real = sample_mixture_of_gaussians(
        n_samples=args.num_gen,
        means=means,
        stds=stds,
        weights=weights,
        seed=args.seed,
    ).numpy().squeeze(-1)

    plot_loss_curve(losses, output_dir / "training_loss.png", smoothing_window=args.log_every)
    plot_histogram_comparison(real, generated, output_dir / "generated_vs_real.png")

    subset = sample_mixture_of_gaussians(
        n_samples=600,
        means=means,
        stds=stds,
        weights=weights,
        seed=args.seed + 1,
    )
    t_values = [0, args.timesteps // 4, args.timesteps // 2, (3 * args.timesteps) // 4, args.timesteps - 1]
    plot_forward_noising(
        x_start=subset,
        diffusion=diffusion,
        timesteps_to_plot=t_values,
        save_path=output_dir / "forward_noising.png",
    )

    print(f"Training complete. Checkpoint saved to: {checkpoint_path}")
    print(f"Plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
