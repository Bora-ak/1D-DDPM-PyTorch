"""Plotting helpers for training and sampling results."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# Use a local writable cache/config location and non-interactive backend.
_mpl_config_dir = Path("outputs/.mplconfig")
_cache_dir = Path("outputs/.cache")
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(losses: list[float], save_path: str | Path, smoothing_window: int = 100) -> None:
    _ensure_parent(save_path)
    loss_arr = np.asarray(losses, dtype=np.float32)
    if smoothing_window > 1 and len(loss_arr) >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float32) / float(smoothing_window)
        smoothed = np.convolve(loss_arr, kernel, mode="valid")
        smooth_x = np.arange(smoothing_window - 1, len(loss_arr))
    else:
        smoothed = loss_arr
        smooth_x = np.arange(len(loss_arr))

    plt.figure(figsize=(7, 4))
    plt.plot(loss_arr, color="tab:blue", alpha=0.3, linewidth=1.0, label="Raw loss")
    plt.plot(smooth_x, smoothed, color="tab:blue", linewidth=2.0, label=f"Moving avg ({smoothing_window})")
    plt.title("Training Loss (Noise Prediction MSE)")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_histogram_comparison(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    save_path: str | Path,
    bins: int = 80,
    x_min: float = -5.0,
    x_max: float = 5.0,
) -> None:
    _ensure_parent(save_path)
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    plt.figure(figsize=(7, 4))
    plt.hist(real_samples, bins=bin_edges, density=True, alpha=0.6, label="Real data")
    plt.hist(generated_samples, bins=bin_edges, density=True, alpha=0.6, label="Generated")
    plt.title("Real vs Generated 1D Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def plot_forward_noising(
    x_start: torch.Tensor,
    diffusion,
    timesteps_to_plot: list[int],
    save_path: str | Path,
    bins: int = 80,
) -> None:
    """Visualize how q(x_t | x_0) becomes noisier as t increases."""
    _ensure_parent(save_path)
    fig, axes = plt.subplots(1, len(timesteps_to_plot), figsize=(4 * len(timesteps_to_plot), 3.4))
    if len(timesteps_to_plot) == 1:
        axes = [axes]

    x_start = x_start.to(diffusion.device)
    for ax, t_value in zip(axes, timesteps_to_plot):
        t = torch.full((x_start.shape[0],), int(t_value), device=diffusion.device, dtype=torch.long)
        x_t = diffusion.q_sample(x_start, t)
        ax.hist(x_t.squeeze(-1).cpu().numpy(), bins=bins, density=True, color="tab:orange", alpha=0.8)
        ax.set_title(f"t = {t_value}")
        ax.set_xlabel("x_t")
        ax.set_ylabel("Density")
        ax.set_xlim(-5, 5)

    fig.suptitle("Forward Diffusion (Noising) Process", y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
