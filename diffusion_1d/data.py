"""Data utilities for a 1D toy distribution."""

from __future__ import annotations

import numpy as np
import torch


def sample_mixture_of_gaussians(
    n_samples: int,
    means: tuple[float, ...] = (-2.0, 0.0, 2.0),
    stds: tuple[float, ...] = (0.25, 0.25, 0.25),
    weights: tuple[float, ...] = (0.4, 0.2, 0.4),
    seed: int | None = None,
) -> torch.Tensor:
    """Sample a 1D mixture of Gaussians.

    Returns a tensor of shape (n_samples, 1).
    """
    if not (len(means) == len(stds) == len(weights)):
        raise ValueError("means, stds, and weights must have the same length.")

    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr < 0):
        raise ValueError("Mixture weights must be non-negative.")
    weights_arr = weights_arr / weights_arr.sum()

    rng = np.random.default_rng(seed)
    component_ids = rng.choice(len(means), size=n_samples, p=weights_arr)

    samples = np.empty(n_samples, dtype=np.float32)
    for idx, (mu, sigma) in enumerate(zip(means, stds)):
        mask = component_ids == idx
        count = int(mask.sum())
        if count > 0:
            samples[mask] = rng.normal(loc=mu, scale=sigma, size=count)

    return torch.from_numpy(samples).unsqueeze(-1)
