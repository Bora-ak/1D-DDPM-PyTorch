"""Small MLP denoiser for 1D DDPM."""

from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    This maps each integer timestep t to a continuous vector so the network
    can condition its noise prediction on the current diffusion step.
    """
    half = dim // 2
    device = timesteps.device

    # Log-spaced frequencies used by transformer-style sinusoidal embeddings.
    freq_exp = -math.log(10_000) * torch.arange(half, device=device) / max(half - 1, 1)
    freqs = torch.exp(freq_exp)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class DenoiseMLP(nn.Module):
    """Predict epsilon (added Gaussian noise) from x_t and t."""

    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 2:
            raise ValueError("num_hidden_layers must be at least 2.")
        self.time_dim = time_dim

        layers: list[nn.Module] = []
        in_dim = 1 + time_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_dim)
        inputs = torch.cat([x_t, t_emb], dim=1)
        return self.net(inputs)
