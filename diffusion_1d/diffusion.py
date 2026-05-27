"""DDPM forward and reverse diffusion utilities."""

from __future__ import annotations

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    """Linear beta schedule used by the forward diffusion process."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(coeff: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract per-sample coefficients at time indices t and reshape for broadcasting."""
    batch_size = t.shape[0]
    out = coeff.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DDPM1D:
    """Minimal DDPM implementation for 1D data."""

    def __init__(
        self,
        model: torch.nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.timesteps = timesteps
        self.device = device

        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance for q(x_{t-1} | x_t, x_0), used in reverse sampling.
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(posterior_variance, min=1e-20)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Reverse diffusion: sample x_{t-1} from p_theta(x_{t-1} | x_t)."""
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        # DDPM mean using the model prediction epsilon_theta(x_t, t).
        eps_theta = self.model(x_t, t)
        model_mean = sqrt_recip_alpha_t * (
            x_t - (betas_t / sqrt_one_minus_alpha_bar_t) * eps_theta
        )

        if t_index == 0:
            return model_mean

        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Start from Gaussian noise and run reverse diffusion to get x_0 samples."""
        x_t = torch.randn(shape, device=self.device)
        for t_index in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), t_index, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, t_index)
        return x_t
