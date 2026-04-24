"""Tiny VAE for MNIST -- stage 1.

Mirrors the ``z = mu + sigma * eps`` reparameterization used by the
Wav-VAE bottleneck in LongCat-AudioDiT (section 3.1).  The decoder
outputs Bernoulli-style logits and we train with BCE + KL.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z

    @torch.no_grad()
    def sample(self, num: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num, self.latent_dim, device=device)
        logits = self.decode(z)
        img = torch.sigmoid(logits).view(num, 1, 28, 28)
        return img


def vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_flat = target.flatten(1)
    recon = F.binary_cross_entropy_with_logits(logits, target_flat, reduction="none").sum(dim=1).mean()
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    loss = recon + beta * kl
    return loss, {"loss": loss.item(), "recon": recon.item(), "kl": kl.item()}
