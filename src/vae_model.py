"""
vae_model.py
------------
A genuine Variational Autoencoder (encoder network, reparameterization
trick, KL-regularized training loop) for generating synthetic minority-
class (default=1) tabular records.

This replaces the earlier PCA-based approximation used when PyTorch could
not be installed in this environment (see model_validation_report.md for
that history) — this is now a real VAE, trained exclusively on the
minority class of the TRAINING fold only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class VAETrainingResult:
    model: VAE
    mu_scaler: np.ndarray
    std_scaler: np.ndarray
    final_recon_loss: float
    final_kld_loss: float
    loss_history: list


def train_vae(
    X_minority: np.ndarray,
    hidden_dim: int = 64,
    latent_dim: int = 12,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    kld_weight: float = 0.02,
    seed: int = 42,
) -> VAETrainingResult:
    """
    Trains a VAE on X_minority ONLY (the minority-class rows of the
    TRAINING fold — callers must never pass test-fold rows in here).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    mu_scaler = X_minority.mean(axis=0)
    std_scaler = X_minority.std(axis=0) + 1e-8
    X_norm = (X_minority - mu_scaler) / std_scaler

    input_dim = X_norm.shape[1]
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    n = X_tensor.shape[0]
    loss_history = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_recon, epoch_kld = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = X_tensor[idx]

            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction="mean")
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_weight * kld_loss
            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item() * len(idx)
            epoch_kld += kld_loss.item() * len(idx)

        loss_history.append({"epoch": epoch, "recon_loss": epoch_recon / n, "kld_loss": epoch_kld / n})

    return VAETrainingResult(
        model=model,
        mu_scaler=mu_scaler,
        std_scaler=std_scaler,
        final_recon_loss=loss_history[-1]["recon_loss"],
        final_kld_loss=loss_history[-1]["kld_loss"],
        loss_history=loss_history,
    )


def generate_synthetic_samples(result: VAETrainingResult, n_samples: int, seed: int = 42) -> np.ndarray:
    torch.manual_seed(seed)
    model = result.model
    model.eval()
    latent_dim = model.fc_mu.out_features
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        synthetic_norm = model.decode(z).numpy()
    synthetic = synthetic_norm * result.std_scaler + result.mu_scaler
    return synthetic
