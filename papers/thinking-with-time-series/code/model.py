"""
model.py — Core model architectures for Thinking with Time-Series (TTS).

Contains:
  - FinancialQueryEncoder: encodes market state + query into conditioning vector
  - TemporalUNet: diffusion-based trajectory generator
  - DiffusionTrajectoryGenerator: wraps the UNet with diffusion schedule
  - RegimeHMM: VIX-conditional hidden Markov model for regime detection
  - TTSFullModel: end-to-end TTS inference pipeline
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FinancialQueryEncoder(nn.Module):
    """Encodes market state and query embedding into a conditioning vector.

    Architecture:
      - Market state branch: processes (lookback, n_assets) via temporal attention
      - Query branch: processes query embedding via MLP
      - Fusion: concatenation + projection to conditioning dimension
    """

    def __init__(
        self,
        n_assets: int = 15,
        lookback: int = 60,
        query_dim: int = 128,
        hidden_dim: int = 256,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.n_assets = n_assets

        # Market state encoder: temporal convolution
        self.state_conv1 = nn.Conv1d(n_assets, hidden_dim, kernel_size=5, padding=2)
        self.state_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.state_pool = nn.AdaptiveAvgPool1d(1)
        self.state_proj = nn.Linear(hidden_dim, cond_dim // 2)

        # VIX + regime embedding
        self.vix_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, cond_dim // 8),
        )

        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cond_dim // 2 - cond_dim // 8),
            nn.ReLU(),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(
        self,
        market_state: torch.Tensor,  # (B, lookback, n_assets)
        vix: torch.Tensor,  # (B,)
        query_emb: torch.Tensor,  # (B, query_dim)
    ) -> torch.Tensor:
        B = market_state.shape[0]

        # Market state: (B, lookback, n_assets) -> (B, n_assets, lookback)
        x = market_state.transpose(1, 2)
        x = F.relu(self.state_conv1(x))
        x = F.relu(self.state_conv2(x))
        x = self.state_pool(x).squeeze(-1)  # (B, hidden_dim)
        state_feat = self.state_proj(x)  # (B, cond_dim//2)

        # VIX
        vix_feat = self.vix_encoder(vix.unsqueeze(-1))  # (B, cond_dim//8)

        # Query
        query_feat = self.query_encoder(query_emb)

        # Concatenate and fuse
        cond = torch.cat([state_feat, vix_feat, query_feat], dim=-1)  # (B, cond_dim)
        cond = self.fusion(cond)

        return cond


class TemporalUNetBlock(nn.Module):
    """A single block in the temporal U-Net with cross-asset attention."""

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, n_assets: int):
        super().__init__()
        self.n_assets = n_assets

        # Temporal self-attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=4, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(in_dim)

        # Cross-asset attention
        self.asset_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=4, batch_first=True
        )
        self.asset_norm = nn.LayerNorm(in_dim)

        # Conditioning injection via FiLM
        self.film_scale = nn.Linear(cond_dim, out_dim)
        self.film_bias = nn.Linear(cond_dim, out_dim)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim),
        )
        self.norm_ff = nn.LayerNorm(out_dim)

        # Timestep embedding
        self.time_proj = nn.Linear(in_dim, in_dim)

        # Downsample/upsample if needed
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,  # (B, T, n_assets, D)
        cond: torch.Tensor,  # (B, cond_dim)
        time_emb: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        B, T, A, D = x.shape

        # Add timestep embedding
        t_emb = self.time_proj(time_emb).unsqueeze(1).unsqueeze(2)  # (B,1,1,D)
        x = x + t_emb

        # Temporal self-attention (attend across time for each asset)
        x_t = x.reshape(B * A, T, D)
        attn_out, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = self.temporal_norm(x_t + attn_out)
        x = x_t.reshape(B, T, A, D)

        # Cross-asset attention (attend across assets for each timestep)
        x_a = x.reshape(B * T, A, D)
        attn_out, _ = self.asset_attn(x_a, x_a, x_a)
        x_a = self.asset_norm(x_a + attn_out)
        x = x_a.reshape(B, T, A, D)

        # FiLM conditioning
        scale = self.film_scale(cond).unsqueeze(1).unsqueeze(2)  # (B,1,1,out_dim)
        bias = self.film_bias(cond).unsqueeze(1).unsqueeze(2)

        # Feedforward
        x = self.proj(x)
        ff_out = self.ff(x)
        x = self.norm_ff(x + ff_out)

        # Apply FiLM
        x = x * (1 + scale) + bias

        return x


class TemporalUNet(nn.Module):
    """U-Net architecture for denoising financial trajectories.

    Input: noisy trajectory (B, T, n_assets) + timestep + conditioning
    Output: predicted noise (B, T, n_assets)
    """

    def __init__(
        self,
        n_assets: int = 15,
        horizon: int = 60,
        hidden_dim: int = 128,
        cond_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.horizon = horizon

        # Input projection
        self.input_proj = nn.Linear(1, hidden_dim)

        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        dims = [hidden_dim]
        for i in range(n_layers):
            out_d = hidden_dim * (2 ** min(i + 1, 3))
            self.encoder.append(
                TemporalUNetBlock(dims[-1], out_d, cond_dim, n_assets)
            )
            dims.append(out_d)

        # Bottleneck
        self.bottleneck = TemporalUNetBlock(
            dims[-1], dims[-1], cond_dim, n_assets
        )

        # Decoder (upsampling path with skip connections)
        self.decoder = nn.ModuleList()
        for i in range(n_layers - 1, -1, -1):
            out_d = hidden_dim * (2 ** min(i, 3))
            in_d = dims[i + 1] + dims[i + 1]  # skip connection
            self.decoder.append(
                TemporalUNetBlock(in_d, out_d, cond_dim, n_assets)
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, T, n_assets) noisy trajectory
        t: torch.Tensor,  # (B,) diffusion timestep
        cond: torch.Tensor,  # (B, cond_dim) conditioning vector
    ) -> torch.Tensor:
        B, T, A = x.shape

        # Project input
        h = self.input_proj(x.unsqueeze(-1))  # (B, T, A, D)

        # Timestep embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (B, D)

        # Encoder
        skips = []
        for block in self.encoder:
            h = block(h, cond, t_emb)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h, cond, t_emb)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            h = torch.cat([h, skip], dim=-1)
            h = block(h, cond, t_emb)

        # Output
        noise_pred = self.output_proj(h).squeeze(-1)  # (B, T, A)
        return noise_pred


class DiffusionTrajectoryGenerator:
    """Wraps TemporalUNet with DDPM diffusion schedule for trajectory generation."""

    def __init__(
        self,
        model: TemporalUNet,
        n_diffusion_steps: int = 8,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.model = model
        self.n_steps = n_diffusion_steps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, n_diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_1ma = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,  # (B, cond_dim)
        shape: Tuple[int, ...],  # (B, T, n_assets)
        device: torch.device,
    ) -> torch.Tensor:
        """Reverse diffusion: generate trajectories from noise."""
        B, T, A = shape
        x = torch.randn(B, T, A, device=device)

        for step in reversed(range(self.n_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            noise_pred = self.model(x, t, cond)

            # DDPM update
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bar[step]
            if step > 0:
                alpha_bar_prev = self.alpha_bar[step - 1]
            else:
                alpha_bar_prev = torch.tensor(1.0)

            # Predicted x0
            x0_pred = (x - self.sqrt_one_minus_alpha_bar[step] * noise_pred) / self.sqrt_alpha_bar[step]
            x0_pred = x0_pred.clamp(-3, 3)  # clip for stability

            # Posterior mean
            posterior_mean = (
                torch.sqrt(alpha_bar_prev) * alpha / (1 - alpha_bar) * x0_pred
                + torch.sqrt(1 - alpha_bar_prev) * (1 - alpha) / (1 - alpha_bar) * x
            )

            if step > 0:
                posterior_var = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha)
                noise = torch.randn_like(x) * torch.sqrt(posterior_var)
                x = posterior_mean + noise
            else:
                x = posterior_mean

        return x

    def training_loss(
        self,
        x0: torch.Tensor,  # (B, T, n_assets) real trajectory
        cond: torch.Tensor,  # (B, cond_dim)
    ) -> torch.Tensor:
        """Compute DDPM training loss."""
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.n_steps, (B,), device=device)
        noise = torch.randn_like(x0)
        x_noisy, _ = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t, cond)
        return F.mse_loss(noise_pred, noise)


class RegimeHMM:
    """VIX-conditional Hidden Markov Model for regime detection on trajectories.

    States: 0=low-vol/bull, 1=high-vol/bear, 2=crisis
    Emissions: trajectory statistics (realized vol, mean return, max drawdown)
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        # Transition matrix (rows sum to 1)
        self.transitions = np.array([
            [0.95, 0.04, 0.01],
            [0.05, 0.88, 0.07],
            [0.02, 0.10, 0.88],
        ])
        # Emission parameters: (mean, std) for each state
        self.emission_params = {
            0: {"vol_mean": 0.12, "vol_std": 0.04, "ret_mean": 0.001, "ret_std": 0.005},
            1: {"vol_mean": 0.25, "vol_std": 0.06, "ret_mean": -0.0005, "ret_std": 0.008},
            2: {"vol_mean": 0.50, "vol_std": 0.15, "ret_mean": -0.003, "ret_std": 0.020},
        }

    def detect_regime(
        self,
        trajectory: np.ndarray,  # (T, n_assets) log returns
        vix_level: float,
    ) -> Tuple[int, np.ndarray]:
        """Detect regime from trajectory statistics.

        Returns: (most_likely_regime, state_probabilities)
        """
        realized_vol = np.std(trajectory) * np.sqrt(252)
        mean_return = np.mean(trajectory)
        max_dd = self._max_drawdown(trajectory.cumsum(axis=0))

        # VIX prior
        vix_probs = self._vix_prior(vix_level)

        # Emission likelihood
        likelihoods = np.zeros(self.n_states)
        for s in range(self.n_states):
            p = self.emission_params[s]
            ll_vol = self._gaussian_ll(realized_vol, p["vol_mean"], p["vol_std"])
            ll_ret = self._gaussian_ll(mean_return, p["ret_mean"], p["ret_std"])
            likelihoods[s] = ll_vol + ll_ret

        # Posterior: prior * likelihood
        posterior = vix_probs * likelihoods
        posterior = posterior / (posterior.sum() + 1e-10)

        return int(np.argmax(posterior)), posterior

    def compute_regime_weights(
        self,
        trajectories: List[np.ndarray],  # list of (T, n_assets) arrays
        vix_level: float,
    ) -> np.ndarray:
        """Compute regime-weighted voting weights for N trajectories."""
        N = len(trajectories)
        weights = np.zeros(N)
        for i, traj in enumerate(trajectories):
            regime, probs = self.detect_regime(traj, vix_level)
            weights[i] = probs[regime]
        # Normalize
        weights = weights / (weights.sum() + 1e-10)
        return weights

    def _vix_prior(self, vix: float) -> np.ndarray:
        """Prior regime probabilities given VIX level."""
        if vix < 20:
            return np.array([0.7, 0.25, 0.05])
        elif vix < 30:
            return np.array([0.2, 0.6, 0.2])
        else:
            return np.array([0.05, 0.35, 0.60])

    @staticmethod
    def _gaussian_ll(x: float, mu: float, sigma: float) -> float:
        return -0.5 * ((x - mu) / (sigma + 1e-8)) ** 2

    @staticmethod
    def _max_drawdown(cum_returns: np.ndarray) -> float:
        running_max = np.maximum.accumulate(cum_returns, axis=0)
        drawdowns = running_max - cum_returns
        return float(np.max(drawdowns))


class TTSFullModel(nn.Module):
    """End-to-end TTS model combining encoder, generator, and aggregator."""

    def __init__(
        self,
        n_assets: int = 15,
        lookback: int = 60,
        horizon: int = 60,
        hidden_dim: int = 128,
        cond_dim: int = 256,
        query_dim: int = 128,
        n_diffusion_steps: int = 8,
        n_trajectories: int = 16,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.horizon = horizon
        self.n_trajectories = n_trajectories

        self.encoder = FinancialQueryEncoder(
            n_assets=n_assets,
            lookback=lookback,
            query_dim=query_dim,
            cond_dim=cond_dim,
        )
        self.unet = TemporalUNet(
            n_assets=n_assets,
            horizon=horizon,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
        )
        self.diffusion = DiffusionTrajectoryGenerator(
            model=self.unet,
            n_diffusion_steps=n_diffusion_steps,
        )
        self.regime_hmm = RegimeHMM(n_states=3)

        # Answer classifier: maps trajectory statistics to answer
        self.answer_head = nn.Sequential(
            nn.Linear(horizon * n_assets + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 classes: negative/neutral/positive
        )

    def forward(
        self,
        market_state: torch.Tensor,  # (B, lookback, n_assets)
        vix: torch.Tensor,  # (B,)
        query_emb: torch.Tensor,  # (B, query_dim)
        future: Optional[torch.Tensor] = None,  # (B, horizon, n_assets) for training
    ) -> dict:
        cond = self.encoder(market_state, vix, query_emb)
        B = cond.shape[0]
        device = cond.device

        if self.training and future is not None:
            # Training: compute diffusion loss
            loss = self.diffusion.training_loss(future, cond)
            return {"loss": loss, "cond": cond}
        else:
            # Inference: generate trajectories and aggregate
            shape = (B, self.horizon, self.n_assets)
            trajectories = []
            for _ in range(self.n_trajectories):
                traj = self.diffusion.generate(cond, shape, device)
                trajectories.append(traj)

            # Stack: (B, N, T, A)
            traj_stack = torch.stack(trajectories, dim=1)

            # Extract trajectory statistics for answer prediction
            stats = self._extract_stats(traj_stack)  # (B, T*A + 16)

            # Predict answer
            logits = self.answer_head(stats)
            return {"logits": logits, "trajectories": traj_stack}

    def _extract_stats(self, traj_stack: torch.Tensor) -> torch.Tensor:
        """Extract statistical features from trajectory ensemble.

        traj_stack: (B, N, T, A)
        Returns: (B, T*A + 16) feature vector
        """
        B, N, T, A = traj_stack.shape

        # Mean trajectory
        mean_traj = traj_stack.mean(dim=1)  # (B, T, A)
        mean_flat = mean_traj.reshape(B, T * A)

        # Cross-trajectory statistics
        traj_returns = traj_stack[:, :, -1, :].mean(dim=-1)  # (B, N)
        traj_vols = traj_stack.std(dim=2).mean(dim=-1)  # (B, N)

        stats_16 = torch.stack([
            traj_returns.mean(dim=1),
            traj_returns.std(dim=1),
            traj_returns.min(dim=1).values,
            traj_returns.max(dim=1).values,
            traj_vols.mean(dim=1),
            traj_vols.std(dim=1),
            (traj_returns < 0).float().mean(dim=1),  # fraction negative
            (traj_returns < -0.1).float().mean(dim=1),  # tail risk
            traj_stack[:, :, -1, :].mean(dim=(1, 2)),  # avg final return
            traj_stack.std(dim=1).mean(dim=(1, 2)),  # disagreement
            traj_stack.abs().max(dim=1).values.max(dim=-1).values,  # max move
            traj_stack[:, :, :, :].std(dim=2).mean(dim=(1, 2)),  # avg vol
            (traj_stack.min(dim=2).values < -0.2).float().mean(dim=(1, 2)),  # crash prob
            traj_stack[:, :, -1, :] .std(dim=1).mean(dim=-1),  # final dispersion
            traj_stack.cumsum(dim=2).max(dim=2).values.mean(dim=(1, 2)),  # avg peak
            traj_stack.cumsum(dim=2).min(dim=2).values.mean(dim=(1, 2)),  # avg trough
        ], dim=-1)  # (B, 16)

        return torch.cat([mean_flat, stats_16], dim=-1)
