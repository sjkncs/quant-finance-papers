"""
model.py — SAM-Vol model architecture for volatility surface reconstruction.

Contains:
  - SparseQuoteEncoder: Point-Transformer for irregularly spaced option quotes
  - NeuralImplicitSurface: continuous IV function of (log-moneyness, maturity)
  - NoArbitrageRegularizer: differentiable butterfly/calendar spread constraints
  - SAMVolModel: end-to-end model combining all components
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class PointTransformerBlock(nn.Module):
    """Self-attention block for unordered point sets (option quotes)."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, D) point features
        mask: (B, N) binary mask for valid points
        """
        normed = self.norm1(x)
        # Create attention mask from point mask
        if mask is not None:
            attn_mask = (1 - mask).bool()  # True = masked
        else:
            attn_mask = None

        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class SparseQuoteEncoder(nn.Module):
    """Encodes sparse option quotes into point features and a global context vector.

    Input: point_cloud (B, N, 7) — each point has (log_m, tau, iv, spread, price, delta, gamma)
    Output: point_features (B, N, D), global_context (B, D)
    """

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional embedding for log-moneyness and maturity
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
        )

        self.blocks = nn.ModuleList([
            PointTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        point_cloud: torch.Tensor,  # (B, N, 7)
        mask: torch.Tensor,  # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project input features
        h = self.input_proj(point_cloud)  # (B, N, D)

        # Add positional embedding from coordinates (log_m, tau)
        coords = point_cloud[:, :, :2]  # (B, N, 2)
        h = h + self.pos_mlp(coords)

        # Point-Transformer layers
        for block in self.blocks:
            h = block(h, mask)

        h = self.norm(h)

        # Global context: masked mean pooling
        mask_exp = mask.unsqueeze(-1)  # (B, N, 1)
        global_ctx = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)  # (B, D)

        return h, global_ctx


class NeuralImplicitSurface(nn.Module):
    """Neural implicit function that maps (log-moneyness, maturity) to implied volatility.

    Uses both global context and local feature interpolation from nearby quotes.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 6,
        bandwidth: float = 0.1,
    ):
        super().__init__()
        self.bandwidth = bandwidth

        # Coordinate encoding (Fourier features)
        self.coord_freq = nn.Parameter(
            torch.randn(16) * 2.0, requires_grad=False
        )  # learnable frequencies

        coord_input_dim = 2 + 2 * 16 * 2  # (k, tau) + sin/cos for each freq and coord
        input_dim = coord_input_dim + d_model + d_model  # coords + global_ctx + local_feat

        # MLP decoder
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])
            # Skip connection at middle layers
            if i == n_layers // 2 - 1:
                layers.append(SkipConnection(input_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def encode_coords(self, k: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Fourier feature encoding of coordinates."""
        coords = torch.stack([k, tau], dim=-1)  # (B, Q, 2)
        features = [coords]
        for freq in self.coord_freq:
            features.append(torch.sin(coords * freq))
            features.append(torch.cos(coords * freq))
        return torch.cat(features, dim=-1)

    def interpolate_local_features(
        self,
        query_k: torch.Tensor,  # (B, Q)
        query_tau: torch.Tensor,  # (B, Q)
        point_cloud: torch.Tensor,  # (B, N, 7) — original quote coords
        point_features: torch.Tensor,  # (B, N, D)
        mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """Attention-weighted interpolation of point features at query locations."""
        B, Q = query_k.shape
        N = point_cloud.shape[1]
        D = point_features.shape[2]

        # Query coordinates
        query_coords = torch.stack([query_k, query_tau], dim=-1)  # (B, Q, 2)
        point_coords = point_cloud[:, :, :2]  # (B, N, 2)

        # Compute distances
        dists = torch.cdist(query_coords, point_coords)  # (B, Q, N)

        # Attention weights (inverse distance, temperature-scaled)
        weights = torch.exp(-dists ** 2 / (2 * self.bandwidth ** 2))  # (B, Q, N)
        weights = weights * mask.unsqueeze(1)  # mask out invalid points
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum of features
        local_feat = torch.bmm(weights, point_features)  # (B, Q, D)
        return local_feat

    def forward(
        self,
        query_k: torch.Tensor,  # (B, Q)
        query_tau: torch.Tensor,  # (B, Q)
        global_ctx: torch.Tensor,  # (B, D)
        point_cloud: torch.Tensor,  # (B, N, 7)
        point_features: torch.Tensor,  # (B, N, D)
        mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """Predict implied volatility at query points."""
        B, Q = query_k.shape
        D = global_ctx.shape[1]

        # Encode coordinates
        coord_feat = self.encode_coords(query_k, query_tau)  # (B, Q, coord_dim)

        # Local features
        local_feat = self.interpolate_local_features(
            query_k, query_tau, point_cloud, point_features, mask
        )  # (B, Q, D)

        # Global context broadcast
        global_broadcast = global_ctx.unsqueeze(1).expand(-1, Q, -1)  # (B, Q, D)

        # Concatenate all features
        features = torch.cat([coord_feat, global_broadcast, local_feat], dim=-1)

        # Predict IV
        iv = self.mlp(features).squeeze(-1)  # (B, Q)

        # Ensure positive IV via softplus
        iv = F.softplus(iv) + 0.01

        return iv


class SkipConnection(nn.Module):
    """Skip connection that adds input features to hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NoArbitrageRegularizer:
    """Differentiable no-arbitrage constraints for volatility surfaces."""

    @staticmethod
    def butterfly_constraint(
        model_fn,  # function(k, tau) -> iv
        k_grid: torch.Tensor,  # (B, n_k)
        tau: float,
        S: float = 100.0,
        r: float = 0.05,
        dk: float = 0.01,
    ) -> torch.Tensor:
        """Enforce d^2C/dK^2 >= 0 (positive risk-neutral density).

        Approximated using finite differences of call prices.
        """
        B = k_grid.shape[0]

        # Evaluate IV at k-dk, k, k+dk
        tau_t = torch.full_like(k_grid[:, :1], tau)
        iv_left = model_fn(k_grid - dk, tau_t)
        iv_center = model_fn(k_grid, tau_t)
        iv_right = model_fn(k_grid + dk, tau_t)

        # Convert to call prices
        K_left = S * torch.exp(k_grid - dk)
        K_center = S * torch.exp(k_grid)
        K_right = S * torch.exp(k_grid + dk)

        C_left = NoArbitrageRegularizer._bs_call(S, K_left, tau, r, iv_left)
        C_center = NoArbitrageRegularizer._bs_call(S, K_center, tau, r, iv_center)
        C_right = NoArbitrageRegularizer._bs_call(S, K_right, tau, r, iv_right)

        # Second derivative: C(K-dK) - 2*C(K) + C(K+dK) >= 0
        d2C = C_left - 2 * C_center + C_right
        violation = torch.relu(-d2C).mean()

        return violation

    @staticmethod
    def calendar_constraint(
        model_fn,
        k: torch.Tensor,  # (B, n_k)
        tau1: float,
        tau2: float,  # tau2 > tau1
        S: float = 100.0,
    ) -> torch.Tensor:
        """Enforce T * sigma^2 increases with T (calendar spread >= 0)."""
        tau1_t = torch.full_like(k[:, :1], tau1)
        tau2_t = torch.full_like(k[:, :1], tau2)

        iv1 = model_fn(k, tau1_t)
        iv2 = model_fn(k, tau2_t)

        # Total variance should increase
        w1 = iv1 ** 2 * tau1
        w2 = iv2 ** 2 * tau2
        violation = torch.relu(w1 - w2).mean()

        return violation

    @staticmethod
    def compute_arbitrage_loss(
        model_fn,
        k_grid: torch.Tensor,
        tau_grid: torch.Tensor,
        S: float = 100.0,
        r: float = 0.05,
    ) -> torch.Tensor:
        """Compute total no-arbitrage penalty across the surface grid."""
        B = k_grid.shape[0]
        n_tau = tau_grid.shape[1]

        butterfly_loss = torch.tensor(0.0, device=k_grid.device)
        calendar_loss = torch.tensor(0.0, device=k_grid.device)

        # Butterfly constraint at each maturity
        for j in range(n_tau):
            tau = tau_grid[0, j].item()
            butterfly_loss = butterfly_loss + NoArbitrageRegularizer.butterfly_constraint(
                model_fn, k_grid, tau, S, r
            )

        # Calendar constraint between consecutive maturities
        for j in range(n_tau - 1):
            tau1 = tau_grid[0, j].item()
            tau2 = tau_grid[0, j + 1].item()
            calendar_loss = calendar_loss + NoArbitrageRegularizer.calendar_constraint(
                model_fn, k_grid, tau1, tau2, S
            )

        return (butterfly_loss / n_tau + calendar_loss / max(n_tau - 1, 1)) * 0.5

    @staticmethod
    def _bs_call(S, K, T, r, sigma):
        """Differentiable Black-Scholes call price using torch."""
        T = max(T, 1e-8)
        d1 = (torch.log(S / K + 1e-8) + (r + 0.5 * sigma ** 2) * T) / (
            sigma * np.sqrt(T) + 1e-8
        )
        d2 = d1 - sigma * np.sqrt(T)
        return S * torch.distributions.Normal(0, 1).cdf(d1) - K * np.exp(-r * T) * torch.distributions.Normal(0, 1).cdf(d2)


class SAMVolModel(nn.Module):
    """End-to-end SAM-Vol model for volatility surface reconstruction."""

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        n_encoder_layers: int = 4,
        n_heads: int = 8,
        hidden_dim: int = 512,
        n_surface_layers: int = 6,
        bandwidth: float = 0.15,
        arb_weight: float = 10.0,
    ):
        super().__init__()
        self.encoder = SparseQuoteEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
        )
        self.surface = NeuralImplicitSurface(
            d_model=d_model,
            hidden_dim=hidden_dim,
            n_layers=n_surface_layers,
            bandwidth=bandwidth,
        )
        self.arb_weight = arb_weight
        self.regularizer = NoArbitrageRegularizer()

    def forward(
        self,
        point_cloud: torch.Tensor,  # (B, N, 7)
        mask: torch.Tensor,  # (B, N)
        eval_k: torch.Tensor,  # (B, n_k)
        eval_tau: torch.Tensor,  # (B, n_tau)
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct volatility surface from sparse quotes."""
        # Encode quotes
        point_features, global_ctx = self.encoder(point_cloud, mask)

        # Create evaluation grid
        B = point_cloud.shape[0]
        n_k = eval_k.shape[1]
        n_tau = eval_tau.shape[1]

        # Flatten grid to query points
        k_flat = eval_k.unsqueeze(2).expand(-1, -1, n_tau).reshape(B, -1)  # (B, n_k * n_tau)
        tau_flat = eval_tau.unsqueeze(1).expand(-1, n_k, -1).reshape(B, -1)  # (B, n_k * n_tau)

        # Predict IV at all grid points
        iv_flat = self.surface(
            k_flat, tau_flat, global_ctx, point_cloud, point_features, mask
        )

        # Reshape to surface
        iv_surface = iv_flat.reshape(B, n_k, n_tau)  # (B, n_k, n_tau)

        return {
            "iv_surface": iv_surface,
            "global_context": global_ctx,
        }

    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        gt_surface: torch.Tensor,  # (B, n_k, n_tau)
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss: reconstruction + no-arbitrage."""
        iv_surface = output["iv_surface"]

        # Reconstruction loss (MSE on implied vol)
        recon_loss = F.mse_loss(iv_surface, gt_surface)

        # No-arbitrage loss (simplified: use finite differences on the predicted surface)
        arb_loss = self._compute_arb_loss(iv_surface)

        total_loss = recon_loss + self.arb_weight * arb_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "arb_loss": arb_loss,
        }

    def _compute_arb_loss(self, iv_surface: torch.Tensor) -> torch.Tensor:
        """Simplified no-arbitrage loss using finite differences on the predicted surface."""
        B, n_k, n_tau = iv_surface.shape

        # Butterfly constraint: d^2(iv)/dk^2 should not create negative density
        # Approximate: second differences in strike direction should be moderate
        if n_k >= 3:
            d2_iv = iv_surface[:, 2:, :] - 2 * iv_surface[:, 1:-1, :] + iv_surface[:, :-2, :]
            # Penalize extreme negative curvature (which suggests negative density)
            butterfly_penalty = torch.relu(-d2_iv - 0.05).mean()
        else:
            butterfly_penalty = torch.tensor(0.0, device=iv_surface.device)

        # Calendar constraint: total variance should increase with maturity
        if n_tau >= 2:
            # Use a representative ATM slice
            atm_idx = n_k // 2
            total_var = iv_surface[:, atm_idx, :] ** 2
            # Maturity spacing (approximate)
            tau_spacing = torch.tensor(
                [1/52, 1/26, 1/12, 1/6, 1/4, 1/2, 3/4, 1, 1.5, 2][:n_tau],
                device=iv_surface.device,
            )
            weighted_var = total_var * tau_spacing.unsqueeze(0)
            # Variance should increase (approximately)
            if weighted_var.shape[1] >= 2:
                d_wvar = weighted_var[:, 1:] - weighted_var[:, :-1]
                calendar_penalty = torch.relu(-d_wvar).mean()
            else:
                calendar_penalty = torch.tensor(0.0, device=iv_surface.device)
        else:
            calendar_penalty = torch.tensor(0.0, device=iv_surface.device)

        return butterfly_penalty + calendar_penalty

    def count_arbitrage_violations(
        self, iv_surface: torch.Tensor, n_k: int, n_tau: int
    ) -> float:
        """Count percentage of grid points with arbitrage violations."""
        B = iv_surface.shape[0]
        violations = 0
        total = 0

        for b in range(B):
            surf = iv_surface[b]  # (n_k, n_tau)
            # Check butterfly
            if n_k >= 3:
                d2 = surf[2:, :] - 2 * surf[1:-1, :] + surf[:-2, :]
                violations += (d2 < -0.05).sum().item()
                total += d2.numel()
            # Check calendar
            if n_tau >= 2:
                tv = surf ** 2
                dtv = tv[:, 1:] - tv[:, :-1]
                violations += (dtv < 0).sum().item()
                total += dtv.numel()

        return violations / max(total, 1)
