"""
data.py — Synthetic volatility surface data generation for SAM-Vol.

Generates synthetic option quotes and volatility surfaces using Heston,
SABR, and SVI models for training and evaluation without external datasets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm


@dataclass
class OptionQuote:
    """A single option quote observation."""

    strike: float
    maturity: float  # years
    implied_vol: float
    bid_ask_spread: float
    call_price: float
    underlying_price: float


@dataclass
class VolSurfaceSnapshot:
    """A single volatility surface snapshot."""

    underlying_price: float
    risk_free_rate: float
    observed_quotes: List[OptionQuote]
    true_surface: np.ndarray  # (n_strikes, n_maturities) true IV surface
    strike_grid: np.ndarray  # (n_strikes,) log-moneyness grid
    maturity_grid: np.ndarray  # (n_maturities,) maturity grid
    asset_class: str


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * max(T, 1e-8)), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def svi_surface(
    k: np.ndarray,  # log-moneyness grid
    tau: float,  # maturity
    a: float = 0.04,
    b: float = 0.1,
    rho: float = -0.3,
    m: float = -0.05,
    sigma: float = 0.1,
) -> np.ndarray:
    """SVI parametrization of implied variance for a single maturity slice.

    Total variance w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    Implied vol = sqrt(w(k) / tau)
    """
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    w = np.maximum(w, 1e-6)  # ensure positive variance
    iv = np.sqrt(w / max(tau, 1e-8))
    return iv


def generate_heston_surface(
    n_strikes: int = 20,
    n_maturities: int = 10,
    S: float = 100.0,
    r: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a volatility surface using randomized Heston-like parameters.

    Returns: (strike_grid, maturity_grid, iv_surface)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random Heston-like parameters
    v0 = rng.uniform(0.01, 0.08)  # initial variance
    kappa = rng.uniform(0.5, 3.0)  # mean reversion speed
    theta = rng.uniform(0.01, 0.06)  # long-run variance
    xi = rng.uniform(0.1, 0.8)  # vol of vol
    rho_sv = rng.uniform(-0.8, -0.1)  # correlation

    # Log-moneyness grid: -0.3 to +0.3 (roughly 75% to 135% moneyness)
    k_grid = np.linspace(-0.3, 0.3, n_strikes)

    # Maturity grid: 7 days to 2 years
    tau_grid = np.array([7, 14, 30, 60, 90, 180, 270, 365, 540, 730]) / 365.0
    tau_grid = tau_grid[:n_maturities]

    # Generate IV surface using Heston approximation
    iv_surface = np.zeros((n_strikes, n_maturities))
    for j, tau in enumerate(tau_grid):
        for i, k in enumerate(k_grid):
            # Heston implied variance approximation
            # (simplified: using moment-matching expansion)
            var_avg = theta + (v0 - theta) * (1 - np.exp(-kappa * tau)) / (kappa * tau + 1e-8)
            skew = rho_sv * xi * (1 - np.exp(-kappa * tau)) / (kappa * tau + 1e-8)
            smile_width = xi ** 2 / (4 * kappa) * (1 - np.exp(-2 * kappa * tau)) / (kappa * tau + 1e-8)

            iv_sq = var_avg + skew * k + smile_width * k ** 2
            iv_sq = max(iv_sq, 1e-6)
            iv_surface[i, j] = np.sqrt(iv_sq)

    return k_grid, tau_grid, iv_surface


def generate_sabr_surface(
    n_strikes: int = 20,
    n_maturities: int = 10,
    S: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a volatility surface using randomized SABR-like parameters."""
    if rng is None:
        rng = np.random.default_rng()

    alpha = rng.uniform(0.1, 0.4)
    beta = rng.uniform(0.3, 0.9)
    rho = rng.uniform(-0.6, -0.1)
    nu = rng.uniform(0.1, 0.5)

    k_grid = np.linspace(-0.3, 0.3, n_strikes)
    tau_grid = np.array([7, 14, 30, 60, 90, 180, 270, 365, 540, 730]) / 365.0
    tau_grid = tau_grid[:n_maturities]

    iv_surface = np.zeros((n_strikes, n_maturities))
    for j, tau in enumerate(tau_grid):
        for i, k in enumerate(k_grid):
            # SABR implied vol approximation (Hagan formula, simplified)
            K_eff = S * np.exp(k)
            z = nu / alpha * (S * K_eff) ** ((1 - beta) / 2) * k
            x = np.log((np.sqrt(1 + z ** 2 / 4) + z / 2 + 1e-10) / (1 + 1e-10))
            if abs(x) < 1e-6:
                x = 1.0
            else:
                x = z / (x + 1e-10)

            F = S * np.exp(0)  # forward ≈ spot for simplicity
            mid = (F * K_eff) ** ((1 - beta) / 2)
            term1 = alpha / (mid * (1 + (1 - beta) ** 2 / 24 * k ** 2))
            term2 = x

            # Correction terms
            corr = 1 + (
                (1 - beta) ** 2 / 24 * alpha ** 2 / mid ** 2
                + rho * beta * nu * alpha / (4 * mid)
                + (2 - 3 * rho ** 2) / 24 * nu ** 2
            ) * tau

            iv_surface[i, j] = max(term1 * term2 * corr, 0.01)

    return k_grid, tau_grid, iv_surface


def sample_sparse_quotes(
    k_grid: np.ndarray,
    tau_grid: np.ndarray,
    iv_surface: np.ndarray,
    n_quotes: int = 30,
    S: float = 100.0,
    r: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> List[OptionQuote]:
    """Sample sparse option quotes from a full surface with noise."""
    if rng is None:
        rng = np.random.default_rng()

    n_k, n_tau = iv_surface.shape
    quotes = []

    # Sample random strike-maturity pairs
    k_indices = rng.choice(n_k, size=n_quotes, replace=True)
    tau_indices = rng.choice(n_tau, size=n_quotes, replace=True)

    for ki, ti in zip(k_indices, tau_indices):
        k = k_grid[ki]
        tau = tau_grid[ti]
        true_iv = iv_surface[ki, ti]

        # Add observation noise proportional to moneyness distance from ATM
        noise = rng.normal(0, 0.005 + 0.01 * abs(k))
        observed_iv = max(true_iv + noise, 0.01)

        K = S * np.exp(k)
        bid_ask = rng.uniform(0.005, 0.03) * (1 + abs(k))
        call_price = black_scholes_call(S, K, tau, r, observed_iv)

        quotes.append(OptionQuote(
            strike=K,
            maturity=tau,
            implied_vol=observed_iv,
            bid_ask_spread=bid_ask,
            call_price=call_price,
            underlying_price=S,
        ))

    return quotes


class SAMVolDataset(Dataset):
    """PyTorch dataset for SAM-Vol training."""

    def __init__(
        self,
        n_surfaces: int = 1000,
        n_quotes_range: Tuple[int, int] = (20, 60),
        n_eval_strikes: int = 20,
        n_eval_maturities: int = 10,
        seed: int = 42,
    ):
        self.n_surfaces = n_surfaces
        self.n_quotes_range = n_quotes_range
        self.n_eval_strikes = n_eval_strikes
        self.n_eval_maturities = n_eval_maturities
        self.rng = np.random.default_rng(seed)
        self.S = 100.0
        self.r = 0.05

        # Pre-generate surfaces
        self.surfaces = []
        for i in range(n_surfaces):
            if i % 3 == 0:
                k_grid, tau_grid, iv_surface = generate_heston_surface(
                    n_eval_strikes, n_eval_maturities, self.S, self.r, self.rng
                )
            elif i % 3 == 1:
                k_grid, tau_grid, iv_surface = generate_sabr_surface(
                    n_eval_strikes, n_eval_maturities, self.S, self.rng
                )
            else:
                # SVI
                k_grid = np.linspace(-0.3, 0.3, n_eval_strikes)
                tau_grid = np.array([7, 14, 30, 60, 90, 180, 270, 365, 540, 730]) / 365.0
                tau_grid = tau_grid[:n_eval_maturities]
                iv_surface = np.zeros((n_eval_strikes, n_eval_maturities))
                params = self.rng.uniform(
                    [0.01, 0.05, -0.5, -0.1, 0.05],
                    [0.08, 0.2, 0.0, 0.1, 0.3],
                )
                for j, tau in enumerate(tau_grid):
                    iv_surface[:, j] = svi_surface(
                        k_grid, tau,
                        a=params[0], b=params[1], rho=params[2],
                        m=params[3], sigma=params[4],
                    )

            n_quotes = self.rng.integers(n_quotes_range[0], n_quotes_range[1] + 1)
            quotes = sample_sparse_quotes(
                k_grid, tau_grid, iv_surface, n_quotes, self.S, self.r, self.rng
            )
            self.surfaces.append((k_grid, tau_grid, iv_surface, quotes))

    def __len__(self) -> int:
        return self.n_surfaces

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        k_grid, tau_grid, iv_surface, quotes = self.surfaces[idx]

        # Encode observed quotes as point cloud
        n_quotes = len(quotes)
        point_cloud = np.zeros((n_quotes, 7))
        for i, q in enumerate(quotes):
            log_m = np.log(q.strike / self.S)
            point_cloud[i, 0] = log_m  # log-moneyness
            point_cloud[i, 1] = q.maturity  # time to maturity
            point_cloud[i, 2] = q.implied_vol  # observed IV
            point_cloud[i, 3] = q.bid_ask_spread  # spread
            point_cloud[i, 4] = q.call_price / self.S  # normalized price
            # Greeks (approximate)
            d1 = (log_m + (self.r + 0.5 * q.implied_vol ** 2) * q.maturity) / (
                q.implied_vol * np.sqrt(max(q.maturity, 1e-8)) + 1e-8
            )
            point_cloud[i, 5] = norm.cdf(d1)  # delta
            point_cloud[i, 6] = norm.pdf(d1) / (
                self.S * q.implied_vol * np.sqrt(max(q.maturity, 1e-8)) + 1e-8
            )  # gamma proxy

        # Pad to max quotes
        max_q = 60
        if point_cloud.shape[0] < max_q:
            pad = np.zeros((max_q - point_cloud.shape[0], 7))
            mask = np.zeros(max_q)
            mask[: point_cloud.shape[0]] = 1.0
            point_cloud = np.concatenate([point_cloud, pad], axis=0)
        else:
            point_cloud = point_cloud[:max_q]
            mask = np.ones(max_q)

        # Build evaluation grid
        eval_k = np.linspace(-0.3, 0.3, self.n_eval_strikes)
        eval_tau = np.array([7, 14, 30, 60, 90, 180, 270, 365, 540, 730]) / 365.0
        eval_tau = eval_tau[: self.n_eval_maturities]

        # Ground truth surface (interpolated to eval grid)
        gt_surface = iv_surface  # already on eval grid

        return {
            "point_cloud": torch.tensor(point_cloud, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "eval_k": torch.tensor(eval_k, dtype=torch.float32),
            "eval_tau": torch.tensor(eval_tau, dtype=torch.float32),
            "gt_surface": torch.tensor(gt_surface, dtype=torch.float32),
            "underlying_price": torch.tensor(self.S, dtype=torch.float32),
        }


def build_samvol_dataloaders(
    n_surfaces: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders."""
    n_train = int(n_surfaces * 0.7)
    n_val = int(n_surfaces * 0.15)
    n_test = n_surfaces - n_train - n_val

    train_ds = SAMVolDataset(n_surfaces=n_train, seed=seed)
    val_ds = SAMVolDataset(n_surfaces=n_val, seed=seed + 1)
    test_ds = SAMVolDataset(n_surfaces=n_test, seed=seed + 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
