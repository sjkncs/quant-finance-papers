"""
data.py — Multi-resolution financial time-series data generation for MMRF.

Generates synthetic multi-resolution market data (tick, minute, hourly,
daily, weekly) with realistic autocorrelation, volatility clustering,
and cross-asset correlation structure.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class ResolutionConfig:
    """Configuration for a single resolution level."""

    name: str
    n_features: int
    tokens_per_asset: int
    aggregation_factor: int  # how many finer-resolution periods per token


RESOLUTION_CONFIGS = [
    ResolutionConfig("tick", 5, 500, 1),
    ResolutionConfig("minute", 7, 500, 5),
    ResolutionConfig("hourly", 6, 200, 12),
    ResolutionConfig("daily", 8, 120, 24),
    ResolutionConfig("weekly", 5, 60, 5),
]


class MultiResolutionDataGenerator:
    """Generate synthetic multi-resolution financial data with
    realistic autocorrelation decay, volatility clustering, and
    cross-asset correlation."""

    def __init__(
        self,
        n_assets: int = 10,
        n_days: int = 2000,
        seed: int = 42,
    ):
        self.n_assets = n_assets
        self.n_days = n_days
        self.rng = np.random.default_rng(seed)

    def generate_daily_returns(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate daily log-returns with GARCH-like vol clustering.

        Returns:
            returns: (n_days, n_assets)
            volatility: (n_days, n_assets)
            regime: (n_days,) in {0, 1, 2}
        """
        # Cross-asset correlation
        A = self.rng.standard_normal((self.n_assets, self.n_assets)) * 0.3
        corr = A @ A.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        L = np.linalg.cholesky(corr)

        # Regime process
        regime = np.zeros(self.n_days, dtype=int)
        regime_params = [
            (0.0003, 0.010, 0.97),   # bull
            (-0.0001, 0.022, 0.92),  # bear
            (-0.0008, 0.050, 0.85),  # crisis
        ]
        for t in range(1, self.n_days):
            r = regime[t - 1]
            if self.rng.random() > regime_params[r][2]:
                others = [x for x in range(3) if x != r]
                regime[t] = self.rng.choice(others)
            else:
                regime[t] = r

        # Generate returns with GARCH
        returns = np.zeros((self.n_days, self.n_assets))
        volatility = np.zeros((self.n_days, self.n_assets))

        for a in range(self.n_assets):
            volatility[0, a] = 0.015
            for t in range(self.n_days):
                r = regime[t]
                mu = regime_params[r][0]
                omega = 1e-6
                alpha = 0.08
                beta = 0.88

                z = self.rng.standard_normal()
                if t > 0:
                    volatility[t, a] = np.sqrt(
                        omega
                        + alpha * returns[t - 1, a] ** 2
                        + beta * volatility[t - 1, a] ** 2
                    )
                    volatility[t, a] = max(volatility[t, a], 0.001)
                returns[t, a] = mu + volatility[t, a] * z

        # Add cross-asset correlation
        for t in range(self.n_days):
            z = self.rng.standard_normal(self.n_assets)
            corr_z = L @ z
            returns[t] = (
                returns[t] * 0.5
                + volatility[t] * corr_z * 0.5
            )

        return returns, volatility, regime

    def aggregate_to_resolution(
        self,
        daily_returns: np.ndarray,
        daily_vol: np.ndarray,
        resolution: ResolutionConfig,
    ) -> np.ndarray:
        """Aggregate daily data to a given resolution level.

        Returns features array of shape (n_periods, n_assets, n_features).
        """
        n_days, n_assets = daily_returns.shape
        agg = resolution.aggregation_factor
        # For simplicity, aggregate from daily
        if agg <= 1:
            agg = 1
        n_periods = n_days // agg

        features = np.zeros((n_periods, n_assets, resolution.n_features))
        for t in range(n_periods):
            start = t * agg
            end = min(start + agg, n_days)
            chunk = daily_returns[start:end]  # (agg, n_assets)
            vol_chunk = daily_vol[start:end]

            for a in range(n_assets):
                features[t, a, 0] = chunk[:, a].sum()  # cumulative return
                features[t, a, 1] = chunk[:, a].std() if len(chunk) > 1 else 0  # vol
                features[t, a, 2] = vol_chunk[:, a].mean()  # avg vol
                if resolution.n_features > 3:
                    features[t, a, 3] = abs(chunk[:, a]).max()  # max abs return
                if resolution.n_features > 4:
                    features[t, a, 4] = np.sign(chunk[:, a].sum())  # direction
                if resolution.n_features > 5:
                    skew = 0.0
                    if len(chunk) > 2 and chunk[:, a].std() > 0:
                        from scipy import stats
                        skew = stats.skew(chunk[:, a])
                    features[t, a, 5] = skew
                if resolution.n_features > 6:
                    features[t, a, 6] = self.rng.uniform(0.5, 2.0)  # volume proxy

        return features


def compute_compressed_summary(
    features: np.ndarray,  # (n_periods, n_assets, n_features)
    n_top_eigenvalues: int = 5,
    n_autocorr_lags: int = 10,
) -> np.ndarray:
    """Compute compressed statistical summary for a resolution level.

    Returns a 1D vector of dimension d_c=30.
    """
    n_periods, n_assets, n_features = features.shape
    returns = features[:, :, 0]  # (n_periods, n_assets)

    summary_parts = []

    # 1. Rolling statistics (8 dims)
    for window in [min(20, n_periods), min(60, n_periods), min(n_periods, n_periods)]:
        if window > 0:
            chunk = returns[-window:]
            summary_parts.extend([
                np.mean(chunk),
                np.std(chunk) + 1e-8,
            ])
        else:
            summary_parts.extend([0.0, 1e-8])
    # skewness and kurtosis of last 60 periods
    last_n = min(60, n_periods)
    if last_n > 2:
        from scipy import stats as sp_stats
        flat = returns[-last_n:].flatten()
        summary_parts.extend([sp_stats.skew(flat), sp_stats.kurtosis(flat)])
    else:
        summary_parts.extend([0.0, 0.0])

    # 2. Autocorrelation structure (10 dims)
    market_returns = returns.mean(axis=1)  # (n_periods,)
    for lag in range(1, n_autocorr_lags + 1):
        if len(market_returns) > lag:
            ac = np.corrcoef(market_returns[:-lag], market_returns[lag:])[0, 1]
            summary_parts.append(ac if not np.isnan(ac) else 0.0)
        else:
            summary_parts.append(0.0)

    # 3. Cross-asset correlation eigenvalues (5 dims)
    if n_assets >= n_top_eigenvalues and n_periods > 1:
        corr_mat = np.corrcoef(returns.T)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        eigenvalues = np.sort(np.linalg.eigvalsh(corr_mat))[::-1][:n_top_eigenvalues]
        summary_parts.extend(eigenvalues.tolist())
    else:
        summary_parts.extend([1.0] * n_top_eigenvalues)

    # 4. Volatility regime features (4 dims)
    if n_periods > 60:
        current_vol = returns[-20:].std() * np.sqrt(252)
        hist_vol = returns[-60:].std() * np.sqrt(252)
        vol_ratio = current_vol / (hist_vol + 1e-8)
    else:
        vol_ratio = 1.0
        current_vol = returns.std() * np.sqrt(252) if n_periods > 0 else 0.0
    summary_parts.extend([
        vol_ratio,
        min(current_vol / 0.8, 1.0),  # normalized VIX proxy
        1.0 if current_vol > 0.3 else 0.0,  # crisis indicator
        np.clip(vol_ratio - 1.0, -1, 1),  # vol change
    ])

    # 5. Trend features (3 dims)
    if n_periods >= 50:
        ma50 = returns[-50:].mean()
        ma200 = returns[-min(200, n_periods):].mean()
        std200 = returns[-min(200, n_periods):].std() + 1e-8
        summary_parts.extend([
            (ma50 - ma200) / std200,  # distance from 200-MA z-score
            ma50,  # slope proxy
            1.0 if ma50 > ma200 else (-1.0 if ma50 < ma200 else 0.0),  # trend
        ])
    else:
        summary_parts.extend([0.0, 0.0, 0.0])

    return np.array(summary_parts[:30], dtype=np.float32)


class MultiResolutionDataset(Dataset):
    """PyTorch dataset for MMRF with multi-resolution token sequences."""

    def __init__(
        self,
        resolution_data: Dict[str, np.ndarray],
        n_assets: int = 10,
        token_dim: int = 64,
        window_width: int = 2,
    ):
        self.resolution_data = resolution_data  # {name: (n_periods, n_assets, n_feat)}
        self.resolution_names = list(resolution_data.keys())
        self.n_assets = n_assets
        self.token_dim = token_dim
        self.window_width = window_width

        # Determine the number of samples (based on coarsest resolution)
        coarsest = self.resolution_names[-1]
        self.n_samples = max(1, resolution_data[coarsest].shape[0] - 10)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {}

        # For each resolution, extract a window of tokens
        for res_name in self.resolution_names:
            data = self.resolution_data[res_name]
            n_periods = data.shape[0]

            # Map sample index to period index
            period_idx = min(
                int(idx * n_periods / self.n_samples),
                n_periods - 1,
            )

            # Extract token window (last 30 periods for this resolution)
            lookback = min(30, period_idx + 1)
            start = period_idx - lookback + 1
            tokens = data[start : period_idx + 1]  # (lookback, n_assets, n_feat)

            # Flatten assets and features into token dimension
            tokens_flat = tokens.reshape(lookback, -1)  # (lookback, n_assets * n_feat)

            # Pad or truncate to fixed length
            max_len = 30
            if tokens_flat.shape[0] < max_len:
                pad = np.zeros((max_len - tokens_flat.shape[0], tokens_flat.shape[1]))
                tokens_flat = np.concatenate([pad, tokens_flat], axis=0)

            sample[f"tokens_{res_name}"] = torch.tensor(
                tokens_flat[:max_len], dtype=torch.float32
            )

        # Compute compressed summaries for resolutions outside the window
        summaries = []
        for i, res_name in enumerate(self.resolution_names):
            data = self.resolution_data[res_name]
            n_periods = data.shape[0]
            period_idx = min(
                int(idx * n_periods / self.n_samples),
                n_periods - 1,
            )
            # Use data up to the current point
            history = data[: period_idx + 1]
            summary = compute_compressed_summary(history)
            summaries.append(summary)

        sample["compressed_summaries"] = torch.tensor(
            np.stack(summaries), dtype=torch.float32
        )  # (n_resolutions, 30)

        # Target: next-period return at each resolution
        for res_name in self.resolution_names:
            data = self.resolution_data[res_name]
            n_periods = data.shape[0]
            period_idx = min(
                int(idx * n_periods / self.n_samples) + 1,
                n_periods - 1,
            )
            target_return = data[period_idx, :, 0].mean()  # avg across assets
            sample[f"target_{res_name}"] = torch.tensor(
                target_return, dtype=torch.float32
            )

        return sample


def build_mmrf_dataloaders(
    n_assets: int = 10,
    n_days: int = 2000,
    batch_size: int = 32,
    window_width: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train/val/test DataLoaders for MMRF."""
    gen = MultiResolutionDataGenerator(n_assets=n_assets, n_days=n_days, seed=seed)
    daily_returns, daily_vol, regime = gen.generate_daily_returns()

    # Generate data for each resolution
    resolution_data = {}
    for rc in RESOLUTION_CONFIGS:
        features = gen.aggregate_to_resolution(daily_returns, daily_vol, rc)
        resolution_data[rc.name] = features

    # Split: 70/15/15 by time
    # Use the daily data to determine split points
    n_total = daily_returns.shape[0]
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    # Create split datasets
    train_data = {}
    val_data = {}
    test_data = {}
    for name, features in resolution_data.items():
        n_p = features.shape[0]
        t1 = int(n_p * 0.70)
        t2 = int(n_p * 0.85)
        train_data[name] = features[:t1]
        val_data[name] = features[t1:t2]
        test_data[name] = features[t2:]

    train_ds = MultiResolutionDataset(train_data, n_assets, window_width=window_width)
    val_ds = MultiResolutionDataset(val_data, n_assets, window_width=window_width)
    test_ds = MultiResolutionDataset(test_data, n_assets, window_width=window_width)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    res_names = [rc.name for rc in RESOLUTION_CONFIGS]
    return train_loader, val_loader, test_loader, res_names
