"""
data.py - Data loading and synthetic portfolio data generation for TROLL-Risk.

Generates realistic multi-asset portfolio data with regime-switching dynamics,
correlation shifts, and fat-tailed returns for training and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    """Configuration for synthetic portfolio data generation."""
    n_equities: int = 50
    n_bonds: int = 20
    n_commodities: int = 10
    n_fx: int = 5
    n_days: int = 3800  # ~15 years of trading days
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    risk_free_rate: float = 0.02
    crisis_probability: float = 0.02


class SyntheticPortfolioData:
    """Generates and manages synthetic multi-asset portfolio data.

    Produces daily returns with regime-switching behavior (normal/crisis),
    time-varying correlations, and fat-tailed distributions to simulate
    realistic financial market dynamics.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None, seed: int = 42):
        self.config = config or PortfolioConfig()
        self.rng = np.random.RandomState(seed)
        self.n_assets = (
            self.config.n_equities
            + self.config.n_bonds
            + self.config.n_commodities
            + self.config.n_fx
        )
        self.returns: Optional[np.ndarray] = None
        self.prices: Optional[np.ndarray] = None
        self.regimes: Optional[np.ndarray] = None
        self.asset_names: list = []

    def _generate_asset_names(self) -> list:
        """Create descriptive asset names for each instrument."""
        names = []
        names += [f"EQUITY_{i:02d}" for i in range(self.config.n_equities)]
        names += [f"BOND_{i:02d}" for i in range(self.config.n_bonds)]
        names += [f"COMM_{i:02d}" for i in range(self.config.n_commodities)]
        names += [f"FX_{i:02d}" for i in range(self.config.n_fx)]
        return names

    def _generate_regimes(self) -> np.ndarray:
        """Generate regime indicators (0=normal, 1=crisis) via Markov chain."""
        n = self.config.n_days
        regimes = np.zeros(n, dtype=int)
        # Transition probabilities
        p_stay_normal = 0.98
        p_stay_crisis = 0.85
        current = 0
        for t in range(n):
            regimes[t] = current
            if current == 0:
                if self.rng.random() > p_stay_normal:
                    current = 1
            else:
                if self.rng.random() > p_stay_crisis:
                    current = 0
        return regimes

    def _generate_correlation_matrix(self, regime: int) -> np.ndarray:
        """Generate a correlation matrix appropriate for the given regime.

        Crisis regimes have elevated average correlations (0.6-0.8) while
        normal regimes have moderate correlations (0.2-0.4).
        """
        n = self.n_assets
        if regime == 0:
            base_corr = 0.25
        else:
            base_corr = 0.70

        # Generate random positive-definite correlation matrix
        A = self.rng.randn(n, n) * 0.1
        A = A + A.T
        np.fill_diagonal(A, 0)
        corr = base_corr * np.ones((n, n)) * 0.5 + A
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        # Clip to valid correlation range
        corr = np.clip(corr, -0.3, 1.0)
        # Ensure positive definiteness via nearest PSD
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 0.01)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        return corr

    def generate(self) -> Dict[str, np.ndarray]:
        """Generate the full synthetic dataset.

        Returns:
            Dictionary with keys 'returns', 'prices', 'regimes', 'asset_names'.
        """
        self.asset_names = self._generate_asset_names()
        self.regimes = self._generate_regimes()

        # Asset-class-specific parameters
        n = self.config.n_days
        na = self.n_assets

        # Annualized volatility by asset class
        vols = np.concatenate([
            self.rng.uniform(0.15, 0.40, self.config.n_equities),
            self.rng.uniform(0.03, 0.10, self.config.n_bonds),
            self.rng.uniform(0.12, 0.30, self.config.n_commodities),
            self.rng.uniform(0.05, 0.12, self.config.n_fx),
        ])
        daily_vols = vols / np.sqrt(252)

        # Annualized drift by asset class
        drifts = np.concatenate([
            self.rng.uniform(0.04, 0.12, self.config.n_equities),
            self.rng.uniform(0.01, 0.04, self.config.n_bonds),
            self.rng.uniform(0.00, 0.06, self.config.n_commodities),
            self.rng.uniform(-0.01, 0.02, self.config.n_fx),
        ])
        daily_drifts = drifts / 252

        # Generate returns with regime-dependent correlations
        self.returns = np.zeros((n, na))
        cache_corr: Dict[int, np.ndarray] = {}

        for t in range(n):
            regime = self.regimes[t]
            if regime not in cache_corr:
                cache_corr[regime] = self._generate_correlation_matrix(regime)
            corr = cache_corr[regime]

            # Covariance = diag(vols) @ corr @ diag(vols)
            cov = np.outer(daily_vols, daily_vols) * corr

            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                cov += 1e-6 * np.eye(na)
                L = np.linalg.cholesky(cov)

            z = self.rng.randn(na)

            # Fat tails via Student-t mixing
            if regime == 1:
                # Crisis: heavier tails (df=4)
                scale = np.sqrt(self.rng.chisquare(4) / 4)
                z = z / max(scale, 0.1)

            self.returns[t] = daily_drifts + L @ z

        # Compute prices from returns
        self.prices = np.zeros((n + 1, na))
        self.prices[0] = 100.0  # Starting price
        for t in range(n):
            self.prices[t + 1] = self.prices[t] * (1 + self.returns[t])

        return {
            "returns": self.returns,
            "prices": self.prices[1:],
            "regimes": self.regimes,
            "asset_names": self.asset_names,
            "daily_vols": daily_vols,
            "daily_drifts": daily_drifts,
        }

    def get_splits(self) -> Tuple[Dict, Dict, Dict]:
        """Split data into train/validation/test sets.

        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries.
        """
        if self.returns is None:
            self.generate()

        n = self.config.n_days
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        splits = {}
        for name, start, end in [
            ("train", 0, train_end),
            ("val", train_end, val_end),
            ("test", val_end, n),
        ]:
            splits[name] = {
                "returns": self.returns[start:end],
                "prices": self.prices[start:end],
                "regimes": self.regimes[start:end],
            }
        return splits["train"], splits["val"], splits["test"]


class PortfolioStateBuilder:
    """Constructs state vectors from raw portfolio data for the RL agent.

    The state includes:
    - Recent log returns (lookback window)
    - Rolling volatility and correlation features
    - Current portfolio weights
    - Portfolio-level statistics (return, volatility, drawdown)
    """

    def __init__(
        self,
        lookback: int = 20,
        n_assets: int = 85,
    ):
        self.lookback = lookback
        self.n_assets = n_assets
        # State dimension: lookback*n_assets (returns) + n_assets (vols) + n_assets (weights) + 3 (stats)
        self.state_dim = lookback * n_assets + 2 * n_assets + 3

    def build_state(
        self,
        returns_history: np.ndarray,
        current_weights: np.ndarray,
        portfolio_return: float,
        portfolio_vol: float,
        current_drawdown: float,
    ) -> np.ndarray:
        """Build the state vector for the RL agent.

        Args:
            returns_history: Array of shape (lookback, n_assets) with recent returns.
            current_weights: Current portfolio allocation (n_assets,).
            portfolio_return: Current period portfolio return.
            portfolio_vol: Rolling realized volatility.
            current_drawdown: Current drawdown from peak (negative value).

        Returns:
            State vector of dimension state_dim.
        """
        # Flatten returns history
        flat_returns = returns_history[-self.lookback:].flatten()

        # Rolling volatility per asset
        if returns_history.shape[0] >= self.lookback:
            rolling_vol = np.std(returns_history[-self.lookback:], axis=0)
        else:
            rolling_vol = np.zeros(self.n_assets)

        # Portfolio statistics
        stats = np.array([portfolio_return, portfolio_vol, current_drawdown])

        state = np.concatenate([flat_returns, rolling_vol, current_weights, stats])
        return state.astype(np.float32)

    def build_initial_state(self, n_assets: int) -> np.ndarray:
        """Build a zero-initialized state for the first step."""
        returns_history = np.zeros((self.lookback, n_assets))
        weights = np.ones(n_assets) / n_assets  # Equal weight
        return self.build_state(returns_history, weights, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    config = PortfolioConfig(n_equities=10, n_bonds=5, n_commodities=3, n_fx=2)
    data_gen = SyntheticPortfolioData(config, seed=42)
    data = data_gen.generate()

    print(f"Generated {data['returns'].shape[0]} days x {data['returns'].shape[1]} assets")
    print(f"Return range: [{data['returns'].min():.4f}, {data['returns'].max():.4f}]")
    print(f"Mean daily return: {data['returns'].mean():.6f}")
    print(f"Crisis days: {data['regimes'].sum()} / {len(data['regimes'])}")

    train, val, test = data_gen.get_splits()
    print(f"Train: {train['returns'].shape}, Val: {val['returns'].shape}, Test: {test['returns'].shape}")
