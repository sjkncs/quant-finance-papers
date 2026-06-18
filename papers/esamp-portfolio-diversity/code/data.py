"""
data.py - Portfolio Data Generation for PortESamp
=================================================
Generates synthetic asset returns, market features, and portfolio
evaluation data for training and testing the PortESamp framework.

Key features:
- Synthetic multi-factor return generation with regime structure
- Market feature computation (volatility, momentum, correlation)
- Portfolio evaluation metrics (Sharpe, drawdown, entropy)
- Realistic covariance structure with sector clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AssetUniverseConfig:
    """Configuration for synthetic asset universe."""
    num_assets: int = 100
    num_sectors: int = 10
    num_factors: int = 5
    num_days: int = 2520  # ~10 years
    annual_vol_range: Tuple[float, float] = (0.15, 0.55)
    annual_return_range: Tuple[float, float] = (-0.05, 0.25)
    seed: int = 42


@dataclass
class MarketRegime:
    """Market regime parameters."""
    name: str
    drift_multiplier: float
    vol_multiplier: float
    correlation_shift: float
    duration_days: int


def get_default_regimes() -> List[MarketRegime]:
    """Get default market regime sequence for simulation."""
    return [
        MarketRegime("bull", 1.0, 0.8, -0.05, 756),       # 3 years
        MarketRegime("correction", 0.3, 1.3, 0.10, 126),   # 6 months
        MarketRegime("bear", -0.5, 1.5, 0.20, 378),        # 1.5 years
        MarketRegime("recovery", 1.2, 1.2, 0.05, 252),     # 1 year
        MarketRegime("bull_late", 0.8, 0.9, -0.03, 504),   # 2 years
        MarketRegime("stagnation", 0.1, 0.7, 0.00, 252),   # 1 year
        MarketRegime("rally", 1.5, 1.0, -0.08, 252),       # 1 year
    ]


class AssetUniverseGenerator:
    """Generates a synthetic asset universe with realistic properties."""

    def __init__(self, config: AssetUniverseConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.sector_assignments = self._assign_sectors()
        self.factor_loadings = self._generate_factor_loadings()
        self.idiosyncratic_vol = self._generate_idio_vol()
        self.expected_returns = self._generate_expected_returns()

    def _assign_sectors(self) -> np.ndarray:
        """Assign each asset to a sector."""
        return self.rng.randint(0, self.config.num_sectors, size=self.config.num_assets)

    def _generate_factor_loadings(self) -> np.ndarray:
        """Generate factor loadings with sector structure.

        Returns:
            Matrix of shape (num_assets, num_factors).
        """
        # Common factor structure: sector-level loadings + asset-level noise
        sector_loadings = self.rng.randn(
            self.config.num_sectors, self.config.num_factors
        ) * 0.5
        asset_loadings = np.zeros((self.config.num_assets, self.config.num_factors))
        for i in range(self.config.num_assets):
            sector = self.sector_assignments[i]
            asset_loadings[i] = sector_loadings[sector] + self.rng.randn(
                self.config.num_factors
            ) * 0.2
        return asset_loadings

    def _generate_idio_vol(self) -> np.ndarray:
        """Generate idiosyncratic volatilities."""
        low, high = self.config.annual_vol_range
        return self.rng.uniform(low, high, size=self.config.num_assets) / np.sqrt(252)

    def _generate_expected_returns(self) -> np.ndarray:
        """Generate daily expected returns."""
        low, high = self.config.annual_return_range
        annual = self.rng.uniform(low, high, size=self.config.num_assets)
        return annual / 252

    def generate_returns(
        self, regimes: Optional[List[MarketRegime]] = None
    ) -> pd.DataFrame:
        """Generate daily returns for all assets.

        Args:
            regimes: Optional list of market regimes. Uses defaults if None.

        Returns:
            DataFrame of shape (num_days, num_assets) with daily returns.
        """
        if regimes is None:
            regimes = get_default_regimes()

        num_days = self.config.num_days
        num_assets = self.config.num_assets
        returns = np.zeros((num_days, num_assets))

        # Generate factor returns
        factor_returns = self.rng.randn(num_days, self.config.num_factors) * 0.01

        day = 0
        for regime in regimes:
            for d in range(regime.duration_days):
                if day >= num_days:
                    break
                # Adjust factor returns by regime
                regime_factor = factor_returns[day] * regime.vol_multiplier
                regime_factor[0] += regime.drift_multiplier * 0.0005  # market factor drift

                # Asset returns = factor_model + idiosyncratic
                factor_component = self.factor_loadings @ regime_factor
                idio_component = self.rng.randn(num_assets) * self.idiosyncratic_vol

                # Add correlation shift (increases in stress)
                if regime.correlation_shift > 0:
                    common_shock = self.rng.randn() * regime.correlation_shift * 0.01
                    factor_component += common_shock

                returns[day] = self.expected_returns + factor_component + idio_component
                day += 1
            if day >= num_days:
                break

        # Fill remaining days with neutral regime
        while day < num_days:
            regime_factor = factor_returns[day] * 0.9
            factor_component = self.factor_loadings @ regime_factor
            idio_component = self.rng.randn(num_assets) * self.idiosyncratic_vol
            returns[day] = self.expected_returns + factor_component + idio_component
            day += 1

        columns = [f"asset_{i}" for i in range(num_assets)]
        return pd.DataFrame(returns, columns=columns)

    def compute_market_features(self, returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Compute market features for portfolio optimization.

        Args:
            returns: DataFrame of asset returns.
            window: Rolling window for feature computation.

        Returns:
            DataFrame of market features per asset per day.
        """
        features = {}

        # Rolling volatility
        features["volatility"] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling momentum (20-day return)
        features["momentum"] = returns.rolling(window).sum()

        # Rolling mean return
        features["mean_return"] = returns.rolling(window).mean() * 252

        # Rolling skewness
        features["skewness"] = returns.rolling(window).skew()

        # Volume proxy: absolute return magnitude
        features["abs_return"] = returns.rolling(window).mean().abs()

        return pd.concat(features, axis=1, names=["feature", "asset"])


def compute_portfolio_metrics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.04,
) -> Dict[str, float]:
    """Compute portfolio performance metrics.

    Args:
        returns: DataFrame of daily asset returns.
        weights: Portfolio weight vector.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Dictionary of performance metrics.
    """
    port_returns = returns.values @ weights
    annual_return = port_returns.mean() * 252
    annual_vol = port_returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / max(annual_vol, 1e-8)

    # Maximum drawdown
    cum_returns = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = float(drawdowns.min())

    # Calmar ratio
    calmar = annual_return / max(abs(max_dd), 1e-8)

    return {
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }


def compute_weight_entropy(weights: np.ndarray) -> float:
    """Compute normalized Shannon entropy of portfolio weights.

    Higher entropy = more diversified allocation.
    Maximum entropy = log(N) for uniform weights.

    Args:
        weights: Portfolio weight vector (non-negative, sums to 1).

    Returns:
        Normalized entropy in [0, 1].
    """
    # Filter out zero weights
    w = weights[weights > 1e-8]
    if len(w) == 0:
        return 0.0
    entropy = -np.sum(w * np.log(w))
    max_entropy = np.log(len(weights))
    return float(entropy / max(max_entropy, 1e-8))


def compute_strategy_diversity(portfolios: List[np.ndarray]) -> float:
    """Compute diversity of a set of portfolio strategies.

    Uses average pairwise entropy: the mean weight-space entropy
    across all portfolios in the set.

    Args:
        portfolios: List of weight vectors.

    Returns:
        Average weight-space entropy across portfolios.
    """
    if len(portfolios) == 0:
        return 0.0
    entropies = [compute_weight_entropy(w) for w in portfolios]
    return float(np.mean(entropies))


def generate_mean_variance_portfolio(
    returns: pd.DataFrame,
    target_return: Optional[float] = None,
    lambda_risk: float = 1.0,
) -> np.ndarray:
    """Generate a mean-variance optimal portfolio.

    Uses simple analytic solution with shrinkage covariance.

    Args:
        returns: DataFrame of asset returns.
        target_return: Optional target annual return.
        lambda_risk: Risk aversion parameter.

    Returns:
        Optimal weight vector.
    """
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    N = len(mu)

    # Ledoit-Wolf shrinkage toward identity
    shrinkage = 0.1
    cov_shrunk = (1 - shrinkage) * cov + shrinkage * np.eye(N) * np.trace(cov) / N

    # Analytic solution: w = (1/lambda) * Sigma^{-1} * mu
    try:
        cov_inv = np.linalg.inv(cov_shrunk)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_shrunk)

    weights = (1.0 / lambda_risk) * cov_inv @ mu

    # Project onto simplex (non-negative, sum to 1)
    weights = np.maximum(weights, 0)
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(N) / N

    return weights


def generate_risk_parity_portfolio(returns: pd.DataFrame) -> np.ndarray:
    """Generate a risk parity portfolio.

    Equalizes risk contribution across all assets using iterative
    proportional scaling.

    Args:
        returns: DataFrame of asset returns.

    Returns:
        Risk parity weight vector.
    """
    cov = returns.cov().values * 252
    N = cov.shape[0]

    # Inverse-volatility as starting point
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / (vols + 1e-8)
    weights = inv_vol / inv_vol.sum()

    # Iterative risk parity adjustment
    for _ in range(50):
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk / (port_vol + 1e-8)
        target_contrib = port_vol / N

        # Adjust weights toward equal risk contribution
        adjustment = target_contrib / (risk_contrib + 1e-8)
        weights = weights * adjustment
        weights = np.maximum(weights, 1e-6)
        weights = weights / weights.sum()

    return weights


if __name__ == "__main__":
    config = AssetUniverseConfig(num_assets=100, num_days=504, seed=42)
    gen = AssetUniverseGenerator(config)
    returns = gen.generate_returns()
    print(f"Generated returns: {returns.shape}")
    print(f"Mean daily return: {returns.mean().mean():.6f}")
    print(f"Mean daily vol: {returns.std().mean():.4f}")

    # Generate portfolio types
    mv_weights = generate_mean_variance_portfolio(returns)
    rp_weights = generate_risk_parity_portfolio(returns)

    mv_metrics = compute_portfolio_metrics(returns, mv_weights)
    rp_metrics = compute_portfolio_metrics(returns, rp_weights)

    print(f"\nMean-Variance: Sharpe={mv_metrics['sharpe']:.3f}, "
          f"MaxDD={mv_metrics['max_drawdown']:.1%}")
    print(f"Risk Parity:   Sharpe={rp_metrics['sharpe']:.3f}, "
          f"MaxDD={rp_metrics['max_drawdown']:.1%}")

    print(f"\nMV entropy: {compute_weight_entropy(mv_weights):.3f}")
    print(f"RP entropy: {compute_weight_entropy(rp_weights):.3f}")
