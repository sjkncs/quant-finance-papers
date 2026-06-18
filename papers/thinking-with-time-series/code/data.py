"""
data.py — Data loading, preprocessing, and synthetic data generation for TTS.

Generates synthetic financial time-series data and MarketThinkBench-style
scenarios so the full pipeline can run without external datasets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


@dataclass
class MarketScenario:
    """A single MarketThinkBench scenario."""

    query: str
    assets: List[str]
    horizon_days: int
    regime_label: int  # 0=low-vol/bull, 1=high-vol/bear, 2=crisis
    ground_truth: int  # 0=negative, 1=neutral, 2=positive
    difficulty: int  # 1-5
    category: str  # regime / causality / tail_risk / rebalancing


@dataclass
class MarketState:
    """Observable market state at time t."""

    prices: np.ndarray  # (n_assets, lookback) log-return series
    vix: float
    yields: np.ndarray  # yield curve (n_maturities,)
    regime: int


class SyntheticMarketDataGenerator:
    """Generate synthetic multi-asset financial time-series using
    geometric Brownian motion with regime switching and stochastic volatility."""

    ASSET_UNIVERSE = [
        "SPY", "QQQ", "IWM", "TLT", "GLD",
        "USO", "FXI", "EWJ", "EWG", "EEM",
        "XLF", "XLE", "XLK", "XLV", "XLU",
    ]

    def __init__(
        self,
        n_assets: int = 15,
        n_days: int = 5000,
        seed: int = 42,
    ):
        self.n_assets = min(n_assets, len(self.ASSET_UNIVERSE))
        self.n_days = n_days
        self.rng = np.random.default_rng(seed)
        self.asset_names = self.ASSET_UNIVERSE[: self.n_assets]

    def generate_full_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (log_returns, volatility, regime_sequence).

        log_returns:  (n_days, n_assets)
        volatility:   (n_days, n_assets)  instantaneous vol
        regime_seq:   (n_days,)           regime index 0/1/2
        """
        # Regime parameters: (mu, sigma, transition_probs)
        regime_params = [
            {"mu": 0.0004, "sigma": 0.012, "stay": 0.97},   # bull
            {"mu": -0.0002, "sigma": 0.025, "stay": 0.92},  # bear
            {"mu": -0.001, "sigma": 0.055, "stay": 0.85},   # crisis
        ]

        # Generate regime sequence
        regime_seq = np.zeros(self.n_days, dtype=int)
        for t in range(1, self.n_days):
            r = regime_seq[t - 1]
            if self.rng.random() > regime_params[r]["stay"]:
                # Transition to a different regime
                others = [x for x in range(3) if x != r]
                regime_seq[t] = self.rng.choice(others)
            else:
                regime_seq[t] = r

        # Cross-asset correlation matrix (random positive-definite)
        A = self.rng.standard_normal((self.n_assets, self.n_assets)) * 0.3
        corr = A @ A.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        L = np.linalg.cholesky(corr)

        # Generate correlated returns with regime-dependent params
        log_returns = np.zeros((self.n_days, self.n_assets))
        volatility = np.zeros((self.n_days, self.n_assets))
        for t in range(self.n_days):
            r = regime_seq[t]
            mu = regime_params[r]["mu"]
            sigma = regime_params[r]["sigma"]
            z = self.rng.standard_normal(self.n_assets)
            corr_z = L @ z
            log_returns[t] = mu + sigma * corr_z
            volatility[t] = sigma * np.ones(self.n_assets)

        # Add GARCH-like vol clustering
        for a in range(self.n_assets):
            for t in range(1, self.n_days):
                volatility[t, a] = (
                    0.9 * volatility[t - 1, a]
                    + 0.1 * abs(log_returns[t - 1, a])
                    + 0.001
                )
            log_returns[:, a] = (
                log_returns[:, a] * volatility[:, a] / volatility[:, a].mean()
            )

        return log_returns, volatility, regime_seq

    def generate_vix_proxy(self, log_returns: np.ndarray) -> np.ndarray:
        """Generate a VIX-like volatility index from market returns."""
        market_return = log_returns[:, :3].mean(axis=1)
        rolling_vol = np.zeros(len(market_return))
        window = 20
        for t in range(window, len(market_return)):
            rolling_vol[t] = np.std(market_return[t - window : t]) * np.sqrt(252)
        # Scale to VIX-like range (10-80)
        rolling_vol = rolling_vol * 100
        rolling_vol = np.clip(rolling_vol, 10, 80)
        return rolling_vol


class MarketThinkBench:
    """Generate synthetic MarketThinkBench scenarios with ground truth."""

    QUERY_TEMPLATES = {
        "regime": [
            "What market regime will {asset} be in over the next {horizon} days?",
            "Is the current volatility regime for {asset} likely to persist?",
            "Will market conditions shift from {current_regime} in the next {horizon} days?",
        ],
        "causality": [
            "If {asset1} drops 5% tomorrow, how will {asset2} respond over {horizon} days?",
            "What is the cross-asset impact on {asset2} given a shock to {asset1}?",
            "How does a rate hike affect the correlation between {asset1} and {asset2}?",
        ],
        "tail_risk": [
            "What is the probability that {asset} drops more than 10% in {horizon} days?",
            "Estimate the tail risk for {asset} under current market conditions.",
            "What is the expected maximum drawdown of {asset} over {horizon} days?",
        ],
        "rebalancing": [
            "Should the portfolio increase allocation to {asset} over {horizon} days?",
            "What is the optimal hedge ratio for {asset} given current volatility?",
            "How should exposure to {asset} change if VIX exceeds 30?",
        ],
    }

    def __init__(self, n_scenarios: int = 4200, seed: int = 42):
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(seed)
        self.scenarios = self._generate_scenarios()

    def _generate_scenarios(self) -> List[MarketScenario]:
        scenarios = []
        categories = ["regime", "causality", "tail_risk", "rebalancing"]
        per_cat = self.n_scenarios // len(categories)
        assets = SyntheticMarketDataGenerator.ASSET_UNIVERSE

        for cat in categories:
            for i in range(per_cat):
                asset = self.rng.choice(assets)
                horizon = int(self.rng.choice([5, 10, 20, 60]))
                regime = int(self.rng.choice([0, 1, 2]))
                difficulty = int(self.rng.integers(1, 6))

                if cat == "causality":
                    asset2 = self.rng.choice([a for a in assets if a != asset])
                    tmpl = self.rng.choice(self.QUERY_TEMPLATES[cat])
                    query = tmpl.format(
                        asset1=asset, asset2=asset2, horizon=horizon
                    )
                else:
                    tmpl = self.rng.choice(self.QUERY_TEMPLATES[cat])
                    query = tmpl.format(
                        asset=asset,
                        horizon=horizon,
                        current_regime=["bull", "bear", "crisis"][regime],
                    )

                # Ground truth: simple heuristic based on regime and category
                gt = self._compute_ground_truth(cat, regime, difficulty)

                scenarios.append(
                    MarketScenario(
                        query=query,
                        assets=[asset],
                        horizon_days=horizon,
                        regime_label=regime,
                        ground_truth=gt,
                        difficulty=difficulty,
                        category=cat,
                    )
                )
        return scenarios

    def _compute_ground_truth(self, cat: str, regime: int, difficulty: int) -> int:
        """Heuristic ground truth for synthetic scenarios."""
        if cat == "regime":
            # Regime persistence
            if regime == 0:
                return 0  # stays bull
            elif regime == 1:
                return 1  # neutral/transition
            else:
                return 1  # crisis transitions
        elif cat == "causality":
            return 0 if regime >= 1 else 2  # negative in bad regimes
        elif cat == "tail_risk":
            return 0 if regime == 2 else (1 if regime == 1 else 2)
        else:  # rebalancing
            return 2 if regime == 0 else 0

    def get_split(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[List[MarketScenario], List[MarketScenario], List[MarketScenario]]:
        n = len(self.scenarios)
        idx = self.rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = [self.scenarios[i] for i in idx[:n_train]]
        val = [self.scenarios[i] for i in idx[n_train : n_train + n_val]]
        test = [self.scenarios[i] for i in idx[n_train + n_val :]]
        return train, val, test


class TTSDataset(Dataset):
    """PyTorch dataset wrapping synthetic market data for TTS training."""

    def __init__(
        self,
        log_returns: np.ndarray,
        volatility: np.ndarray,
        regime_seq: np.ndarray,
        vix: np.ndarray,
        scenarios: List[MarketScenario],
        lookback: int = 60,
        horizon: int = 60,
        n_assets: int = 15,
    ):
        self.log_returns = torch.tensor(log_returns, dtype=torch.float32)
        self.volatility = torch.tensor(volatility, dtype=torch.float32)
        self.regime_seq = torch.tensor(regime_seq, dtype=torch.long)
        self.vix = torch.tensor(vix, dtype=torch.float32)
        self.scenarios = scenarios
        self.lookback = lookback
        self.horizon = horizon
        self.n_assets = n_assets

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> dict:
        scenario = self.scenarios[idx]
        # Pick a random start point with enough future data
        max_start = len(self.log_returns) - self.lookback - self.horizon
        start = np.random.randint(0, max_start)

        market_state = self.log_returns[start : start + self.lookback]  # (lookback, n_assets)
        future = self.log_returns[
            start + self.lookback : start + self.lookback + self.horizon
        ]  # (horizon, n_assets)
        vol_state = self.volatility[start : start + self.lookback]
        vix_val = self.vix[start + self.lookback]
        regime = self.regime_seq[start + self.lookback]

        # Simple query encoding: hash-based embedding
        query_emb = self._encode_query(scenario.query)

        return {
            "market_state": market_state,
            "future": future,
            "vol_state": vol_state,
            "vix": vix_val,
            "regime": regime,
            "query_emb": query_emb,
            "ground_truth": scenario.ground_truth,
            "category": scenario.category,
            "horizon_days": scenario.horizon_days,
        }

    def _encode_query(self, query: str, dim: int = 128) -> torch.Tensor:
        """Simple deterministic query encoding using character hashing."""
        emb = torch.zeros(dim)
        for i, ch in enumerate(query):
            idx = (ord(ch) * (i + 1)) % dim
            emb[idx] += 1.0
        # Normalize
        norm = emb.norm()
        if norm > 0:
            emb = emb / norm
        return emb


def build_dataloaders(
    n_scenarios: int = 420,
    batch_size: int = 32,
    seed: int = 42,
):
    """Build train/val/test DataLoaders with synthetic data."""
    data_gen = SyntheticMarketDataGenerator(seed=seed)
    log_returns, volatility, regime_seq = data_gen.generate_full_history()
    vix = data_gen.generate_vix_proxy(log_returns)

    bench = MarketThinkBench(n_scenarios=n_scenarios, seed=seed)
    train_scenarios, val_scenarios, test_scenarios = bench.get_split()

    train_ds = TTSDataset(log_returns, volatility, regime_seq, vix, train_scenarios)
    val_ds = TTSDataset(log_returns, volatility, regime_seq, vix, val_scenarios)
    test_ds = TTSDataset(log_returns, volatility, regime_seq, vix, test_scenarios)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, data_gen.asset_names
