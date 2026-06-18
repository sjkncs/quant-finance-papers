"""
data.py - QuantBench Market Simulation Data Generation
=======================================================
Generates synthetic multi-asset market data with regime transitions
for long-horizon trading agent evaluation.

Key features:
- Multi-regime simulation (Bull, Bear, Crash, Recovery, Stagnation)
- Semi-Markov regime transitions with configurable probabilities
- Multi-asset universe (equities, bonds, commodities, currencies, crypto)
- Realistic correlation structure that varies by regime
- Historical and synthetic scenario generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class Regime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    CRASH = "crash"
    RECOVERY = "recovery"
    STAGNATION = "stagnation"


@dataclass
class RegimeParams:
    """Parameters for a single market regime."""
    regime: Regime
    drift_annual: float        # annualized expected return for equities
    vol_annual: float          # annualized volatility for equities
    bond_drift: float          # annualized bond return
    commodity_drift: float     # annualized commodity return
    base_correlation: float    # average cross-asset correlation
    spread_multiplier: float   # bid-ask spread multiplier
    duration_range: Tuple[int, int]  # min/max days


REGIME_PARAMS = {
    Regime.BULL: RegimeParams(
        Regime.BULL, drift_annual=0.10, vol_annual=0.15,
        bond_drift=0.03, commodity_drift=0.05,
        base_correlation=0.30, spread_multiplier=1.0,
        duration_range=(504, 1008),  # 2-4 years
    ),
    Regime.BEAR: RegimeParams(
        Regime.BEAR, drift_annual=-0.15, vol_annual=0.25,
        bond_drift=0.05, commodity_drift=-0.10,
        base_correlation=0.50, spread_multiplier=1.5,
        duration_range=(126, 504),  # 0.5-2 years
    ),
    Regime.CRASH: RegimeParams(
        Regime.CRASH, drift_annual=-0.60, vol_annual=0.45,
        bond_drift=0.08, commodity_drift=-0.30,
        base_correlation=0.80, spread_multiplier=3.0,
        duration_range=(10, 63),  # 2 weeks - 3 months
    ),
    Regime.RECOVERY: RegimeParams(
        Regime.RECOVERY, drift_annual=0.20, vol_annual=0.22,
        bond_drift=0.02, commodity_drift=0.12,
        base_correlation=0.35, spread_multiplier=1.2,
        duration_range=(126, 504),
    ),
    Regime.STAGNATION: RegimeParams(
        Regime.STAGNATION, drift_annual=0.01, vol_annual=0.12,
        bond_drift=0.02, commodity_drift=0.00,
        base_correlation=0.25, spread_multiplier=1.0,
        duration_range=(126, 504),
    ),
}


# Default regime transition probability matrix
# Rows: current regime, Columns: next regime
TRANSITION_MATRIX = {
    Regime.BULL:       {Regime.BULL: 0.70, Regime.BEAR: 0.12, Regime.CRASH: 0.05,
                        Regime.RECOVERY: 0.03, Regime.STAGNATION: 0.10},
    Regime.BEAR:       {Regime.BULL: 0.10, Regime.BEAR: 0.45, Regime.CRASH: 0.25,
                        Regime.RECOVERY: 0.15, Regime.STAGNATION: 0.05},
    Regime.CRASH:      {Regime.BULL: 0.05, Regime.BEAR: 0.15, Regime.CRASH: 0.10,
                        Regime.RECOVERY: 0.60, Regime.STAGNATION: 0.10},
    Regime.RECOVERY:   {Regime.BULL: 0.50, Regime.BEAR: 0.10, Regime.CRASH: 0.05,
                        Regime.RECOVERY: 0.20, Regime.STAGNATION: 0.15},
    Regime.STAGNATION: {Regime.BULL: 0.35, Regime.BEAR: 0.20, Regime.CRASH: 0.08,
                        Regime.RECOVERY: 0.17, Regime.STAGNATION: 0.20},
}


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""
    name: str
    num_days: int = 2520
    regime_sequence: Optional[List[Regime]] = None
    custom_transition_matrix: Optional[Dict] = None
    seed: int = 42


@dataclass
class AssetConfig:
    """Asset universe configuration."""
    num_equities: int = 50
    num_bonds: int = 10
    num_commodities: int = 5
    num_currencies: int = 3
    num_crypto: int = 1
    initial_prices: Optional[Dict[str, float]] = None

    @property
    def total_assets(self) -> int:
        return (self.num_equities + self.num_bonds + self.num_commodities
                + self.num_currencies + self.num_crypto)


class RegimeSimulator:
    """Simulates market regime transitions using a semi-Markov process."""

    def __init__(
        self,
        transition_matrix: Optional[Dict] = None,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.transition_matrix = transition_matrix or TRANSITION_MATRIX

    def generate_regime_sequence(
        self,
        num_days: int,
        initial_regime: Regime = Regime.BULL,
    ) -> Tuple[List[Regime], List[int]]:
        """Generate a sequence of regimes and their durations.

        Args:
            num_days: Total number of trading days.
            initial_regime: Starting regime.

        Returns:
            Tuple of (regime_list, transition_days).
        """
        regimes = []
        transition_days = []
        current_regime = initial_regime
        day = 0

        while day < num_days:
            params = REGIME_PARAMS[current_regime]
            duration = self.rng.randint(*params.duration_range)
            duration = min(duration, num_days - day)

            regimes.append((current_regime, day, day + duration))
            transition_days.append(day + duration)
            day += duration

            # Sample next regime
            probs = self.transition_matrix[current_regime]
            regime_names = list(probs.keys())
            prob_values = [probs[r] for r in regime_names]
            current_regime = regime_names[
                self.rng.choice(len(regime_names), p=prob_values)
            ]

        return regimes, transition_days[:-1]  # exclude last (past end)

    def get_regime_at_day(self, regimes: List, day: int) -> Regime:
        """Get the regime at a specific day."""
        for regime, start, end in regimes:
            if start <= day < end:
                return regime
        return regimes[-1][0]


class MarketSimulator:
    """Simulates multi-asset market dynamics with regime-dependent parameters."""

    def __init__(
        self,
        asset_config: AssetConfig = AssetConfig(),
        seed: int = 42,
    ):
        self.config = asset_config
        self.rng = np.random.RandomState(seed)
        self.N = asset_config.total_assets
        self._initialize_prices()

    def _initialize_prices(self):
        """Set initial prices for all assets."""
        prices = {}
        for i in range(self.config.num_equities):
            prices[f"eq_{i}"] = self.rng.uniform(30, 500)
        for i in range(self.config.num_bonds):
            prices[f"bond_{i}"] = 100.0
        for i in range(self.config.num_commodities):
            prices[f"commodity_{i}"] = self.rng.uniform(20, 200)
        for i in range(self.config.num_currencies):
            prices[f"fx_{i}"] = self.rng.uniform(0.8, 1.5)
        for i in range(self.config.num_crypto):
            prices[f"crypto_{i}"] = self.rng.uniform(10000, 50000)
        self.prices = prices

    def generate_correlation_matrix(
        self, regime: Regime
    ) -> np.ndarray:
        """Generate a regime-dependent correlation matrix.

        Args:
            regime: Current market regime.

        Returns:
            Correlation matrix of shape (N, N).
        """
        params = REGIME_PARAMS[regime]
        base_corr = params.base_correlation

        # Build block-structured correlation matrix
        corr = np.full((self.N, self.N), base_corr)

        # Within-equity correlations are higher
        eq_end = self.config.num_equities
        corr[:eq_end, :eq_end] = base_corr + 0.15

        # Bonds are negatively correlated with equities in crashes
        bond_start = eq_end
        bond_end = bond_start + self.config.num_bonds
        if regime in [Regime.CRASH, Regime.BEAR]:
            corr[:eq_end, bond_start:bond_end] = -0.3
            corr[bond_start:bond_end, :eq_end] = -0.3

        # Set diagonal to 1
        np.fill_diagonal(corr, 1.0)

        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, 0.01)
        corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)

        return corr

    def simulate_scenario(
        self,
        scenario: ScenarioConfig,
    ) -> pd.DataFrame:
        """Simulate a complete scenario with regime transitions.

        Args:
            scenario: Scenario configuration.

        Returns:
            DataFrame with columns for each asset's daily returns plus
            'regime' and 'day' columns.
        """
        regime_sim = RegimeSimulator(
            transition_matrix=scenario.custom_transition_matrix,
            seed=scenario.seed,
        )

        if scenario.regime_sequence:
            # Use predefined regime sequence
            regimes = []
            day = 0
            regime_idx = 0
            days_per_regime = scenario.num_days // len(scenario.regime_sequence)
            for r in scenario.regime_sequence:
                start = day
                end = min(day + days_per_regime, scenario.num_days)
                regimes.append((r, start, end))
                day = end
                regime_idx += 1
            transition_days = [r[1] for r in regimes[1:]]
        else:
            regimes, transition_days = regime_sim.generate_regime_sequence(
                scenario.num_days
            )

        self._initialize_prices()
        self.rng = np.random.RandomState(scenario.seed)

        # Generate daily returns
        all_data = []
        for day in range(scenario.num_days):
            regime = regime_sim.get_regime_at_day(regimes, day)
            params = REGIME_PARAMS[regime]

            # Regime-dependent parameters
            daily_vol = params.vol_annual / np.sqrt(252)
            daily_drift = params.drift_annual / 252

            # Generate correlated returns
            corr = self.generate_correlation_matrix(regime)
            z = self.rng.multivariate_normal(np.zeros(self.N), corr)

            # Scale by asset-type-specific volatility
            vol_scale = np.ones(self.N)
            eq_end = self.config.num_equities
            bond_end = eq_end + self.config.num_bonds
            comm_end = bond_end + self.config.num_commodities
            fx_end = comm_end + self.config.num_currencies

            # Equities: base vol with individual variation
            vol_scale[:eq_end] = daily_vol * self.rng.uniform(0.7, 1.5, eq_end)
            # Bonds: lower vol
            vol_scale[eq_end:bond_end] = daily_vol * 0.3
            # Commodities: higher vol
            vol_scale[bond_end:comm_end] = daily_vol * 1.3
            # FX: moderate vol
            vol_scale[comm_end:fx_end] = daily_vol * 0.5
            # Crypto: very high vol
            vol_scale[fx_end:] = daily_vol * 3.0

            # Compute returns
            drift_vec = np.zeros(self.N)
            drift_vec[:eq_end] = daily_drift
            drift_vec[eq_end:bond_end] = params.bond_drift / 252
            drift_vec[bond_end:comm_end] = params.commodity_drift / 252
            drift_vec[comm_end:fx_end] = 0.0  # FX: near-zero drift
            drift_vec[fx_end:] = daily_drift * 1.5  # Crypto: amplified

            returns = drift_vec + vol_scale * z

            row = {"day": day, "regime": regime.value}
            for i, name in enumerate(self.prices.keys()):
                row[f"{name}_return"] = returns[i]

            # Add regime transition flag
            row["is_transition"] = day in transition_days
            all_data.append(row)

        return pd.DataFrame(all_data)


def get_historical_scenarios() -> List[ScenarioConfig]:
    """Get predefined historical scenarios.

    Returns:
        List of scenario configs mimicking historical events.
    """
    return [
        ScenarioConfig(
            name="2008_Financial_Crisis",
            num_days=1008,
            regime_sequence=[
                Regime.BULL, Regime.BEAR, Regime.CRASH,
                Regime.RECOVERY, Regime.BULL,
            ],
            seed=2008,
        ),
        ScenarioConfig(
            name="2020_Pandemic",
            num_days=756,
            regime_sequence=[
                Regime.BULL, Regime.CRASH, Regime.RECOVERY,
                Regime.BULL,
            ],
            seed=2020,
        ),
        ScenarioConfig(
            name="2022_Rate_Hikes",
            num_days=504,
            regime_sequence=[
                Regime.BULL, Regime.STAGNATION, Regime.BEAR,
                Regime.STAGNATION,
            ],
            seed=2022,
        ),
        ScenarioConfig(
            name="Long_Cycle_10yr",
            num_days=2520,
            regime_sequence=[
                Regime.BULL, Regime.STAGNATION, Regime.BEAR,
                Regime.CRASH, Regime.RECOVERY, Regime.BULL,
                Regime.STAGNATION,
            ],
            seed=100,
        ),
        ScenarioConfig(
            name="Rapid_Oscillation",
            num_days=1260,
            regime_sequence=[
                Regime.BULL, Regime.BEAR, Regime.BULL, Regime.BEAR,
                Regime.CRASH, Regime.RECOVERY, Regime.STAGNATION,
            ],
            seed=200,
        ),
    ]


if __name__ == "__main__":
    config = AssetConfig(num_equities=20, num_bonds=5, num_commodities=3,
                          num_currencies=2, num_crypto=1)
    sim = MarketSimulator(config, seed=42)

    scenario = ScenarioConfig(name="test", num_days=504, seed=42)
    data = sim.simulate_scenario(scenario)

    print(f"Simulated {len(data)} days for {config.total_assets} assets")
    print(f"Regime distribution:\n{data['regime'].value_counts()}")
    print(f"Number of regime transitions: {data['is_transition'].sum()}")
    print(f"\nMean daily returns by regime:")
    for regime in data['regime'].unique():
        mask = data['regime'] == regime
        mean_ret = data.filter(like='_return').loc[mask].mean().mean() * 252
        print(f"  {regime}: {mean_ret:.2%} annualized")
