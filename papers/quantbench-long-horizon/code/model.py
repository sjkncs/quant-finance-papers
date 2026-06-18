"""
model.py - QuantBench Trading Agent Implementations
====================================================
Implements multiple trading agent architectures for evaluation
on the QuantBench benchmark.

Key components:
- TradingAgent: Base agent interface
- MomentumAgent: Trend-following strategy agent
- MeanReversionAgent: Mean-reversion strategy agent
- RegimeAwareAgent: Agent with explicit regime detection
- PortfolioOptimizer: Multi-strategy portfolio optimizer
- RiskManager: Portfolio risk management module
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class Strategy(Enum):
    """Available trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CARRY = "carry"
    VOL_TARGET = "vol_target"


@dataclass
class AgentState:
    """Current state of a trading agent."""
    capital: float
    day: int
    portfolio_weights: np.ndarray
    gross_exposure: float
    current_strategy: Strategy
    detected_regime: str
    max_drawdown: float
    high_water_mark: float
    equity_curve: List[float] = field(default_factory=list)
    regime_history: List[str] = field(default_factory=list)
    strategy_history: List[str] = field(default_factory=list)


@dataclass
class TradeAction:
    """Action taken by the agent at a single time step."""
    target_weights: np.ndarray
    strategy: Strategy
    risk_limit: float
    hedge_ratio: float


class RiskManager:
    """Portfolio risk management module.

    Monitors portfolio risk metrics and adjusts exposure based on
    drawdown limits and volatility targets.
    """

    def __init__(
        self,
        max_drawdown_trigger: float = -0.15,
        target_vol: float = 0.15,
        max_gross_exposure: float = 2.0,
        max_position: float = 0.10,
    ):
        self.max_drawdown_trigger = max_drawdown_trigger
        self.target_vol = target_vol
        self.max_gross_exposure = max_gross_exposure
        self.max_position = max_position

    def check_risk_limits(
        self,
        weights: np.ndarray,
        equity_curve: List[float],
        realized_vol: float,
    ) -> Tuple[np.ndarray, bool]:
        """Check risk limits and adjust weights if necessary.

        Args:
            weights: Proposed portfolio weights.
            equity_curve: Historical equity values.
            realized_vol: Current realized volatility.

        Returns:
            Tuple of (adjusted_weights, risk_limit_triggered).
        """
        adjusted = weights.copy()
        triggered = False

        # Check drawdown trigger
        if len(equity_curve) > 1:
            current = equity_curve[-1]
            peak = max(equity_curve)
            drawdown = (current - peak) / peak
            if drawdown < self.max_drawdown_trigger:
                adjusted *= 0.5  # reduce exposure by half
                triggered = True

        # Volatility targeting
        if realized_vol > 0:
            vol_scalar = min(self.target_vol / realized_vol, 1.5)
            adjusted *= vol_scalar

        # Gross exposure limit
        gross = np.abs(adjusted).sum()
        if gross > self.max_gross_exposure:
            adjusted *= self.max_gross_exposure / gross

        # Position size limit
        for i in range(len(adjusted)):
            if abs(adjusted[i]) > self.max_position:
                adjusted[i] = np.sign(adjusted[i]) * self.max_position

        return adjusted, triggered


class RegimeDetector:
    """Detects market regime transitions from price data.

    Uses a combination of volatility regime, momentum regime,
    and correlation changes to classify the current market state.
    """

    def __init__(
        self,
        lookback_short: int = 10,
        lookback_medium: int = 30,
        lookback_long: int = 60,
        vol_threshold_high: float = 0.25,
        vol_threshold_crash: float = 0.40,
    ):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_crash = vol_threshold_crash

    def detect_regime(self, returns_history: np.ndarray) -> str:
        """Detect the current market regime.

        Uses a simple rule-based classifier:
        - High vol + negative return -> Crash or Bear
        - Low vol + positive return -> Bull
        - Low vol + zero return -> Stagnation
        - Declining vol + positive return -> Recovery

        Args:
            returns_history: Array of daily portfolio returns.

        Returns:
            Detected regime name string.
        """
        if len(returns_history) < self.lookback_long:
            return "unknown"

        recent_short = returns_history[-self.lookback_short:]
        recent_medium = returns_history[-self.lookback_medium:]
        recent_long = returns_history[-self.lookback_long:]

        vol_short = recent_short.std() * np.sqrt(252)
        vol_long = recent_long.std() * np.sqrt(252)
        ret_medium = recent_medium.mean() * 252
        vol_declining = vol_short < vol_long * 0.8

        if vol_short > self.vol_threshold_crash and ret_medium < -0.10:
            return "crash"
        elif vol_short > self.vol_threshold_high and ret_medium < -0.05:
            return "bear"
        elif vol_declining and ret_medium > 0.05:
            return "recovery"
        elif vol_short < 0.15 and abs(ret_medium) < 0.03:
            return "stagnation"
        elif vol_short < 0.20 and ret_medium > 0.03:
            return "bull"
        else:
            return "bull"  # default

    def detect_transition(
        self,
        returns_history: np.ndarray,
        previous_regime: str,
    ) -> Tuple[str, bool]:
        """Detect if a regime transition has occurred.

        Args:
            returns_history: Array of daily returns.
            previous_regime: Previously detected regime.

        Returns:
            Tuple of (current_regime, transition_detected).
        """
        current = self.detect_regime(returns_history)
        transition = current != previous_regime
        return current, transition


class MomentumAgent:
    """Trend-following trading agent.

    Allocates based on past momentum signals: overweight winners,
    underweight losers. No regime detection capability.
    """

    def __init__(
        self,
        lookback: int = 60,
        top_k: int = 10,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.lookback = lookback
        self.top_k = top_k
        self.risk_manager = risk_manager or RiskManager()
        self.strategy = Strategy.MOMENTUM

    def decide(
        self,
        returns_data: np.ndarray,
        capital: float,
        equity_curve: List[float],
    ) -> TradeAction:
        """Generate trading decision based on momentum signals.

        Args:
            returns_data: Asset returns matrix (days x assets).
            capital: Current capital.
            equity_curve: Historical equity values.

        Returns:
            TradeAction with target weights.
        """
        N = returns_data.shape[1]

        if len(returns_data) < self.lookback:
            weights = np.ones(N) / N
        else:
            # Compute momentum scores
            recent = returns_data[-self.lookback:]
            momentum_scores = recent.mean(axis=0) * 252

            # Top-k momentum portfolio
            top_indices = np.argsort(momentum_scores)[-self.top_k:]
            weights = np.zeros(N)
            for idx in top_indices:
                if momentum_scores[idx] > 0:
                    weights[idx] = 1.0 / self.top_k
            # If no positive momentum, go to cash
            if weights.sum() < 0.01:
                weights = np.zeros(N)

        # Risk management
        realized_vol = returns_data[-20:].std() * np.sqrt(252) if len(returns_data) >= 20 else 0.15
        mean_portfolio_return = (returns_data[-20:] @ weights).mean() if len(returns_data) >= 20 else 0
        portfolio_returns = returns_data[-60:] @ weights if len(returns_data) >= 60 else np.array([0])
        weights, _ = self.risk_manager.check_risk_limits(
            weights, equity_curve, realized_vol
        )

        return TradeAction(
            target_weights=weights,
            strategy=self.strategy,
            risk_limit=1.0,
            hedge_ratio=0.0,
        )


class MeanReversionAgent:
    """Mean-reversion trading agent.

    Overweights recently underperforming assets, expecting
    price normalization. No regime detection capability.
    """

    def __init__(
        self,
        lookback: int = 20,
        num_oversold: int = 15,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.lookback = lookback
        self.num_oversold = num_oversold
        self.risk_manager = risk_manager or RiskManager()
        self.strategy = Strategy.MEAN_REVERSION

    def decide(
        self,
        returns_data: np.ndarray,
        capital: float,
        equity_curve: List[float],
    ) -> TradeAction:
        """Generate mean-reversion trading decision."""
        N = returns_data.shape[1]

        if len(returns_data) < self.lookback:
            weights = np.ones(N) / N
        else:
            recent = returns_data[-self.lookback:]
            scores = recent.mean(axis=0) * 252  # annualized

            # Overweight worst performers (expecting reversion)
            bottom_indices = np.argsort(scores)[:self.num_oversold]
            weights = np.zeros(N)
            for idx in bottom_indices:
                if scores[idx] < 0:  # only buy truly oversold
                    weights[idx] = 1.0 / self.num_oversold

            if weights.sum() < 0.01:
                weights = np.ones(N) / N  # equal weight if no signal

        realized_vol = returns_data[-20:].std() * np.sqrt(252) if len(returns_data) >= 20 else 0.15
        weights, _ = self.risk_manager.check_risk_limits(
            weights, equity_curve, realized_vol
        )

        return TradeAction(
            target_weights=weights,
            strategy=self.strategy,
            risk_limit=1.0,
            hedge_ratio=0.0,
        )


class RegimeAwareAgent:
    """Agent with explicit regime detection and strategy switching.

    Detects the current regime and switches between strategies:
    - Bull: Momentum
    - Bear: Reduced exposure + carry
    - Crash: Cash / hedge
    - Recovery: Aggressive momentum
    - Stagnation: Mean reversion + carry
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.detector = regime_detector or RegimeDetector()
        self.risk_manager = risk_manager or RiskManager(
            max_drawdown_trigger=-0.20,
            target_vol=0.12,
        )
        self.momentum_agent = MomentumAgent(lookback=60, top_k=10)
        self.mean_rev_agent = MeanReversionAgent(lookback=20, num_oversold=15)
        self.current_regime = "bull"
        self.regime_history: List[str] = []

    def decide(
        self,
        returns_data: np.ndarray,
        capital: float,
        equity_curve: List[float],
    ) -> TradeAction:
        """Generate regime-aware trading decision."""
        # Detect current regime
        portfolio_returns = returns_data[-120:].mean(axis=1) if returns_data.shape[0] >= 120 else returns_data.mean(axis=1)
        new_regime, transition = self.detector.detect_transition(
            portfolio_returns, self.current_regime
        )

        if transition:
            self.current_regime = new_regime
            self.regime_history.append(new_regime)

        N = returns_data.shape[1]

        # Select strategy based on regime
        if self.current_regime == "crash":
            # Defensive: mostly cash with small bond allocation
            weights = np.zeros(N)
            bond_end = min(N, 60)  # assume bonds are after equities
            bond_start = min(N, 50)
            if bond_start < bond_end:
                bond_weight = 0.3 / (bond_end - bond_start)
                weights[bond_start:bond_end] = bond_weight
            action = TradeAction(weights, Strategy.VOL_TARGET, 0.5, 0.8)

        elif self.current_regime == "bear":
            # Reduced exposure with carry focus
            base_action = self.mean_rev_agent.decide(returns_data, capital, equity_curve)
            weights = base_action.target_weights * 0.5  # reduce exposure
            action = TradeAction(weights, Strategy.CARRY, 0.8, 0.3)

        elif self.current_regime == "recovery":
            # Aggressive momentum
            base_action = self.momentum_agent.decide(returns_data, capital, equity_curve)
            weights = base_action.target_weights * 1.3  # increase exposure
            action = TradeAction(weights, Strategy.MOMENTUM, 1.2, 0.0)

        elif self.current_regime == "stagnation":
            # Mean reversion with moderate exposure
            action = self.mean_rev_agent.decide(returns_data, capital, equity_curve)
            action = TradeAction(
                action.target_weights * 0.8, Strategy.MEAN_REVERSION, 1.0, 0.1
            )

        else:  # bull
            action = self.momentum_agent.decide(returns_data, capital, equity_curve)

        # Apply risk management
        realized_vol = returns_data[-20:].std() * np.sqrt(252) if len(returns_data) >= 20 else 0.15
        weights, triggered = self.risk_manager.check_risk_limits(
            action.target_weights, equity_curve, realized_vol
        )
        if triggered:
            action = TradeAction(weights, action.strategy, 0.5, 0.5)

        return action


class EnsembleAgent:
    """Multi-strategy ensemble agent that combines signals.

    Runs multiple sub-agents and combines their signals based
    on recent performance and regime assessment.
    """

    def __init__(self, seed: int = 42):
        self.momentum = MomentumAgent(lookback=40, top_k=10)
        self.momentum_long = MomentumAgent(lookback=120, top_k=15)
        self.mean_rev = MeanReversionAgent(lookback=20, num_oversold=15)
        self.regime_aware = RegimeAwareAgent()
        self.rng = np.random.RandomState(seed)
        self.risk_manager = RiskManager(
            max_drawdown_trigger=-0.18,
            target_vol=0.14,
        )

    def decide(
        self,
        returns_data: np.ndarray,
        capital: float,
        equity_curve: List[float],
    ) -> TradeAction:
        """Generate ensemble trading decision."""
        # Get signals from all sub-agents
        actions = [
            self.momentum.decide(returns_data, capital, equity_curve),
            self.momentum_long.decide(returns_data, capital, equity_curve),
            self.mean_rev.decide(returns_data, capital, equity_curve),
            self.regime_aware.decide(returns_data, capital, equity_curve),
        ]

        # Simple equal-weight ensemble
        weights = np.mean([a.target_weights for a in actions], axis=0)

        # Risk management
        realized_vol = returns_data[-20:].std() * np.sqrt(252) if len(returns_data) >= 20 else 0.15
        weights, triggered = self.risk_manager.check_risk_limits(
            weights, equity_curve, realized_vol
        )

        # Determine dominant strategy
        strategy_counts = {}
        for a in actions:
            s = a.strategy.value
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        dominant = max(strategy_counts, key=strategy_counts.get)

        return TradeAction(
            target_weights=weights,
            strategy=Strategy(dominant),
            risk_limit=1.0,
            hedge_ratio=0.15,
        )


if __name__ == "__main__":
    # Quick smoke test
    N = 20
    T = 200
    returns = np.random.randn(T, N) * 0.02

    agents = {
        "Momentum": MomentumAgent(),
        "MeanReversion": MeanReversionAgent(),
        "RegimeAware": RegimeAwareAgent(),
        "Ensemble": EnsembleAgent(),
    }

    for name, agent in agents.items():
        action = agent.decide(returns, 1e8, [1e8] * 100)
        exposure = np.abs(action.target_weights).sum()
        print(f"{name}: exposure={exposure:.2f}, strategy={action.strategy.value}")

    print("\nAll agents smoke test passed!")
