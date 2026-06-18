"""
data.py - Synthetic Market Data Generation for Trade Execution
=============================================================
Generates realistic institutional order flow data and simulated market
microstructure for training and evaluating execution agents.

Key features:
- Synthetic order generation with realistic size/horizon distributions
- Market microstructure simulation (order book, spread, volume)
- U-shaped intraday volume profile
- Regime-dependent volatility (VIX-like)
- Implementation shortfall calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class OrderConfig:
    """Configuration for synthetic order generation."""
    num_orders: int = 10000
    min_size: float = 1e6
    max_size: float = 500e6
    median_size: float = 12e6
    min_horizon_intervals: int = 12
    max_horizon_intervals: int = 96
    default_horizon_intervals: int = 48
    interval_minutes: int = 5
    num_assets: int = 50
    seed: int = 42


@dataclass
class MarketState:
    """Current market microstructure state."""
    mid_price: float
    spread: float
    bid_depth: np.ndarray  # depth at best 3 bid levels
    ask_depth: np.ndarray  # depth at best 3 ask levels
    recent_volume: np.ndarray  # volume over last k intervals
    realized_vol: float
    vix_level: float


@dataclass
class ExecutionResult:
    """Result of executing a child order."""
    executed_shares: float
    avg_price: float
    market_impact_bps: float
    slippage_bps: float


class IntradayVolumeProfile:
    """Models the U-shaped intraday volume profile."""

    def __init__(self, num_intervals: int = 48):
        self.num_intervals = num_intervals
        self.profile = self._build_profile()

    def _build_profile(self) -> np.ndarray:
        """Build U-shaped volume profile: high at open/close, low mid-day."""
        t = np.linspace(0, 1, self.num_intervals)
        # U-shape: combination of exponential decay from open and rise toward close
        open_effect = 2.5 * np.exp(-5 * t)
        close_effect = 2.0 * np.exp(-5 * (1 - t))
        base = 0.5
        profile = base + open_effect + close_effect
        profile = profile / profile.sum()
        return profile

    def sample_volume(self, interval_idx: int, total_daily_volume: float) -> float:
        """Sample volume for a specific intraday interval."""
        noise = np.random.lognormal(0, 0.3)
        return total_daily_volume * self.profile[interval_idx] * noise


class MarketSimulator:
    """Simulates market microstructure for execution."""

    def __init__(
        self,
        base_price: float = 100.0,
        base_spread_bps: float = 5.0,
        daily_vol: float = 0.02,
        vix_level: float = 18.0,
        num_intervals: int = 48,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.base_price = base_price
        self.base_spread_bps = base_spread_bps
        self.daily_vol = daily_vol
        self.vix_level = vix_level
        self.num_intervals = num_intervals
        self.interval_vol = daily_vol / np.sqrt(num_intervals)
        self.volume_profile = IntradayVolumeProfile(num_intervals)
        self.daily_volume = 5e6  # shares
        self._reset()

    def _reset(self):
        """Reset market state to initial conditions."""
        vol_multiplier = self.vix_level / 18.0
        self.state = MarketState(
            mid_price=self.base_price,
            spread=self.base_spread_bps * 1e-4 * self.base_price * vol_multiplier,
            bid_depth=self.rng.uniform(500, 5000, size=3),
            ask_depth=self.rng.uniform(500, 5000, size=3),
            recent_volume=np.ones(5) * self.daily_volume / self.num_intervals,
            realized_vol=self.daily_vol * vol_multiplier,
            vix_level=self.vix_level,
        )

    def step(self, interval_idx: int) -> MarketState:
        """Advance market by one interval (no execution)."""
        vol_multiplier = self.vix_level / 18.0
        price_change = self.rng.normal(0, self.interval_vol * vol_multiplier)
        self.state.mid_price *= (1 + price_change)

        # Update spread
        spread_noise = self.rng.lognormal(0, 0.2)
        self.state.spread = (
            self.base_spread_bps * 1e-4 * self.state.mid_price * vol_multiplier * spread_noise
        )

        # Update order book depth
        self.state.bid_depth = self.rng.uniform(500, 5000, size=3)
        self.state.ask_depth = self.rng.uniform(500, 5000, size=3)

        # Update volume
        interval_vol = self.volume_profile.sample_volume(
            interval_idx, self.daily_volume
        )
        self.state.recent_volume = np.roll(self.state.recent_volume, 1)
        self.state.recent_volume[0] = interval_vol

        return self.state

    def execute_child_order(
        self, shares: float, aggressiveness: float, is_buy: bool
    ) -> ExecutionResult:
        """Execute a child order with market impact."""
        mid = self.state.mid_price
        half_spread = self.state.spread / 2

        # Aggressiveness: 0 = passive limit, 1 = market order
        # Price paid: limit at mid +/- half_spread*(1-2*agg) to market at mid +/- half_spread
        if is_buy:
            exec_price = mid + half_spread * (2 * aggressiveness - 1 + 1) / 2
        else:
            exec_price = mid - half_spread * (2 * aggressiveness - 1 + 1) / 2

        # Market impact: temporary + permanent, scaled by order size vs available liquidity
        available_liquidity = self.state.ask_depth.sum() if is_buy else self.state.bid_depth.sum()
        participation_rate = shares / max(available_liquidity, 1.0)

        # Temporary impact: sqrt model (Almgren-Chriss style)
        temp_impact = 0.1 * np.sqrt(participation_rate) * mid
        # Permanent impact
        perm_impact = 0.05 * participation_rate * mid

        impact_bps = (temp_impact + perm_impact) / mid * 1e4

        # Apply permanent impact to mid price
        if is_buy:
            self.state.mid_price += perm_impact
        else:
            self.state.mid_price -= perm_impact

        # Slippage from spread crossing
        slippage_bps = abs(exec_price - mid) / mid * 1e4

        # Execute (with partial fill possibility for passive orders)
        fill_prob = 0.5 + 0.5 * aggressiveness
        executed = shares * min(fill_prob * (1 + self.rng.normal(0, 0.1)), 1.0)
        executed = max(executed, 0)

        avg_price = exec_price + (temp_impact if is_buy else -temp_impact) / 2

        return ExecutionResult(
            executed_shares=executed,
            avg_price=avg_price,
            market_impact_bps=impact_bps,
            slippage_bps=slippage_bps,
        )


def generate_synthetic_orders(config: OrderConfig) -> pd.DataFrame:
    """Generate synthetic institutional orders.

    Args:
        config: Order generation configuration.

    Returns:
        DataFrame with columns: order_id, asset_id, size_usd, is_buy,
        horizon_intervals, arrival_price, vix_level, daily_volume.
    """
    rng = np.random.RandomState(config.seed)

    # Log-normal distribution for order sizes (matches empirical distribution)
    log_sizes = rng.normal(np.log(config.median_size), 1.0, size=config.num_orders)
    sizes = np.clip(np.exp(log_sizes), config.min_size, config.max_size)

    # Horizon: mostly default with some variation
    horizons = rng.choice(
        [config.min_horizon_intervals, config.default_horizon_intervals,
         config.max_horizon_intervals],
        size=config.num_orders,
        p=[0.1, 0.7, 0.2],
    )

    # VIX levels: mixture of regimes
    vix_regimes = rng.choice([15.0, 18.0, 22.0, 28.0, 35.0], size=config.num_orders,
                              p=[0.25, 0.30, 0.25, 0.12, 0.08])

    # Arrival prices: random around 100
    arrival_prices = rng.uniform(50, 500, size=config.num_orders)

    orders = pd.DataFrame({
        "order_id": range(config.num_orders),
        "asset_id": rng.randint(0, config.num_assets, size=config.num_orders),
        "size_usd": sizes,
        "is_buy": rng.choice([True, False], size=config.num_orders),
        "horizon_intervals": horizons,
        "arrival_price": arrival_prices,
        "vix_level": vix_regimes,
        "daily_volume": rng.lognormal(np.log(5e6), 0.5, size=config.num_orders),
    })

    return orders


def simulate_order_execution(
    order: pd.Series,
    seed: int = 0,
) -> Dict:
    """Simulate the full execution of a single parent order.

    Args:
        order: Series representing a single order's parameters.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with execution trajectory and implementation shortfall.
    """
    rng = np.random.RandomState(seed + int(order["order_id"]))
    sim = MarketSimulator(
        base_price=order["arrival_price"],
        vix_level=order["vix_level"],
        num_intervals=int(order["horizon_intervals"]),
        seed=seed + int(order["order_id"]),
    )

    total_shares = order["size_usd"] / order["arrival_price"]
    remaining_shares = total_shares
    executed_total = 0.0
    cost_weighted_price = 0.0
    trajectory = []

    for t in range(int(order["horizon_intervals"])):
        state = sim.step(t)

        # Simple TWAP-like policy as baseline
        target_shares = remaining_shares / max(int(order["horizon_intervals"]) - t, 1)
        target_shares = min(target_shares, remaining_shares)

        # Random aggressiveness for diversity
        agg = rng.uniform(0.3, 0.7)

        if target_shares > 0:
            result = sim.execute_child_order(
                target_shares, agg, bool(order["is_buy"])
            )
            executed_total += result.executed_shares
            cost_weighted_price += result.executed_shares * result.avg_price
            remaining_shares -= result.executed_shares
        else:
            result = ExecutionResult(0, state.mid_price, 0, 0)

        trajectory.append({
            "step": t,
            "remaining_shares": remaining_shares,
            "executed_shares": result.executed_shares,
            "avg_price": result.avg_price,
            "mid_price": state.mid_price,
            "spread": state.spread,
            "aggressiveness": agg,
        })

    # Implementation shortfall
    if executed_total > 0:
        vwap_exec = cost_weighted_price / executed_total
    else:
        vwap_exec = order["arrival_price"]

    if order["is_buy"]:
        impl_shortfall_bps = (vwap_exec - order["arrival_price"]) / order["arrival_price"] * 1e4
    else:
        impl_shortfall_bps = (order["arrival_price"] - vwap_exec) / order["arrival_price"] * 1e4

    return {
        "order_id": order["order_id"],
        "trajectory": trajectory,
        "vwap_exec": vwap_exec,
        "arrival_price": order["arrival_price"],
        "impl_shortfall_bps": impl_shortfall_bps,
        "fill_rate": executed_total / total_shares,
    }


def build_training_batch(
    orders_df: pd.DataFrame,
    batch_size: int = 64,
    seed: int = 42,
) -> List[Dict]:
    """Build a training batch of simulated executions.

    Args:
        orders_df: DataFrame of orders.
        batch_size: Number of orders per batch.
        seed: Random seed.

    Returns:
        List of execution result dictionaries.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(orders_df), size=min(batch_size, len(orders_df)), replace=False)
    batch = []
    for idx in indices:
        result = simulate_order_execution(orders_df.iloc[idx], seed=seed)
        batch.append(result)
    return batch


if __name__ == "__main__":
    config = OrderConfig(num_orders=1000, seed=42)
    orders = generate_synthetic_orders(config)
    print(f"Generated {len(orders)} synthetic orders")
    print(f"Order size range: ${orders['size_usd'].min():,.0f} - ${orders['size_usd'].max():,.0f}")
    print(f"VIX levels: {orders['vix_level'].value_counts().to_dict()}")

    # Simulate a few orders
    for i in range(3):
        result = simulate_order_execution(orders.iloc[i], seed=42)
        print(f"\nOrder {result['order_id']}: "
              f"IS={result['impl_shortfall_bps']:.2f} bps, "
              f"fill={result['fill_rate']:.1%}")
