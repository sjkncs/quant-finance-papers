"""
main.py - QuantBench Evaluation Pipeline
=========================================
Main script for running QuantBench simulations and evaluating
trading agents across long-horizon market scenarios.

Usage:
    python main.py --mode run --num_days 504 --agent regime_aware
    python main.py --mode compare --num_scenarios 5
    python main.py --mode regime_analysis
    python main.py --mode ablation

Key features:
- Full multi-year simulation with regime transitions
- Multiple agent evaluation (momentum, mean-reversion, regime-aware, ensemble)
- Survival metrics (survival rate, recovery time, max drawdown)
- Regime detection accuracy analysis
- Performance stratified by regime type
"""

import argparse
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from data import (
    AssetConfig,
    MarketSimulator,
    RegimeSimulator,
    ScenarioConfig,
    Regime,
    REGIME_PARAMS,
    get_historical_scenarios,
)
from model import (
    MomentumAgent,
    MeanReversionAgent,
    RegimeAwareAgent,
    EnsembleAgent,
    RiskManager,
    RegimeDetector,
    TradeAction,
    Strategy,
)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    initial_capital: float = 1e8  # $100M
    ruin_threshold: float = 1e7   # $10M
    transaction_cost_bps: float = 5.0  # 5 bps per trade
    slippage_bps: float = 2.0
    num_days: int = 1008  # 4 years for quick testing
    seed: int = 42


@dataclass
class AgentResult:
    """Results from running an agent on a scenario."""
    agent_name: str
    scenario_name: str
    final_capital: float
    cagr: float
    max_drawdown: float
    sharpe: float
    survival: bool
    recovery_days: int
    regime_detection_accuracy: float
    num_strategy_switches: int
    equity_curve: List[float]
    regime_history: List[str]
    daily_returns: List[float]


def run_simulation(
    agent,
    scenario: ScenarioConfig,
    config: EvalConfig,
    asset_config: AssetConfig,
) -> AgentResult:
    """Run a single agent on a single scenario.

    Args:
        agent: Trading agent instance.
        scenario: Scenario to simulate.
        config: Evaluation configuration.
        asset_config: Asset universe configuration.

    Returns:
        AgentResult with full evaluation metrics.
    """
    # Generate market data
    sim = MarketSimulator(asset_config, seed=scenario.seed)
    market_data = sim.simulate_scenario(scenario)

    N = asset_config.total_assets
    return_cols = [c for c in market_data.columns if c.endswith("_return")]
    returns_matrix = market_data[return_cols].values
    regimes = market_data["regime"].values
    transition_flags = market_data["is_transition"].values

    num_days = min(config.num_days, len(market_data))

    # Agent state
    capital = config.initial_capital
    equity_curve = [capital]
    daily_returns_list = []
    current_weights = np.zeros(N)
    regime_history = []
    strategy_history = []
    high_water_mark = capital
    max_dd = 0.0
    recovery_start = None
    recovery_days = 0
    regime_predictions = []
    true_regimes_at_prediction = []

    for day in range(num_days):
        if day < 60:
            # Warmup period: equal weight
            current_weights = np.ones(N) / N
            daily_ret = returns_matrix[day] @ current_weights
            cost = np.abs(current_weights - np.zeros(N)).sum() * config.transaction_cost_bps * 1e-4
            capital *= (1 + daily_ret - cost)
            equity_curve.append(capital)
            daily_returns_list.append(daily_ret - cost)
            continue

        # Get historical returns for agent
        hist_returns = returns_matrix[:day + 1]

        # Agent decision
        action = agent.decide(hist_returns, capital, equity_curve)

        # Compute trade cost
        weight_change = np.abs(action.target_weights - current_weights).sum()
        trade_cost = weight_change * config.transaction_cost_bps * 1e-4
        slippage = weight_change * config.slippage_bps * 1e-4

        # Execute
        daily_ret = returns_matrix[day] @ action.target_weights
        net_ret = daily_ret - trade_cost - slippage
        capital *= (1 + net_ret)
        current_weights = action.target_weights

        equity_curve.append(capital)
        daily_returns_list.append(net_ret)

        # Track regime detection
        if hasattr(agent, 'detector'):
            port_ret = hist_returns[-60:].mean(axis=1)
            detected = agent.detector.detect_regime(port_ret)
            regime_predictions.append(detected)
            true_regimes_at_prediction.append(regimes[day])

        if hasattr(agent, 'current_regime'):
            regime_history.append(agent.current_regime)
        strategy_history.append(action.strategy.value)

        # Track drawdown
        if capital > high_water_mark:
            high_water_mark = capital
            if recovery_start is not None:
                recovery_days = day - recovery_start
                recovery_start = None

        dd = (capital - high_water_mark) / high_water_mark
        if dd < max_dd:
            max_dd = dd
            if recovery_start is None:
                recovery_start = day

    # If never recovered, set recovery days to remaining time
    if recovery_start is not None:
        recovery_days = num_days - recovery_start

    # Compute metrics
    total_return = capital / config.initial_capital
    years = num_days / 252
    cagr = total_return ** (1 / max(years, 0.01)) - 1

    ret_array = np.array(daily_returns_list)
    sharpe = (ret_array.mean() * 252) / max(ret_array.std() * np.sqrt(252), 1e-8)

    survival = capital > config.ruin_threshold

    # Regime detection accuracy
    if len(regime_predictions) > 0:
        correct = sum(1 for p, t in zip(regime_predictions, true_regimes_at_prediction) if p == t)
        regime_accuracy = correct / len(regime_predictions)
    else:
        regime_accuracy = 0.0

    # Strategy switches
    num_switches = sum(
        1 for i in range(1, len(strategy_history))
        if strategy_history[i] != strategy_history[i - 1]
    )

    agent_name = type(agent).__name__
    return AgentResult(
        agent_name=agent_name,
        scenario_name=scenario.name,
        final_capital=capital,
        cagr=cagr,
        max_drawdown=max_dd,
        sharpe=sharpe,
        survival=survival,
        recovery_days=recovery_days,
        regime_detection_accuracy=regime_accuracy,
        num_strategy_switches=num_switches,
        equity_curve=equity_curve,
        regime_history=regime_history,
        daily_returns=daily_returns_list,
    )


def create_agents() -> Dict[str, object]:
    """Create all trading agents for evaluation.

    Returns:
        Dictionary mapping agent name to agent instance.
    """
    return {
        "Momentum": MomentumAgent(
            lookback=60, top_k=10,
            risk_manager=RiskManager(max_drawdown_trigger=-0.15),
        ),
        "MeanReversion": MeanReversionAgent(
            lookback=20, num_oversold=15,
            risk_manager=RiskManager(max_drawdown_trigger=-0.15),
        ),
        "RegimeAware": RegimeAwareAgent(
            regime_detector=RegimeDetector(),
            risk_manager=RiskManager(max_drawdown_trigger=-0.20, target_vol=0.12),
        ),
        "Ensemble": EnsembleAgent(seed=42),
    }


def run_full_evaluation(
    scenarios: List[ScenarioConfig],
    config: EvalConfig,
    asset_config: AssetConfig,
) -> pd.DataFrame:
    """Run full evaluation of all agents across all scenarios.

    Args:
        scenarios: List of scenarios.
        config: Evaluation configuration.
        asset_config: Asset universe configuration.

    Returns:
        DataFrame with results for all agent-scenario pairs.
    """
    agents = create_agents()
    all_results = []

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario.name}")
        for agent_name, agent in agents.items():
            # Fresh agent instance for each scenario
            if agent_name == "Momentum":
                agent_inst = MomentumAgent(lookback=60, top_k=10)
            elif agent_name == "MeanReversion":
                agent_inst = MeanReversionAgent(lookback=20)
            elif agent_name == "RegimeAware":
                agent_inst = RegimeAwareAgent()
            else:
                agent_inst = EnsembleAgent(seed=scenario.seed)

            result = run_simulation(agent_inst, scenario, config, asset_config)
            all_results.append({
                "agent": agent_name,
                "scenario": scenario.name,
                "final_capital": result.final_capital,
                "cagr": result.cagr,
                "max_drawdown": result.max_drawdown,
                "sharpe": result.sharpe,
                "survival": result.survival,
                "recovery_days": result.recovery_days,
                "regime_accuracy": result.regime_detection_accuracy,
                "strategy_switches": result.num_strategy_switches,
            })
            status = "SURVIVED" if result.survival else "FAILED"
            print(f"    {agent_name:<15}: CAGR={result.cagr:>7.2%}, "
                  f"MaxDD={result.max_drawdown:>8.1%}, "
                  f"Sharpe={result.sharpe:>6.2f}, [{status}]")

    return pd.DataFrame(all_results)


def compute_summary_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics across scenarios.

    Args:
        results_df: Full results DataFrame.

    Returns:
        Summary DataFrame with one row per agent.
    """
    summaries = []
    for agent in results_df["agent"].unique():
        agent_data = results_df[results_df["agent"] == agent]
        summaries.append({
            "agent": agent,
            "survival_rate": agent_data["survival"].mean(),
            "avg_cagr": agent_data["cagr"].mean(),
            "avg_max_dd": agent_data["max_drawdown"].mean(),
            "avg_sharpe": agent_data["sharpe"].mean(),
            "avg_recovery_days": agent_data["recovery_days"].mean(),
            "avg_regime_accuracy": agent_data["regime_accuracy"].mean(),
            "avg_strategy_switches": agent_data["strategy_switches"].mean(),
        })
    return pd.DataFrame(summaries).sort_values("survival_rate", ascending=False)


def regime_stratified_analysis(
    results_df: pd.DataFrame,
    scenarios: List[ScenarioConfig],
    config: EvalConfig,
    asset_config: AssetConfig,
) -> Dict[str, Dict[str, float]]:
    """Analyze agent performance stratified by regime type.

    Returns:
        Nested dictionary: agent -> regime -> metric.
    """
    agents = create_agents()
    regime_performance: Dict[str, Dict[str, List[float]]] = {}

    # Run one scenario and track per-regime returns
    scenario = scenarios[0]
    sim = MarketSimulator(asset_config, seed=scenario.seed)
    market_data = sim.simulate_scenario(scenario)
    return_cols = [c for c in market_data.columns if c.endswith("_return")]
    returns_matrix = market_data[return_cols].values
    regimes = market_data["regime"].values

    for agent_name, agent in agents.items():
        regime_performance[agent_name] = {
            "bull": [], "bear": [], "crash": [],
            "recovery": [], "stagnation": [],
        }

        # Simple equal-weight portfolio per regime
        for day in range(60, min(config.num_days, len(market_data))):
            regime = regimes[day]
            if regime in regime_performance[agent_name]:
                daily_ret = returns_matrix[day].mean()  # simplified
                regime_performance[agent_name][regime].append(daily_ret)

    # Compute Sharpe per regime
    result = {}
    for agent_name, regime_data in regime_performance.items():
        result[agent_name] = {}
        for regime, returns in regime_data.items():
            if len(returns) > 10:
                arr = np.array(returns)
                sharpe = (arr.mean() * 252) / max(arr.std() * np.sqrt(252), 1e-8)
                result[agent_name][regime] = float(sharpe)
            else:
                result[agent_name][regime] = 0.0

    return result


def main():
    parser = argparse.ArgumentParser(description="QuantBench Evaluation")
    parser.add_argument("--mode", type=str, default="compare",
                        choices=["run", "compare", "regime_analysis", "ablation"])
    parser.add_argument("--num_days", type=int, default=504)
    parser.add_argument("--num_assets", type=int, default=30)
    parser.add_argument("--num_scenarios", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    config = EvalConfig(num_days=args.num_days, seed=args.seed)
    asset_config = AssetConfig(
        num_equities=args.num_assets,
        num_bonds=max(3, args.num_assets // 10),
        num_commodities=max(2, args.num_assets // 25),
        num_currencies=2,
        num_crypto=1,
    )

    scenarios = get_historical_scenarios()[:args.num_scenarios]
    # Adjust scenario length
    for s in scenarios:
        s.num_days = min(s.num_days, args.num_days)

    if args.mode == "compare":
        print("=" * 70)
        print("QuantBench: Long-Horizon Trading Agent Evaluation")
        print("=" * 70)
        print(f"Assets: {asset_config.total_assets}, Days: {args.num_days}, "
              f"Scenarios: {len(scenarios)}")

        start_time = time.time()
        results = run_full_evaluation(scenarios, config, asset_config)
        elapsed = time.time() - start_time

        print(f"\n{'=' * 70}")
        print("Summary Statistics")
        print("=" * 70)
        summary = compute_summary_stats(results)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nTotal evaluation time: {elapsed:.1f}s")

    elif args.mode == "run":
        print("=" * 70)
        print("Single Agent Run")
        print("=" * 70)
        agents = create_agents()
        scenario = scenarios[0]
        scenario.num_days = args.num_days

        for name, agent in agents.items():
            if name == "RegimeAware":
                agent = RegimeAwareAgent()
            elif name == "Ensemble":
                agent = EnsembleAgent(seed=args.seed)

            result = run_simulation(agent, scenario, config, asset_config)
            print(f"\n{name}:")
            print(f"  Final Capital: ${result.final_capital:,.0f}")
            print(f"  CAGR: {result.cagr:.2%}")
            print(f"  Max Drawdown: {result.max_drawdown:.1%}")
            print(f"  Sharpe: {result.sharpe:.2f}")
            print(f"  Survival: {'YES' if result.survival else 'NO'}")
            print(f"  Recovery Days: {result.recovery_days}")
            print(f"  Regime Detection: {result.regime_detection_accuracy:.1%}")
            print(f"  Strategy Switches: {result.num_strategy_switches}")

    elif args.mode == "regime_analysis":
        print("=" * 70)
        print("Regime-Stratified Analysis")
        print("=" * 70)

        regime_perf = regime_stratified_analysis(
            pd.DataFrame(), scenarios, config, asset_config
        )

        print(f"\n{'Agent':<15} {'Bull':>8} {'Bear':>8} {'Crash':>8} "
              f"{'Recovery':>10} {'Stagnation':>12}")
        print("-" * 70)
        for agent, regime_data in regime_perf.items():
            print(f"{agent:<15} "
                  f"{regime_data.get('bull', 0):>8.2f} "
                  f"{regime_data.get('bear', 0):>8.2f} "
                  f"{regime_data.get('crash', 0):>8.2f} "
                  f"{regime_data.get('recovery', 0):>10.2f} "
                  f"{regime_data.get('stagnation', 0):>12.2f}")

    elif args.mode == "ablation":
        print("=" * 70)
        print("Ablation: Regime Detection vs Alpha Generation")
        print("=" * 70)

        scenario = scenarios[0]
        scenario.num_days = args.num_days

        # Baseline: momentum agent (no regime detection)
        base_agent = MomentumAgent(lookback=60)
        base_result = run_simulation(base_agent, scenario, config, asset_config)
        print(f"\nBaseline (Momentum, no regime detection):")
        print(f"  Survival: {base_result.survival}, CAGR: {base_result.cagr:.2%}, "
              f"MaxDD: {base_result.max_drawdown:.1%}")

        # With regime detection
        regime_agent = RegimeAwareAgent()
        regime_result = run_simulation(regime_agent, scenario, config, asset_config)
        print(f"\nRegimeAware (with regime detection):")
        print(f"  Survival: {regime_result.survival}, CAGR: {regime_result.cagr:.2%}, "
              f"MaxDD: {regime_result.max_drawdown:.1%}")

        # Ensemble
        ensemble_agent = EnsembleAgent(seed=args.seed)
        ens_result = run_simulation(ensemble_agent, scenario, config, asset_config)
        print(f"\nEnsemble (multi-strategy):")
        print(f"  Survival: {ens_result.survival}, CAGR: {ens_result.cagr:.2%}, "
              f"MaxDD: {ens_result.max_drawdown:.1%}")

        # Improvement analysis
        dd_improvement = base_result.max_drawdown - regime_result.max_drawdown
        cagr_improvement = regime_result.cagr - base_result.cagr
        print(f"\nRegime detection impact:")
        print(f"  MaxDD improvement: {dd_improvement:.1%}")
        print(f"  CAGR improvement: {cagr_improvement:.2%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
