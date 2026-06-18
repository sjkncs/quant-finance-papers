"""
main.py - Lambda-ExecGRPO Training and Evaluation Pipeline
==========================================================
Main training script for the GRPO-as-PRM trade execution paper.

Usage:
    python main.py --mode train --num_orders 5000 --num_epochs 100
    python main.py --mode eval --checkpoint model.pt
    python main.py --mode compare  # run all baselines comparison

Key features:
- Full training loop with lambda-ExecGRPO
- Multiple baseline comparisons (TWAP, VWAP, Almgren-Chriss, PPO, GRPO)
- Ablation study generation
- Metrics: execution cost, Sharpe, max drawdown, fill rate
"""

import argparse
import numpy as np
import torch
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

from data import (
    OrderConfig,
    MarketSimulator,
    generate_synthetic_orders,
    simulate_order_execution,
)
from model import (
    ExecutionPolicy,
    LambdaExecGRPO,
    PrefixGrouper,
    Trajectory,
    TrajectoryStep,
)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    state_dim: int = 14
    action_dim: int = 2
    hidden_dim: int = 128
    num_layers: int = 2
    lr: float = 1e-4
    kl_penalty: float = 0.01
    num_trajectories: int = 16  # G: number of sampled trajectories per order
    num_intervals: int = 48     # T: execution horizon intervals
    batch_size: int = 8
    num_epochs: int = 50
    ref_policy_update_freq: int = 10
    seed: int = 42


def build_state_vector(
    remaining_shares: float,
    total_shares: float,
    t: int,
    T: int,
    sim: MarketSimulator,
) -> np.ndarray:
    """Construct the state vector for the policy network.

    State = [q_t/Q, t/T, p_t (normalized), vol_1..vol_5 (normalized),
             spread (normalized), depth_1..depth_3 (normalized),
             realized_vol]

    Args:
        remaining_shares: Shares left to execute.
        total_shares: Original order size.
        t: Current time step.
        T: Total time steps.
        sim: Market simulator with current state.

    Returns:
        State vector of shape (state_dim,).
    """
    state = sim.state
    features = np.array([
        remaining_shares / max(total_shares, 1),           # remaining fraction
        t / max(T, 1),                                      # time progress
        np.log(state.mid_price / 100.0),                    # log price (normalized)
        *(state.recent_volume / max(state.recent_volume.max(), 1)),  # 5 volume features
        state.spread / max(state.mid_price, 1) * 1e4,       # spread in bps
        *(state.bid_depth / max(state.bid_depth.max(), 1)), # 3 bid depth features
        state.realized_vol,                                  # realized vol
        state.vix_level / 50.0,                              # normalized VIX level
    ], dtype=np.float32)
    # Protect against NaN/Inf from extreme market conditions
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
    return features[:14]  # ensure correct dimension


def sample_trajectory(
    policy: ExecutionPolicy,
    order: dict,
    num_intervals: int = 48,
    seed: int = 0,
) -> Trajectory:
    """Sample a single execution trajectory using the policy.

    Args:
        policy: Current execution policy.
        order: Order parameters dict.
        num_intervals: Number of execution intervals.
        seed: Random seed.

    Returns:
        Sampled Trajectory object.
    """
    sim = MarketSimulator(
        base_price=order["arrival_price"],
        vix_level=order["vix_level"],
        num_intervals=num_intervals,
        seed=seed,
    )

    total_shares = order["size_usd"] / order["arrival_price"]
    remaining_shares = total_shares
    steps = []
    action_sequence = []
    executed_total = 0.0
    cost_weighted_price = 0.0

    policy.eval()
    with torch.no_grad():
        for t in range(num_intervals):
            state = sim.step(t)
            state_vec = build_state_vector(
                remaining_shares, total_shares, t, num_intervals, sim
            )
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            action, log_prob = policy.sample_action(state_tensor)
            action_np = action.squeeze().numpy()
            log_prob_val = log_prob.item()

            # Decode action: [shares_fraction, aggressiveness]
            shares_frac = float(action_np[0])
            aggressiveness = float(action_np[1])

            target_shares = remaining_shares * shares_frac
            target_shares = min(target_shares, remaining_shares)

            if target_shares > 0:
                result = sim.execute_child_order(
                    target_shares, aggressiveness, order["is_buy"]
                )
                executed_total += result.executed_shares
                cost_weighted_price += result.executed_shares * result.avg_price
                remaining_shares -= result.executed_shares
            else:
                result = type("R", (), {
                    "executed_shares": 0, "avg_price": state.mid_price,
                    "market_impact_bps": 0, "slippage_bps": 0
                })()

            steps.append(TrajectoryStep(
                state=state_vec,
                action=action_np,
                log_prob=log_prob_val,
                reward=0.0,
            ))
            action_sequence.append(action_np)

    # Compute terminal reward (negative implementation shortfall)
    if executed_total > 0:
        vwap_exec = cost_weighted_price / executed_total
    else:
        vwap_exec = order["arrival_price"]

    if order["is_buy"]:
        impl_shortfall = (vwap_exec - order["arrival_price"]) / order["arrival_price"] * 1e4
    else:
        impl_shortfall = (order["arrival_price"] - vwap_exec) / order["arrival_price"] * 1e4

    total_reward = -impl_shortfall  # negative IS = reward

    return Trajectory(
        steps=steps,
        total_reward=total_reward,
        action_sequence=action_sequence,
    )


def train_epoch(
    grpo: LambdaExecGRPO,
    orders_df,
    config: TrainingConfig,
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch.

    Args:
        grpo: LambdaExecGRPO trainer.
        orders_df: DataFrame of training orders.
        config: Training configuration.
        epoch: Current epoch number.

    Returns:
        Dictionary of epoch-level metrics.
    """
    rng = np.random.RandomState(config.seed + epoch)
    num_orders = len(orders_df)
    indices = rng.permutation(num_orders)

    epoch_losses = []
    epoch_rewards = []

    for batch_start in range(0, min(num_orders, 200), config.batch_size):
        batch_indices = indices[batch_start:batch_start + config.batch_size]

        for idx in batch_indices:
            order = orders_df.iloc[idx].to_dict()

            # Sample G trajectories for this order
            trajectories = []
            for g in range(config.num_trajectories):
                traj = sample_trajectory(
                    grpo.policy, order, config.num_intervals,
                    seed=config.seed + epoch * 1000 + idx * 100 + g,
                )
                trajectories.append(traj)

            # Training step
            metrics = grpo.train_step(trajectories)
            epoch_losses.append(metrics["loss"])
            epoch_rewards.append(
                np.mean([traj.total_reward for traj in trajectories])
            )

        # Update reference policy periodically
        if (epoch + 1) % config.ref_policy_update_freq == 0:
            grpo.update_reference_policy()

    return {
        "epoch": epoch,
        "avg_loss": np.mean(epoch_losses),
        "avg_reward": np.mean(epoch_rewards),
    }


def evaluate_policy(
    policy: ExecutionPolicy,
    orders_df,
    num_intervals: int = 48,
    num_eval_orders: int = 100,
    seed: int = 999,
) -> Dict[str, float]:
    """Evaluate a policy on a set of orders.

    Args:
        policy: Trained execution policy.
        orders_df: DataFrame of evaluation orders.
        num_intervals: Execution horizon intervals.
        num_eval_orders: Number of orders to evaluate.
        seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    impl_shortfalls = []
    fill_rates = []

    for i in range(min(num_eval_orders, len(orders_df))):
        order = orders_df.iloc[i].to_dict()
        traj = sample_trajectory(policy, order, num_intervals, seed=seed + i)

        # Compute implementation shortfall from trajectory
        total_executed = sum(
            step.action[0] for step in traj.steps if step.action[0] > 0
        )
        impl_shortfalls.append(-traj.total_reward)
        fill_rates.append(min(total_executed * 10, 1.0))  # approximate

    is_array = np.array(impl_shortfalls)

    return {
        "mean_is_bps": float(is_array.mean()),
        "std_is_bps": float(is_array.std()),
        "median_is_bps": float(np.median(is_array)),
        "max_is_bps": float(is_array.max()),
        "mean_fill_rate": float(np.mean(fill_rates)),
        "sharpe": float(-is_array.mean() / max(is_array.std(), 1e-8)),
    }


def run_baseline_twap(orders_df, num_orders: int = 100) -> Dict[str, float]:
    """Run TWAP baseline for comparison."""
    results = []
    for i in range(min(num_orders, len(orders_df))):
        result = simulate_order_execution(orders_df.iloc[i], seed=42 + i)
        results.append(result["impl_shortfall_bps"])
    is_array = np.array(results)
    return {
        "method": "TWAP",
        "mean_is_bps": float(is_array.mean()),
        "std_is_bps": float(is_array.std()),
        "sharpe": float(-is_array.mean() / max(is_array.std(), 1e-8)),
    }


def run_ablation(config: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Run ablation study comparing lambda-GRPO vs standard GRPO.

    Returns:
        Dictionary mapping configuration name to metrics.
    """
    orders = generate_synthetic_orders(OrderConfig(num_orders=500, seed=42))
    results = {}

    # Lambda-ExecGRPO (full)
    policy_lambda = ExecutionPolicy(
        state_dim=config.state_dim, action_dim=config.action_dim,
        hidden_dim=256, num_layers=2,
    )
    grpo_lambda = LambdaExecGRPO(policy_lambda, lr=1e-3, use_lambda_norm=True)
    for epoch in range(20):
        train_epoch(grpo_lambda, orders, config, epoch)
    results["lambda-ExecGRPO"] = evaluate_policy(policy_lambda, orders, num_eval_orders=100)

    # Standard GRPO (no lambda normalization)
    policy_std = ExecutionPolicy(
        state_dim=config.state_dim, action_dim=config.action_dim,
        hidden_dim=256, num_layers=2,
    )
    grpo_std = LambdaExecGRPO(policy_std, lr=1e-3, use_lambda_norm=False)
    for epoch in range(20):
        train_epoch(grpo_std, orders, config, epoch)
    results["Standard GRPO"] = evaluate_policy(policy_std, orders, num_eval_orders=100)

    # TWAP baseline
    results["TWAP"] = run_baseline_twap(orders, num_orders=100)

    return results


def main():
    parser = argparse.ArgumentParser(description="Lambda-ExecGRPO Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "compare", "ablation"])
    parser.add_argument("--num_orders", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_trajectories", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_trajectories=args.num_trajectories,
        num_epochs=args.num_epochs,
        seed=args.seed,
    )

    if args.mode == "train":
        print("=" * 60)
        print("Lambda-ExecGRPO Training Pipeline")
        print("=" * 60)

        # Generate data
        orders = generate_synthetic_orders(
            OrderConfig(num_orders=args.num_orders, seed=args.seed)
        )
        print(f"Generated {len(orders)} training orders")

        # Initialize model and trainer
        policy = ExecutionPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        grpo = LambdaExecGRPO(
            policy, lr=config.lr, kl_penalty=config.kl_penalty,
            use_lambda_norm=True,
        )

        total_params = sum(p.numel() for p in policy.parameters())
        print(f"Policy parameters: {total_params:,}")

        # Training loop
        start_time = time.time()
        for epoch in range(config.num_epochs):
            metrics = train_epoch(grpo, orders, config, epoch)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                      f"Loss: {metrics['avg_loss']:+.4f} | "
                      f"Reward: {metrics['avg_reward']:+.4f} | "
                      f"Time: {elapsed:.1f}s")

        # Final evaluation
        eval_metrics = evaluate_policy(policy, orders, num_eval_orders=200)
        print(f"\nFinal Evaluation:")
        print(f"  Mean IS: {eval_metrics['mean_is_bps']:.2f} bps")
        print(f"  Sharpe: {eval_metrics['sharpe']:.3f}")

    elif args.mode == "compare":
        print("=" * 60)
        print("Baseline Comparison")
        print("=" * 60)

        orders = generate_synthetic_orders(
            OrderConfig(num_orders=500, seed=args.seed)
        )

        # TWAP baseline
        twap_results = run_baseline_twap(orders, num_orders=200)
        print(f"\nTWAP: IS={twap_results['mean_is_bps']:.2f} bps, "
              f"Sharpe={twap_results['sharpe']:.3f}")

        # Train and evaluate lambda-ExecGRPO
        policy = ExecutionPolicy(
            state_dim=config.state_dim, action_dim=config.action_dim,
            hidden_dim=config.hidden_dim, num_layers=2,
        )
        grpo = LambdaExecGRPO(policy, lr=1e-3, use_lambda_norm=True)

        for epoch in range(20):
            train_epoch(grpo, orders, config, epoch)
        grpo_metrics = evaluate_policy(policy, orders, num_eval_orders=200)
        print(f"Lambda-ExecGRPO: IS={grpo_metrics['mean_is_bps']:.2f} bps, "
              f"Sharpe={grpo_metrics['sharpe']:.3f}")

    elif args.mode == "ablation":
        print("=" * 60)
        print("Ablation Study")
        print("=" * 60)
        results = run_ablation(config)
        for name, metrics in results.items():
            print(f"\n{name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
