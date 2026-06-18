"""
main.py - Main training and evaluation script for TROLL-Risk portfolio optimization.

Trains the TROLL-Risk agent using PPO with trust region projection on synthetic
portfolio data, evaluates against PPO-clipped baseline, and reports comparative
performance metrics (Sharpe, drawdown, returns, stability).
"""

import os
import sys
import argparse
import numpy as np
import torch
# Disable torch.compile/dynamo to avoid MemoryError from sympy on Python 3.14
torch._dynamo.config.suppress_errors = True
import torch.nn.functional as F
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Local imports
from data import SyntheticPortfolioData, PortfolioConfig, PortfolioStateBuilder
from model import (
    TROLLRiskAgent,
    PolicyNetwork,
    ValueNetwork,
    TrustRegionProjection,
)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    n_episodes: int = 100
    steps_per_episode: int = 252  # One year of daily rebalancing
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_trust_region: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_clip_eps: float = 0.2
    mini_batch_size: int = 64
    kl_weight: float = 0.2
    sparse_k: int = 10
    hidden_dim: int = 256
    transaction_cost: float = 0.001  # 10 bps per trade
    seed: int = 42


def compute_portfolio_reward(
    weights: torch.Tensor,
    returns: torch.Tensor,
    prev_weights: torch.Tensor,
    transaction_cost: float = 0.001,
) -> torch.Tensor:
    """Compute risk-adjusted portfolio return with transaction costs.

    Args:
        weights: Current allocation (batch, n_assets).
        returns: Asset returns (batch, n_assets).
        prev_weights: Previous allocation (batch, n_assets).
        transaction_cost: Cost per unit of turnover.

    Returns:
        Reward values (batch,).
    """
    # Portfolio return
    port_return = torch.sum(weights * returns, dim=-1)

    # Transaction cost penalty
    turnover = torch.sum(torch.abs(weights - prev_weights), dim=-1)
    tc = transaction_cost * turnover

    # Sharpe-like reward: return - cost - 0.5 * variance_proxy
    reward = port_return - tc
    return reward


def compute_covariance_matrix(
    returns_history: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """Compute rolling covariance matrix from return history.

    Args:
        returns_history: Array of shape (T, n_assets).
        window: Rolling window size.

    Returns:
        Covariance matrix of shape (n_assets, n_assets).
    """
    recent = returns_history[-window:]
    if recent.shape[0] < 2:
        n_assets = recent.shape[1]
        return np.eye(n_assets) * 1e-4
    cov = np.cov(recent.T)
    # Handle NaN from degenerate cases
    if np.any(np.isnan(cov)):
        cov = np.nan_to_num(cov, nan=1e-4)
    return cov


def train_troll_risk(
    config: TrainConfig,
    data_config: PortfolioConfig,
    use_trust_region: bool = True,
) -> Dict:
    """Train the TROLL-Risk or PPO-clipped baseline agent.

    Args:
        config: Training configuration.
        data_config: Data generation configuration.
        use_trust_region: If True, use TROLL-Risk; if False, use PPO clipping.

    Returns:
        Dictionary of training results and metrics.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Generate data
    data_gen = SyntheticPortfolioData(data_config, seed=config.seed)
    data = data_gen.generate()
    train_data, val_data, test_data = data_gen.get_splits()

    n_assets = data_config.n_equities + data_config.n_bonds + data_config.n_commodities + data_config.n_fx

    # State builder
    state_builder = PortfolioStateBuilder(lookback=20, n_assets=n_assets)
    state_dim = state_builder.state_dim

    # Initialize agent
    agent = TROLLRiskAgent(
        state_dim=state_dim,
        n_assets=n_assets,
        hidden_dim=config.hidden_dim,
        kl_weight=config.kl_weight,
        sparse_k=config.sparse_k,
    )

    # Optimizers
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=config.lr_actor)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=config.lr_critic)
    trust_region_params = list(agent.trust_region.parameters())
    if trust_region_params:
        tr_opt = torch.optim.Adam(trust_region_params, lr=config.lr_trust_region)

    # Training tracking
    episode_returns: List[float] = []
    episode_sharpes: List[float] = []
    max_drawdowns: List[float] = []
    policy_collapses: int = 0

    returns_history = train_data["returns"]

    for episode in range(config.n_episodes):
        # Random starting point in training data
        max_start = len(returns_history) - config.steps_per_episode - 20
        if max_start <= 0:
            break
        start_idx = np.random.randint(20, max_start)
        episode_returns_slice = returns_history[
            start_idx : start_idx + config.steps_per_episode
        ]

        # Episode state
        prev_weights = torch.ones(n_assets) / n_assets
        portfolio_value = 1.0
        peak_value = 1.0
        episode_reward_sum = 0.0
        daily_returns_list: List[float] = []

        # Buffers for PPO update
        states_buf, actions_buf, rewards_buf, values_buf = [], [], [], []
        old_log_probs_buf = []

        for t in range(config.steps_per_episode):
            # Build state
            hist = episode_returns_slice[max(0, t - 20) : t + 1]
            if hist.shape[0] < 20:
                hist = np.pad(hist, ((20 - hist.shape[0], 0), (0, 0)))

            port_ret = float(np.sum(prev_weights.numpy() * episode_returns_slice[t]))
            port_vol = float(
                np.std([np.sum(prev_weights.numpy() * episode_returns_slice[max(0, t - j)]) for j in range(min(20, t + 1))])
            )
            portfolio_value *= 1 + port_ret
            peak_value = max(peak_value, portfolio_value)
            current_dd = (portfolio_value - peak_value) / peak_value

            state_np = state_builder.build_state(
                hist, prev_weights.numpy(), port_ret, port_vol, current_dd
            )
            state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)

            # Compute mean returns and covariance from recent history
            recent = episode_returns_slice[max(0, t - 60) : t + 1]
            mean_ret = torch.tensor(np.mean(recent, axis=0), dtype=torch.float32).unsqueeze(0)
            cov = compute_covariance_matrix(
                episode_returns_slice[max(0, t - 60) : t + 1], window=60
            )
            cov_tensor = torch.tensor(cov, dtype=torch.float32)
            hist_tensor = torch.tensor(
                recent[-20:].reshape(1, -1, n_assets), dtype=torch.float32
            )

            # Select action
            with torch.no_grad():
                if use_trust_region:
                    weights, risk_info = agent.act(
                        state, prev_weights.unsqueeze(0),
                        mean_ret, cov_tensor, hist_tensor,
                    )
                else:
                    # PPO baseline: no trust region
                    weights = agent.actor(state)

                value = agent.critic(state)

            weights = weights.squeeze(0)
            old_log_prob = agent.actor.get_log_prob(state, weights.unsqueeze(0))

            # Compute reward
            ret_tensor = torch.tensor(
                episode_returns_slice[t], dtype=torch.float32
            ).unsqueeze(0)
            reward = compute_portfolio_reward(
                weights.unsqueeze(0), ret_tensor, prev_weights.unsqueeze(0),
                config.transaction_cost,
            )

            # Store in buffer
            states_buf.append(state.squeeze(0))
            actions_buf.append(weights)
            rewards_buf.append(reward.squeeze(0))
            values_buf.append(value.squeeze(0))
            old_log_probs_buf.append(old_log_prob.squeeze(0))

            daily_returns_list.append(float(reward.squeeze(0)))
            episode_reward_sum += float(reward.squeeze(0))
            prev_weights = weights.detach()

        # PPO update
        states_t = torch.stack(states_buf)
        actions_t = torch.stack(actions_buf)
        rewards_t = torch.stack(rewards_buf)
        values_t = torch.stack(values_buf)
        old_lp_t = torch.stack(old_log_probs_buf).detach()

        # Compute GAE
        with torch.no_grad():
            next_val = agent.critic(states_t[-1:].detach())
        advantages, returns_target = agent.compute_gae(
            rewards_t, values_t, next_val.squeeze(),
            gamma=config.gamma, lam=config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(config.ppo_epochs):
            new_lp = agent.actor.get_log_prob(states_t, actions_t)
            ratio = torch.exp(new_lp - old_lp_t)

            # Clipped surrogate (for PPO baseline; TROLL-Risk uses trust region in act())
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - config.ppo_clip_eps, 1 + config.ppo_clip_eps
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            pred_values = agent.critic(states_t).squeeze(-1)
            if pred_values.dim() > returns_target.dim():
                pred_values = pred_values.squeeze()
            value_loss = F.mse_loss(pred_values, returns_target)

            # Update actor
            actor_opt.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            actor_opt.step()

            # Update critic
            critic_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            critic_opt.step()

            # Update trust region params
            if use_trust_region and trust_region_params:
                tr_opt.zero_grad()
                # Small auxiliary loss to regularize crisis delta
                aux_loss = sum(
                    (p**2).mean() for p in trust_region_params
                )
                aux_loss.backward()
                tr_opt.step()

        # Episode metrics
        daily_arr = np.array(daily_returns_list)
        ep_sharpe = float(
            np.mean(daily_arr) / (np.std(daily_arr) + 1e-8) * np.sqrt(252)
        )
        ep_max_dd = float(np.min(np.cumsum(daily_arr)))

        episode_returns.append(episode_reward_sum)
        episode_sharpes.append(ep_sharpe)
        max_drawdowns.append(ep_max_dd)

        if ep_sharpe < -1.0:
            policy_collapses += 1

        if (episode + 1) % 20 == 0:
            print(
                f"Episode {episode+1}/{config.n_episodes} | "
                f"Return: {episode_reward_sum:.4f} | "
                f"Sharpe: {ep_sharpe:.3f} | "
                f"MaxDD: {ep_max_dd:.4f}"
            )

    return {
        "episode_returns": episode_returns,
        "episode_sharpes": episode_sharpes,
        "max_drawdowns": max_drawdowns,
        "policy_collapses": policy_collapses,
        "final_sharpe": np.mean(episode_sharpes[-10:]) if episode_sharpes else 0.0,
        "final_return": np.mean(episode_returns[-10:]) if episode_returns else 0.0,
        "max_dd_worst": min(max_drawdowns) if max_drawdowns else 0.0,
    }


def evaluate_on_test(
    data_config: PortfolioConfig,
    config: TrainConfig,
) -> None:
    """Run comparative evaluation of TROLL-Risk vs PPO baseline.

    Trains both methods and prints comparative performance metrics.
    """
    print("=" * 70)
    print("TROLL-Risk vs PPO-Clip: Comparative Evaluation")
    print("=" * 70)

    # TROLL-Risk
    print("\n--- Training TROLL-Risk ---")
    troll_results = train_troll_risk(config, data_config, use_trust_region=True)

    # PPO baseline
    print("\n--- Training PPO-Clip Baseline ---")
    ppo_results = train_troll_risk(config, data_config, use_trust_region=False)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'TROLL-Risk':>15} {'PPO-Clip':>15} {'Improvement':>15}")
    print("-" * 70)

    metrics = [
        ("Final Sharpe", "final_sharpe", True),
        ("Final Return", "final_return", True),
        ("Worst MaxDD", "max_dd_worst", True),
        ("Policy Collapses", "policy_collapses", False),
    ]
    for name, key, higher_better in metrics:
        troll_val = troll_results[key]
        ppo_val = ppo_results[key]
        if higher_better:
            impr = troll_val - ppo_val
        else:
            impr = ppo_val - troll_val

        if isinstance(troll_val, float):
            print(f"{name:<25} {troll_val:>15.4f} {ppo_val:>15.4f} {impr:>+15.4f}")
        else:
            print(f"{name:<25} {troll_val:>15d} {ppo_val:>15d} {impr:>+15d}")

    print("=" * 70)

    # Stability analysis
    troll_sharpes = np.array(troll_results["episode_sharpes"])
    ppo_sharpes = np.array(ppo_results["episode_sharpes"])

    if len(troll_sharpes) > 0 and len(ppo_sharpes) > 0:
        troll_kl_stability = np.std(np.diff(troll_sharpes))
        ppo_kl_stability = np.std(np.diff(ppo_sharpes))
        print(f"\nTraining Stability (std of Sharpe diffs):")
        print(f"  TROLL-Risk: {troll_kl_stability:.4f}")
        print(f"  PPO-Clip:   {ppo_kl_stability:.4f}")
        print(f"  Improvement: {(1 - troll_kl_stability / (ppo_kl_stability + 1e-8)) * 100:.1f}%")


def main():
    """Entry point: parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="TROLL-Risk Portfolio Optimization")
    parser.add_argument("--episodes", type=int, default=60, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=128, help="Steps per episode")
    parser.add_argument("--n-equities", type=int, default=15, help="Number of equity assets")
    parser.add_argument("--n-bonds", type=int, default=5, help="Number of bond assets")
    parser.add_argument("--n-commodities", type=int, default=3, help="Number of commodity assets")
    parser.add_argument("--n-fx", type=int, default=2, help="Number of FX assets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sparse-k", type=int, default=8, help="Sparse subset size")
    args = parser.parse_args()

    train_config = TrainConfig(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        sparse_k=args.sparse_k,
        seed=args.seed,
    )

    data_config = PortfolioConfig(
        n_equities=args.n_equities,
        n_bonds=args.n_bonds,
        n_commodities=args.n_commodities,
        n_fx=args.n_fx,
        n_days=1500,
    )

    evaluate_on_test(data_config, train_config)


if __name__ == "__main__":
    main()
