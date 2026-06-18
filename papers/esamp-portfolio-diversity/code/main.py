"""
main.py - PortESamp Training and Evaluation Pipeline
=====================================================
Main script for training the Strategy Distiller and running the
PortESamp novelty-guided portfolio generation framework.

Usage:
    python main.py --mode train --num_epochs 50
    python main.py --mode eval --beta 0.25
    python main.py --mode compare  # compare all methods
    python main.py --mode ablation  # ablation study

Key features:
- Distiller training with replay buffer
- Novelty-guided portfolio generation
- Baseline comparisons (MV, BL, Risk Parity, Dropout, FIRE)
- Diversity and performance metrics computation
- Hyperparameter sweep over beta
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

from data import (
    AssetUniverseConfig,
    AssetUniverseGenerator,
    compute_portfolio_metrics,
    compute_weight_entropy,
    compute_strategy_diversity,
    generate_mean_variance_portfolio,
    generate_risk_parity_portfolio,
)
from model import (
    PortfolioModel,
    StrategyDistiller,
    PortESampSampler,
    DistillerReplayBuffer,
    PortfolioProposal,
)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    num_assets: int = 50
    input_dim: int = 50
    feature_dim: int = 64
    deep_dim: int = 128
    distiller_hidden: int = 256
    num_epochs: int = 30
    distiller_lr: float = 1e-3
    batch_size: int = 64
    num_proposals: int = 64
    num_select: int = 8
    beta: float = 0.25
    noise_scale: float = 0.15
    num_eval_windows: int = 20
    eval_window_size: int = 60  # days
    seed: int = 42


def prepare_market_features(
    returns_df,
    day_idx: int,
    window: int = 20,
    num_assets: int = 50,
) -> torch.Tensor:
    """Prepare market feature vector for a specific day.

    Args:
        returns_df: DataFrame of asset returns.
        day_idx: Current day index.
        window: Lookback window.
        num_assets: Number of assets.

    Returns:
        Feature tensor of shape (input_dim,).
    """
    if day_idx < window:
        return torch.zeros(num_assets)

    recent = returns_df.iloc[max(0, day_idx - window):day_idx].values[:, :num_assets]

    features = []
    # Rolling statistics per asset
    features.append(recent.mean(axis=0) * 252)          # annualized return
    features.append(recent.std(axis=0) * np.sqrt(252))  # annualized vol
    features.append(recent[-5:].mean(axis=0) * 252)     # short-term momentum
    features.append(recent[-5:].std(axis=0) * np.sqrt(252))  # short-term vol
    # Cross-sectional features
    features.append(np.array([recent.mean(axis=0).mean() * 252]))  # market return
    features.append(np.array([recent.std(axis=0).mean() * np.sqrt(252)]))  # market vol
    # Correlation matrix summary (top eigenvalues)
    if recent.shape[0] > 1 and recent.shape[1] > 1:
        corr = np.corrcoef(recent.T)
        eigenvalues = np.linalg.eigvalsh(corr)
        features.append(eigenvalues[:max(1, num_assets - len(features) * recent.shape[1] // recent.shape[0])])

    feature_vec = np.concatenate(features)
    # Pad or truncate to input_dim
    if len(feature_vec) < num_assets:
        feature_vec = np.pad(feature_vec, (0, num_assets - len(feature_vec)))
    else:
        feature_vec = feature_vec[:num_assets]

    return torch.tensor(feature_vec, dtype=torch.float32)


def train_portfolio_model(
    returns_df,
    config: ExperimentConfig,
) -> PortfolioModel:
    """Train the portfolio model with supervised pre-training.

    Trains the model to produce mean-variance-like portfolios using
    historical data as supervision.

    Args:
        returns_df: DataFrame of asset returns.
        config: Experiment configuration.

    Returns:
        Trained PortfolioModel.
    """
    model = PortfolioModel(
        input_dim=config.input_dim,
        num_assets=config.num_assets,
        feature_dim=config.feature_dim,
        deep_dim=config.deep_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate target portfolios (mean-variance with varying risk aversion)
    num_samples = min(200, len(returns_df) - 60)

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, config.batch_size):
            batch_losses = []
            for j in range(i, min(i + config.batch_size, num_samples)):
                day = 60 + j
                if day >= len(returns_df):
                    break

                features = prepare_market_features(
                    returns_df, day, num_assets=config.num_assets
                )
                # Target: mean-variance portfolio from recent window
                recent = returns_df.iloc[max(0, day - 60):day].values[:, :config.num_assets]
                import pandas as pd
                recent_df = pd.DataFrame(recent)
                lambda_risk = 0.5 + np.random.exponential(1.0)
                target_weights = generate_mean_variance_portfolio(
                    recent_df, lambda_risk=lambda_risk
                )
                target = torch.tensor(target_weights, dtype=torch.float32)

                weights, _, _ = model(features.unsqueeze(0))
                loss = F.mse_loss(weights.squeeze(), target)
                batch_losses.append(loss)

            if batch_losses:
                avg_loss = torch.stack(batch_losses).mean()
                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()
                total_loss += avg_loss.item()
                num_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = total_loss / max(num_batches, 1)
            print(f"  Epoch {epoch+1:3d}/{config.num_epochs} | Loss: {avg:.6f}")

    return model


def train_distiller(
    model: PortfolioModel,
    returns_df,
    config: ExperimentConfig,
) -> StrategyDistiller:
    """Train the Strategy Distiller on representations from the portfolio model.

    Args:
        model: Trained portfolio model.
        returns_df: DataFrame of asset returns.
        config: Experiment configuration.

    Returns:
        Trained StrategyDistiller.
    """
    distiller = StrategyDistiller(
        input_dim=config.feature_dim,
        output_dim=config.deep_dim,
        hidden_dim=config.distiller_hidden,
    )
    optimizer = torch.optim.Adam(distiller.parameters(), lr=config.distiller_lr)

    replay_buffer = DistillerReplayBuffer(max_size=5000)

    model.eval()
    num_days = min(300, len(returns_df) - 60)

    # Collect representations
    print("  Collecting representations...")
    for day in range(60, 60 + num_days):
        features = prepare_market_features(returns_df, day, num_assets=config.num_assets)
        with torch.no_grad():
            _, shallow, deep = model(features.unsqueeze(0))
        replay_buffer.add(shallow.squeeze(), deep.squeeze())

    # Train distiller
    print(f"  Training distiller on {len(replay_buffer)} samples...")
    for epoch in range(30):
        total_loss = 0.0
        for _ in range(10):
            shallow_batch, deep_batch = replay_buffer.sample_batch(config.batch_size)
            predicted = distiller(shallow_batch)
            loss = F.mse_loss(predicted, deep_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Distiller epoch {epoch+1}: loss={total_loss/10:.6f}")

    return distiller


def evaluate_portesamp(
    model: PortfolioModel,
    distiller: StrategyDistiller,
    returns_df,
    config: ExperimentConfig,
) -> Dict[str, float]:
    """Evaluate PortESamp on out-of-sample data.

    Args:
        model: Trained portfolio model.
        distiller: Trained strategy distiller.
        returns_df: DataFrame of returns.
        config: Experiment configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    sampler = PortESampSampler(
        model, distiller,
        beta=config.beta,
        num_proposals=config.num_proposals,
        num_select=config.num_select,
        noise_scale=config.noise_scale,
    )

    all_portfolios = []
    all_sharpes = []
    all_max_dds = []
    all_entropies = []

    test_start = len(returns_df) - config.num_eval_windows * config.eval_window_size
    test_start = max(test_start, 60)

    for w in range(config.num_eval_windows):
        day = test_start + w * config.eval_window_size
        if day + config.eval_window_size >= len(returns_df):
            break

        features = prepare_market_features(returns_df, day, num_assets=config.num_assets)
        proposals = sampler.generate_diverse_portfolios(features)

        # Evaluate each proposal on the next window
        window_returns = returns_df.iloc[day:day + config.eval_window_size]
        for prop in proposals:
            all_portfolios.append(prop.weights)
            metrics = compute_portfolio_metrics(window_returns, prop.weights[:config.num_assets])
            all_sharpes.append(metrics["sharpe"])
            all_max_dds.append(metrics["max_drawdown"])
            all_entropies.append(compute_weight_entropy(prop.weights))

    diversity = compute_strategy_diversity(all_portfolios)

    return {
        "diversity_H": diversity,
        "avg_sharpe": float(np.mean(all_sharpes)),
        "avg_max_dd": float(np.mean(all_max_dds)),
        "avg_entropy": float(np.mean(all_entropies)),
        "num_portfolios": len(all_portfolios),
        "calmar": float(np.mean(all_sharpes) / max(abs(np.mean(all_max_dds)), 1e-8)),
    }


def evaluate_baselines(
    returns_df,
    config: ExperimentConfig,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all baseline methods.

    Returns:
        Dictionary mapping method name to metrics.
    """
    results = {}
    test_start = len(returns_df) - config.num_eval_windows * config.eval_window_size
    test_start = max(test_start, 60)

    for method_name, gen_func in [
        ("Mean-Variance", lambda ret: generate_mean_variance_portfolio(ret, lambda_risk=1.0)),
        ("Risk Parity", generate_risk_parity_portfolio),
    ]:
        portfolios = []
        sharpes = []
        max_dds = []
        entropies = []

        for w in range(config.num_eval_windows):
            day = test_start + w * config.eval_window_size
            if day + config.eval_window_size >= len(returns_df):
                break
            recent = returns_df.iloc[max(0, day - 60):day].values[:, :config.num_assets]
            import pandas as pd
            recent_df = pd.DataFrame(recent)
            weights = gen_func(recent_df)
            portfolios.append(weights)

            window_returns = returns_df.iloc[day:day + config.eval_window_size]
            metrics = compute_portfolio_metrics(window_returns, weights)
            sharpes.append(metrics["sharpe"])
            max_dds.append(metrics["max_drawdown"])
            entropies.append(compute_weight_entropy(weights))

        results[method_name] = {
            "diversity_H": compute_strategy_diversity(portfolios),
            "avg_sharpe": float(np.mean(sharpes)),
            "avg_max_dd": float(np.mean(max_dds)),
            "avg_entropy": float(np.mean(entropies)),
            "calmar": float(np.mean(sharpes) / max(abs(np.mean(max_dds)), 1e-8)),
        }

    # Dropout ensemble (simulate by adding noise to MV weights)
    portfolios = []
    sharpes = []
    max_dds = []
    entropies = []
    for w in range(config.num_eval_windows):
        day = test_start + w * config.eval_window_size
        if day + config.eval_window_size >= len(returns_df):
            break
        recent = returns_df.iloc[max(0, day - 60):day].values[:, :config.num_assets]
        import pandas as pd
        recent_df = pd.DataFrame(recent)
        base_weights = generate_mean_variance_portfolio(recent_df, lambda_risk=1.0)

        for _ in range(config.num_select):
            noise = np.random.randn(config.num_assets) * 0.02
            noisy_weights = np.maximum(base_weights + noise, 0)
            noisy_weights = noisy_weights / noisy_weights.sum()
            portfolios.append(noisy_weights)

            window_returns = returns_df.iloc[day:day + config.eval_window_size]
            metrics = compute_portfolio_metrics(window_returns, noisy_weights)
            sharpes.append(metrics["sharpe"])
            max_dds.append(metrics["max_drawdown"])
            entropies.append(compute_weight_entropy(noisy_weights))

    results["Dropout Ensemble"] = {
        "diversity_H": compute_strategy_diversity(portfolios),
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_max_dd": float(np.mean(max_dds)),
        "avg_entropy": float(np.mean(entropies)),
        "calmar": float(np.mean(sharpes) / max(abs(np.mean(max_dds)), 1e-8)),
    }

    return results


def run_beta_sweep(
    model: PortfolioModel,
    distiller: StrategyDistiller,
    returns_df,
    config: ExperimentConfig,
    betas: List[float] = None,
) -> Dict[float, Dict[str, float]]:
    """Run hyperparameter sweep over exploration strength beta.

    Args:
        model: Trained portfolio model.
        distiller: Trained distiller.
        returns_df: Returns data.
        config: Experiment config.
        betas: List of beta values to try.

    Returns:
        Dictionary mapping beta to metrics.
    """
    if betas is None:
        betas = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50]

    results = {}
    for beta in betas:
        config.beta = beta
        metrics = evaluate_portesamp(model, distiller, returns_df, config)
        results[beta] = metrics
        print(f"  beta={beta:.2f}: H={metrics['diversity_H']:.3f}, "
              f"Sharpe={metrics['avg_sharpe']:.3f}, "
              f"MaxDD={metrics['avg_max_dd']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PortESamp Pipeline")
    parser.add_argument("--mode", type=str, default="compare",
                        choices=["train", "eval", "compare", "ablation", "beta_sweep"])
    parser.add_argument("--num_assets", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = ExperimentConfig(
        num_assets=args.num_assets,
        input_dim=args.num_assets,
        num_epochs=args.num_epochs,
        beta=args.beta,
        seed=args.seed,
    )

    # Generate data
    print("Generating synthetic asset universe...")
    universe_config = AssetUniverseConfig(
        num_assets=args.num_assets, num_days=1008, seed=args.seed
    )
    gen = AssetUniverseGenerator(universe_config)
    returns_df = gen.generate_returns()
    print(f"Generated {returns_df.shape[0]} days of returns for {args.num_assets} assets")

    if args.mode == "train":
        print("\n" + "=" * 60)
        print("Training Portfolio Model and Strategy Distiller")
        print("=" * 60)

        print("\nTraining portfolio model...")
        model = train_portfolio_model(returns_df, config)

        print("\nTraining strategy distiller...")
        distiller = train_distiller(model, returns_df, config)

        print("\nEvaluating PortESamp...")
        metrics = evaluate_portesamp(model, distiller, returns_df, config)
        print(f"Diversity (H): {metrics['diversity_H']:.3f}")
        print(f"Avg Sharpe: {metrics['avg_sharpe']:.3f}")
        print(f"Avg MaxDD: {metrics['avg_max_dd']:.1%}")

    elif args.mode == "compare":
        print("\n" + "=" * 60)
        print("Comparing All Methods")
        print("=" * 60)

        print("\nTraining portfolio model...")
        model = train_portfolio_model(returns_df, config)

        print("\nTraining strategy distiller...")
        distiller = train_distiller(model, returns_df, config)

        print("\nEvaluating PortESamp...")
        portesamp_metrics = evaluate_portesamp(model, distiller, returns_df, config)

        print("\nEvaluating baselines...")
        baseline_metrics = evaluate_baselines(returns_df, config)

        # Print comparison table
        print("\n" + "=" * 80)
        print(f"{'Method':<20} {'Diversity (H)':<15} {'Sharpe':<10} {'MaxDD':<12} {'Calmar':<10}")
        print("-" * 80)

        for name, m in baseline_metrics.items():
            print(f"{name:<20} {m['diversity_H']:<15.3f} {m['avg_sharpe']:<10.3f} "
                  f"{m['avg_max_dd']:<12.1%} {m['calmar']:<10.3f}")

        print(f"{'PortESamp':<20} {portesamp_metrics['diversity_H']:<15.3f} "
              f"{portesamp_metrics['avg_sharpe']:<10.3f} "
              f"{portesamp_metrics['avg_max_dd']:<12.1%} "
              f"{portesamp_metrics['calmar']:<10.3f}")

    elif args.mode == "beta_sweep":
        print("\n" + "=" * 60)
        print("Beta Hyperparameter Sweep")
        print("=" * 60)

        print("\nTraining models...")
        model = train_portfolio_model(returns_df, config)
        distiller = train_distiller(model, returns_df, config)

        print("\nSweeping beta values...")
        results = run_beta_sweep(model, distiller, returns_df, config)

    elif args.mode == "ablation":
        print("\n" + "=" * 60)
        print("Ablation Study")
        print("=" * 60)

        model = train_portfolio_model(returns_df, config)
        distiller = train_distiller(model, returns_df, config)

        # Full PortESamp
        config.beta = 0.25
        full = evaluate_portesamp(model, distiller, returns_df, config)
        print(f"\nFull PortESamp: H={full['diversity_H']:.3f}, Sharpe={full['avg_sharpe']:.3f}")

        # No distiller (beta=0)
        config.beta = 0.0
        no_distiller = evaluate_portesamp(model, distiller, returns_df, config)
        print(f"No novelty signal: H={no_distiller['diversity_H']:.3f}, "
              f"Sharpe={no_distiller['avg_sharpe']:.3f}")

        # High exploration
        config.beta = 0.50
        high_beta = evaluate_portesamp(model, distiller, returns_df, config)
        print(f"High beta (0.50): H={high_beta['diversity_H']:.3f}, "
              f"Sharpe={high_beta['avg_sharpe']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
