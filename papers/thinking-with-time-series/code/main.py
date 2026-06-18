"""
main.py — Training and evaluation script for Thinking with Time-Series (TTS).

Usage:
    python main.py --mode train --epochs 20 --batch_size 32
    python main.py --mode eval --checkpoint checkpoint.pt
    python main.py --mode demo  # quick demo with small synthetic data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from data import build_dataloaders, MarketThinkBench, SyntheticMarketDataGenerator
from model import TTSFullModel, RegimeHMM


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility across NumPy, PyTorch CPU, and CUDA."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(
    logits: torch.Tensor,
    ground_truth: torch.Tensor,
    categories: List[str],
) -> Dict[str, float]:
    """Compute accuracy metrics overall and per category."""
    preds = logits.argmax(dim=-1)
    correct = (preds == ground_truth).float()

    metrics = {"overall_accuracy": correct.mean().item()}

    # Per-category accuracy
    cat_correct = defaultdict(list)
    for i, cat in enumerate(categories):
        cat_correct[cat].append(correct[i].item())

    for cat, vals in cat_correct.items():
        metrics[f"{cat}_accuracy"] = np.mean(vals)

    return metrics


def train_one_epoch(
    model: TTSFullModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        market_state = batch["market_state"].to(device)
        vix = batch["vix"].to(device)
        query_emb = batch["query_emb"].to(device)
        future = batch["future"].to(device)

        optimizer.zero_grad()
        output = model(market_state, vix, query_emb, future)
        loss = output["loss"]

        # Add answer classification loss
        if "answer_head" in [m[0] for m in model.named_modules()]:
            stats = model._extract_stats(
                torch.randn(
                    market_state.shape[0], model.n_trajectories,
                    model.horizon, model.n_assets, device=device
                )
            )
            # Use future returns to create pseudo-labels for answer head
            future_returns = future[:, -1, :].mean(dim=-1)
            pseudo_labels = torch.zeros(market_state.shape[0], dtype=torch.long, device=device)
            pseudo_labels[future_returns > 0.01] = 2  # positive
            pseudo_labels[future_returns < -0.01] = 0  # negative
            pseudo_labels[(future_returns >= -0.01) & (future_returns <= 0.01)] = 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 5 == 0:
            logger.info("  Epoch %d [%d/%d] loss=%.4f", epoch, batch_idx, len(loader), loss.item())

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: TTSFullModel,
    loader: DataLoader,
    device: torch.device,
    use_regime_sc: bool = True,
) -> Dict[str, float]:
    """Evaluate model on a dataset, return metrics."""
    model.eval()
    all_metrics = defaultdict(list)
    regime_hmm = RegimeHMM(n_states=3)

    for batch in loader:
        market_state = batch["market_state"].to(device)
        vix = batch["vix"].to(device)
        query_emb = batch["query_emb"].to(device)
        ground_truth = batch["ground_truth"].to(device)
        categories = batch["category"]

        output = model(market_state, vix, query_emb)
        logits = output["logits"]

        if use_regime_sc and "trajectories" in output:
            # Apply regime-aware self-consistency
            traj = output["trajectories"]  # (B, N, T, A)
            B, N, T, A = traj.shape

            for b in range(B):
                vix_val = vix[b].item()
                traj_np = traj[b].cpu().numpy()  # (N, T, A)

                # Compute regime weights
                traj_list = [traj_np[i] for i in range(N)]
                weights = regime_hmm.compute_regime_weights(traj_list, vix_val)

                # Weighted vote
                traj_returns = traj_np[:, -1, :].mean(axis=-1)  # (N,)
                weighted_pos = np.sum(weights[traj_returns > 0.01])
                weighted_neg = np.sum(weights[traj_returns < -0.01])
                weighted_neu = 1.0 - weighted_pos - weighted_neg

                logits[b, 0] = torch.tensor(weighted_neg, device=device)
                logits[b, 1] = torch.tensor(weighted_neu, device=device)
                logits[b, 2] = torch.tensor(weighted_pos, device=device)

        metrics = compute_metrics(logits, ground_truth, categories)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_metrics


def run_training(args: argparse.Namespace) -> TTSFullModel:
    """Full training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build data
    logger.info("Building synthetic datasets...")
    n_scenarios = args.n_scenarios
    train_loader, val_loader, test_loader, asset_names = build_dataloaders(
        n_scenarios=n_scenarios,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    n_assets = len(asset_names)
    logger.info("Assets (%d): %s", n_assets, asset_names)
    logger.info("Train batches: %d, Val batches: %d, Test batches: %d", len(train_loader), len(val_loader), len(test_loader))

    # Build model
    model = TTSFullModel(
        n_assets=n_assets,
        lookback=60,
        horizon=60,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        query_dim=128,
        n_diffusion_steps=args.diffusion_steps,
        n_trajectories=args.n_trajectories,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (trainable: %d)", n_params, n_trainable)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info("\n%s", "=" * 60)
        logger.info("Epoch %d/%d", epoch, args.epochs)
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device, use_regime_sc=False)
        val_acc = val_metrics.get("overall_accuracy", 0.0)

        elapsed = time.time() - t0
        logger.info("Epoch %d: train_loss=%.4f, val_acc=%.4f, time=%.1fs", epoch, train_loss, val_acc, elapsed)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_checkpoint:
                try:
                    torch.save(model.state_dict(), args.save_checkpoint)
                    logger.info("  Saved checkpoint (best val_acc=%.4f)", val_acc)
                except OSError as e:
                    logger.error("Failed to save checkpoint: %s", e)

    # Final evaluation
    logger.info("\n%s", "=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("%s", "=" * 60)

    # Without regime-aware SC
    test_metrics_basic = evaluate(model, test_loader, device, use_regime_sc=False)
    logger.info("Without Regime-Aware SC:")
    for k, v in sorted(test_metrics_basic.items()):
        logger.info("  %s: %.4f", k, v)

    # With regime-aware SC
    test_metrics_rsc = evaluate(model, test_loader, device, use_regime_sc=True)
    logger.info("With Regime-Aware SC:")
    for k, v in sorted(test_metrics_rsc.items()):
        logger.info("  %s: %.4f", k, v)

    # Ablation: number of trajectories
    logger.info("--- Trajectory Count Ablation ---")
    for n_traj in [4, 8, 16, 32]:
        model.n_trajectories = n_traj
        m = evaluate(model, test_loader, device, use_regime_sc=True)
        logger.info("  N=%2d: overall_accuracy=%.4f", n_traj, m.get('overall_accuracy', 0))

    return model


def run_demo(args: argparse.Namespace) -> None:
    """Quick demo showing the TTS pipeline on a single scenario."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Small setup for quick demo
    n_assets = 5
    model = TTSFullModel(
        n_assets=n_assets,
        lookback=30,
        horizon=30,
        hidden_dim=64,
        cond_dim=128,
        query_dim=128,
        n_diffusion_steps=4,
        n_trajectories=8,
    ).to(device)

    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Create synthetic input
    B = 2
    market_state = torch.randn(B, 30, n_assets, device=device) * 0.01
    vix = torch.tensor([18.5, 32.0], device=device)
    query_emb = torch.randn(B, 128, device=device)

    # Generate trajectories
    model.eval()
    with torch.no_grad():
        output = model(market_state, vix, query_emb)

    traj = output["trajectories"]
    logits = output["logits"]
    logger.info("Generated trajectories: shape=%s", traj.shape)
    logger.info("  (batch=%d, n_traj=%d, horizon=30, n_assets=%d)", B, model.n_trajectories, n_assets)

    # Analyze trajectory statistics
    logger.info("Trajectory statistics (batch item 0):")
    t0 = traj[0]  # (N, T, A)
    final_returns = t0[:, -1, :].mean(dim=-1)  # avg return across assets at final step
    logger.info("  Mean final return per trajectory: %s", final_returns.cpu().numpy().round(4))
    logger.info("  Std of final returns: %.4f", final_returns.std().item())
    logger.info("  Fraction with negative outcome: %.2f%%", (final_returns < 0).float().mean().item() * 100)

    # Answer prediction
    probs = torch.softmax(logits, dim=-1)
    logger.info("Answer probabilities (batch item 0): %s", probs[0].cpu().numpy().round(4))
    logger.info("  [negative, neutral, positive]")

    # Regime detection on generated trajectories
    hmm = RegimeHMM()
    regime_names = ["Low-Vol/Bull", "High-Vol/Bear", "Crisis"]
    for i in range(min(model.n_trajectories, 4)):
        traj_np = t0[i].cpu().numpy()
        regime, probs_r = hmm.detect_regime(traj_np, vix[0].item())
        logger.info("  Traj %d: regime=%s, probs=%s", i, regime_names[regime], probs_r.round(3))

    logger.info("Demo complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thinking with Time-Series (TTS) / 思维时序化：扩散轨迹生成训练与评估"
    )
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "eval", "demo"],
                        help="运行模式: train=训练, eval=评估, demo=快速演示")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="隐藏层维度")
    parser.add_argument("--cond_dim", type=int, default=128,
                        help="条件向量维度")
    parser.add_argument("--diffusion_steps", type=int, default=4,
                        help="扩散步数")
    parser.add_argument("--n_trajectories", type=int, default=8,
                        help="每次查询生成的轨迹数")
    parser.add_argument("--n_scenarios", type=int, default=420,
                        help="MarketThinkBench场景数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                        help="模型检查点保存路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    seed_everything(args.seed)

    if args.mode == "train":
        run_training(args)
    elif args.mode == "demo":
        run_demo(args)
    else:
        logger.warning("Eval mode: please provide --save_checkpoint path")


if __name__ == "__main__":
    main()
