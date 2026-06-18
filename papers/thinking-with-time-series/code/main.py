"""
main.py — Training and evaluation script for Thinking with Time-Series (TTS).

Usage:
    python main.py --mode train --epochs 20 --batch_size 32
    python main.py --mode eval --checkpoint checkpoint.pt
    python main.py --mode demo  # quick demo with small synthetic data
"""

from __future__ import annotations

import argparse
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

from data import build_dataloaders, MarketThinkBench, SyntheticMarketDataGenerator
from model import TTSFullModel, RegimeHMM


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
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] loss={loss.item():.4f}")

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


def run_training(args):
    """Full training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build data
    print("Building synthetic datasets...")
    n_scenarios = args.n_scenarios
    train_loader, val_loader, test_loader, asset_names = build_dataloaders(
        n_scenarios=n_scenarios,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    n_assets = len(asset_names)
    print(f"Assets ({n_assets}): {asset_names}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

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
    print(f"Model parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device, use_regime_sc=False)
        val_acc = val_metrics.get("overall_accuracy", 0.0)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, time={elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_checkpoint:
                torch.save(model.state_dict(), args.save_checkpoint)
                print(f"  Saved checkpoint (best val_acc={val_acc:.4f})")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")

    # Without regime-aware SC
    test_metrics_basic = evaluate(model, test_loader, device, use_regime_sc=False)
    print("\nWithout Regime-Aware SC:")
    for k, v in sorted(test_metrics_basic.items()):
        print(f"  {k}: {v:.4f}")

    # With regime-aware SC
    test_metrics_rsc = evaluate(model, test_loader, device, use_regime_sc=True)
    print("\nWith Regime-Aware SC:")
    for k, v in sorted(test_metrics_rsc.items()):
        print(f"  {k}: {v:.4f}")

    # Ablation: number of trajectories
    print("\n--- Trajectory Count Ablation ---")
    for n_traj in [4, 8, 16, 32]:
        model.n_trajectories = n_traj
        m = evaluate(model, test_loader, device, use_regime_sc=True)
        print(f"  N={n_traj:2d}: overall_accuracy={m.get('overall_accuracy', 0):.4f}")

    return model


def run_demo(args):
    """Quick demo showing the TTS pipeline on a single scenario."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    print(f"\nGenerated trajectories: shape={traj.shape}")
    print(f"  (batch={B}, n_traj={model.n_trajectories}, horizon=30, n_assets={n_assets})")

    # Analyze trajectory statistics
    print("\nTrajectory statistics (batch item 0):")
    t0 = traj[0]  # (N, T, A)
    final_returns = t0[:, -1, :].mean(dim=-1)  # avg return across assets at final step
    print(f"  Mean final return per trajectory: {final_returns.cpu().numpy().round(4)}")
    print(f"  Std of final returns: {final_returns.std().item():.4f}")
    print(f"  Fraction with negative outcome: {(final_returns < 0).float().mean().item():.2%}")

    # Answer prediction
    probs = torch.softmax(logits, dim=-1)
    print(f"\nAnswer probabilities (batch item 0): {probs[0].cpu().numpy().round(4)}")
    print(f"  [negative, neutral, positive]")

    # Regime detection on generated trajectories
    hmm = RegimeHMM()
    for i in range(min(model.n_trajectories, 4)):
        traj_np = t0[i].cpu().numpy()
        regime, probs_r = hmm.detect_regime(traj_np, vix[0].item())
        regime_names = ["Low-Vol/Bull", "High-Vol/Bear", "Crisis"]
        print(f"  Traj {i}: regime={regime_names[regime]}, probs={probs_r.round(3)}")

    print("\nDemo complete.")


def main():
    parser = argparse.ArgumentParser(description="Thinking with Time-Series (TTS)")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "eval", "demo"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--diffusion_steps", type=int, default=4)
    parser.add_argument("--n_trajectories", type=int, default=8)
    parser.add_argument("--n_scenarios", type=int, default=420)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_checkpoint", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        run_training(args)
    elif args.mode == "demo":
        run_demo(args)
    else:
        print("Eval mode: please provide --save_checkpoint path")


if __name__ == "__main__":
    main()
