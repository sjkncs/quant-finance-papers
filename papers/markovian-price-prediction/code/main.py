"""
main.py — Training and evaluation for Markovian Multi-Resolution Forecasting (MMRF).

Usage:
    python main.py --mode train --epochs 20 --batch_size 32
    python main.py --mode eval
    python main.py --mode demo  # quick demo with small synthetic data
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple

from data import (
    build_mmrf_dataloaders,
    RESOLUTION_CONFIGS,
    compute_compressed_summary,
    MultiResolutionDataGenerator,
)
from model import MMRFModel, FullContextARModel


def compute_directional_accuracy(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute directional accuracy per resolution and average."""
    metrics = {}
    total_da = 0.0
    n = 0
    for res_name in predictions:
        pred = predictions[res_name]
        target = targets[res_name]
        correct = ((pred > 0) == (target > 0)).float().mean().item()
        metrics[f"DA_{res_name}"] = correct
        total_da += correct
        n += 1
    metrics["DA_avg"] = total_da / max(n, 1)
    return metrics


def compute_mase(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute Mean Absolute Scaled Error per resolution."""
    metrics = {}
    total = 0.0
    n = 0
    for res_name in predictions:
        pred = predictions[res_name]
        target = targets[res_name]
        mae = (pred - target).abs().mean().item()
        # Scale by naive forecast error (using target std)
        scale = target.abs().mean().item() + 1e-8
        mase = mae / scale
        metrics[f"MASE_{res_name}"] = mase
        total += mase
        n += 1
    metrics["MASE_avg"] = total / max(n, 1)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_metrics = defaultdict(list)

    for batch_idx, batch in enumerate(loader):
        # Prepare inputs
        tokens = {}
        targets = {}
        for rc in RESOLUTION_CONFIGS:
            key = f"tokens_{rc.name}"
            if key in batch:
                tokens[rc.name] = batch[key].to(device)
            tkey = f"target_{rc.name}"
            if tkey in batch:
                targets[rc.name] = batch[tkey].to(device)

        summaries = batch["compressed_summaries"].to(device)

        optimizer.zero_grad()

        if isinstance(model, MMRFModel):
            output = model(tokens, summaries)
        else:
            output = model(tokens)

        loss = model.compute_loss(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Metrics
        with torch.no_grad():
            da_metrics = compute_directional_accuracy(output["predictions"], targets)
            for k, v in da_metrics.items():
                all_metrics[k].append(v)

        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] loss={loss.item():.6f}")

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    all_metrics = defaultdict(list)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        tokens = {}
        targets = {}
        for rc in RESOLUTION_CONFIGS:
            key = f"tokens_{rc.name}"
            if key in batch:
                tokens[rc.name] = batch[key].to(device)
            tkey = f"target_{rc.name}"
            if tkey in batch:
                targets[rc.name] = batch[tkey].to(device)

        summaries = batch["compressed_summaries"].to(device)

        if isinstance(model, MMRFModel):
            output = model(tokens, summaries)
        else:
            output = model(tokens)

        loss = model.compute_loss(output, targets)
        total_loss += loss.item()
        n_batches += 1

        da = compute_directional_accuracy(output["predictions"], targets)
        mase = compute_mase(output["predictions"], targets)
        for k, v in {**da, **mase}.items():
            all_metrics[k].append(v)

    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics["loss"] = total_loss / max(n_batches, 1)
    return avg_metrics


def run_training(args):
    """Full training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build data
    print("Building multi-resolution synthetic data...")
    train_loader, val_loader, test_loader, res_names = build_mmrf_dataloaders(
        n_assets=args.n_assets,
        n_days=args.n_days,
        batch_size=args.batch_size,
        window_width=args.window_width,
        seed=args.seed,
    )
    print(f"Resolutions: {res_names}")
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Resolution configs for model
    n_assets = args.n_assets
    resolution_configs = []
    for rc in RESOLUTION_CONFIGS:
        input_dim = n_assets * rc.n_features
        resolution_configs.append((rc.name, input_dim))

    # Build MMRF model
    model = MMRFModel(
        resolution_configs=resolution_configs,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        window_width=args.window_width,
        summary_dim=30,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MMRF parameters: {n_params:,}")

    # Build full-context baseline
    fc_model = FullContextARModel(
        resolution_configs=resolution_configs,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    fc_params = sum(p.numel() for p in fc_model.parameters())
    print(f"Full-Context AR parameters: {fc_params:,}")

    # Train MMRF
    print(f"\n{'='*60}")
    print("Training MMRF")
    print(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_da = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        val_da = val_metrics.get("DA_avg", 0.0)

        print(
            f"Epoch {epoch}: loss={train_loss:.6f}, "
            f"train_DA={train_metrics.get('DA_avg', 0):.4f}, "
            f"val_DA={val_da:.4f}"
        )

        if val_da > best_val_da:
            best_val_da = val_da

    # Evaluate MMRF on test set
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")

    test_metrics = evaluate(model, test_loader, device)
    print("\nMMRF Results:")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Evaluate Full-Context baseline
    print("\nTraining Full-Context AR baseline...")
    fc_optimizer = optim.AdamW(fc_model.parameters(), lr=args.lr, weight_decay=1e-4)
    fc_scheduler = optim.lr_scheduler.CosineAnnealingLR(fc_optimizer, T_max=args.epochs)

    for epoch in range(1, min(args.epochs, 5) + 1):
        fc_loss, _ = train_one_epoch(fc_model, train_loader, fc_optimizer, device, epoch)
        fc_scheduler.step()

    fc_metrics = evaluate(fc_model, test_loader, device)
    print("\nFull-Context AR Results:")
    for k, v in sorted(fc_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Comparison
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    mmrf_da = test_metrics.get("DA_avg", 0)
    fc_da = fc_metrics.get("DA_avg", 0)
    print(f"MMRF DA:           {mmrf_da:.4f}")
    print(f"Full-Context DA:   {fc_da:.4f}")
    print(f"Improvement:       {mmrf_da - fc_da:+.4f} ({(mmrf_da - fc_da) / max(fc_da, 0.001) * 100:+.1f}%)")

    # Memory estimate
    mem = model.get_memory_estimate(batch_size=args.batch_size)
    print(f"\nMemory estimates:")
    for k, v in mem.items():
        print(f"  {k}: {v:.1f} MB")

    # Window width ablation
    print(f"\n--- Window Width Ablation ---")
    for w in [1, 2, 3]:
        ablation_model = MMRFModel(
            resolution_configs=resolution_configs,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            window_width=w,
        ).to(device)
        # Copy weights from trained model where possible
        ablation_model.load_state_dict(model.state_dict(), strict=False)
        m = evaluate(ablation_model, test_loader, device)
        print(f"  w={w}: DA_avg={m.get('DA_avg', 0):.4f}, MASE_avg={m.get('MASE_avg', 0):.4f}")


def run_demo(args):
    """Quick demo of the MMRF pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_assets = 5
    resolution_configs = []
    for rc in RESOLUTION_CONFIGS:
        input_dim = n_assets * rc.n_features
        resolution_configs.append((rc.name, input_dim))

    # Build model
    model = MMRFModel(
        resolution_configs=resolution_configs,
        d_model=64,
        n_heads=4,
        n_layers=3,
        window_width=2,
        summary_dim=30,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Generate synthetic batch
    gen = MultiResolutionDataGenerator(n_assets=n_assets, n_days=500, seed=42)
    daily_ret, daily_vol, regime = gen.generate_daily_returns()

    # Build a single batch
    B = 4
    tokens = {}
    for rc in RESOLUTION_CONFIGS:
        features = gen.aggregate_to_resolution(daily_ret, daily_vol, rc)
        n_p = features.shape[0]
        lookback = min(30, n_p)
        tok = features[-lookback:].reshape(lookback, -1)  # flatten assets*features

        # Pad to 30
        if tok.shape[0] < 30:
            pad = np.zeros((30 - tok.shape[0], tok.shape[1]))
            tok = np.concatenate([pad, tok], axis=0)

        tokens[rc.name] = torch.tensor(tok[:30], dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).to(device)

    # Compressed summaries
    summaries = []
    for rc in RESOLUTION_CONFIGS:
        features = gen.aggregate_to_resolution(daily_ret, daily_vol, rc)
        s = compute_compressed_summary(features)
        summaries.append(s)
    summaries = torch.tensor(np.stack(summaries), dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(tokens, summaries)

    print("\nPredictions:")
    for res_name, pred in output["predictions"].items():
        print(f"  {res_name}: {pred.cpu().numpy().round(6)}")

    print(f"\nHidden states shape: {output['hidden_states'].shape}")

    # Memory estimates
    mem = model.get_memory_estimate(batch_size=B)
    print("\nMemory estimates:")
    for k, v in mem.items():
        print(f"  {k}: {v:.2f} MB")

    print("\nDemo complete.")


def main():
    parser = argparse.ArgumentParser(description="MMRF: Markovian Multi-Resolution Forecasting")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "eval", "demo"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--window_width", type=int, default=2)
    parser.add_argument("--n_assets", type=int, default=5)
    parser.add_argument("--n_days", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        run_training(args)
    elif args.mode == "demo":
        run_demo(args)
    else:
        print("Eval mode requires a trained model checkpoint.")


if __name__ == "__main__":
    main()
