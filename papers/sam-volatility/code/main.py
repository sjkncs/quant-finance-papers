"""
main.py — Training and evaluation for SAM-Vol volatility surface reconstruction.

Usage:
    python main.py --mode train --epochs 30 --batch_size 32
    python main.py --mode eval
    python main.py --mode demo  # quick demo with small synthetic data
"""

from __future__ import annotations

import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from data import (
    build_samvol_dataloaders,
    SAMVolDataset,
    generate_heston_surface,
    generate_sabr_surface,
    sample_sparse_quotes,
    black_scholes_call,
)
from model import SAMVolModel


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility across NumPy, PyTorch CPU, and CUDA."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute root mean squared error of implied volatility."""
    return torch.sqrt(((predicted - target) ** 2).mean()).item()


def compute_mae(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean absolute error."""
    return (predicted - target).abs().mean().item()


def train_one_epoch(
    model: SAMVolModel,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_metrics = defaultdict(float)
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        point_cloud = batch["point_cloud"].to(device)
        mask = batch["mask"].to(device)
        eval_k = batch["eval_k"].to(device)
        eval_tau = batch["eval_tau"].to(device)
        gt_surface = batch["gt_surface"].to(device)

        optimizer.zero_grad()
        output = model(point_cloud, mask, eval_k, eval_tau)
        losses = model.compute_loss(output, gt_surface)
        loss = losses["total_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_metrics[k] += v.item()
        n_batches += 1

        if batch_idx % 5 == 0:
            logger.info(
                "  Epoch %d [%d/%d] loss=%.6f recon=%.6f arb=%.6f",
                epoch, batch_idx, len(loader),
                losses['total_loss'].item(),
                losses['recon_loss'].item(),
                losses['arb_loss'].item(),
            )

    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: SAMVolModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    all_metrics = defaultdict(list)

    for batch in loader:
        point_cloud = batch["point_cloud"].to(device)
        mask = batch["mask"].to(device)
        eval_k = batch["eval_k"].to(device)
        eval_tau = batch["eval_tau"].to(device)
        gt_surface = batch["gt_surface"].to(device)

        output = model(point_cloud, mask, eval_k, eval_tau)
        iv_surface = output["iv_surface"]

        rmse = compute_rmse(iv_surface, gt_surface)
        mae = compute_mae(iv_surface, gt_surface)

        # Arbitrage violations
        arb_viol = model.count_arbitrage_violations(
            iv_surface, iv_surface.shape[1], iv_surface.shape[2]
        )

        all_metrics["rmse"].append(rmse)
        all_metrics["mae"].append(mae)
        all_metrics["arb_violations"].append(arb_viol)

    avg = {k: np.mean(v) for k, v in all_metrics.items()}
    avg["rmse_std"] = np.std(all_metrics["rmse"])
    return avg


def run_training(args: argparse.Namespace) -> SAMVolModel:
    """Full training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build data
    logger.info("Building synthetic volatility surface data...")
    train_loader, val_loader, test_loader = build_samvol_dataloaders(
        n_surfaces=args.n_surfaces,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    logger.info("Train: %d batches, Val: %d, Test: %d",
                len(train_loader), len(val_loader), len(test_loader))

    # Build model
    model = SAMVolModel(
        input_dim=7,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_heads=args.n_heads,
        hidden_dim=args.hidden_dim,
        n_surface_layers=args.n_surface_layers,
        bandwidth=args.bandwidth,
        arb_weight=args.arb_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("SAM-Vol parameters: %d", n_params)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        logger.info("=" * 60)
        logger.info("Epoch %d/%d", epoch, args.epochs)
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        val_rmse = val_metrics["rmse"]

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d: train_loss=%.6f, val_rmse=%.4f, val_arb=%.4f, time=%.1fs",
            epoch, train_metrics['total_loss'], val_rmse,
            val_metrics['arb_violations'], elapsed,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            logger.info("  New best val RMSE: %.4f", val_rmse)

    # Final test evaluation
    logger.info("=" * 60)
    logger.info("Test Set Evaluation")
    logger.info("=" * 60)

    test_metrics = evaluate(model, test_loader, device)
    logger.info("SAM-Vol Test Results:")
    logger.info("  RMSE:              %.4f (+/- %.4f)", test_metrics['rmse'], test_metrics['rmse_std'])
    logger.info("  MAE:               %.4f", test_metrics['mae'])
    logger.info("  Arb Violations:    %.4f%%", test_metrics['arb_violations'] * 100)

    # Baseline comparison: SVI-like interpolation
    logger.info("--- Baseline Comparison (Linear Interpolation) ---")
    baseline_metrics = evaluate_baseline(test_loader, device)
    logger.info("  Linear Interp RMSE: %.4f", baseline_metrics['rmse'])
    logger.info("  Linear Interp MAE:  %.4f", baseline_metrics['mae'])
    logger.info("  Improvement:        %.1f%%",
                (baseline_metrics['rmse'] - test_metrics['rmse']) / baseline_metrics['rmse'] * 100)

    # Ablation: number of observed quotes
    logger.info("--- Quote Density Ablation ---")
    for n_quotes in [10, 20, 40, 60]:
        ablation_ds = SAMVolDataset(
            n_surfaces=50,
            n_quotes_range=(n_quotes, n_quotes + 5),
            seed=99,
        )
        ablation_loader = torch.utils.data.DataLoader(
            ablation_ds, batch_size=16, shuffle=False
        )
        m = evaluate(model, ablation_loader, device)
        logger.info("  N_quotes~%d: RMSE=%.4f, Arb=%.4f%%", n_quotes, m['rmse'], m['arb_violations'] * 100)

    return model


@torch.no_grad()
def evaluate_baseline(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Simple baseline: nearest-neighbor interpolation from observed quotes."""
    all_rmse = []
    all_mae = []

    for batch in loader:
        point_cloud = batch["point_cloud"].to(device)
        mask = batch["mask"].to(device)
        eval_k = batch["eval_k"].to(device)
        eval_tau = batch["eval_tau"].to(device)
        gt_surface = batch["gt_surface"].to(device)

        B, N, _ = point_cloud.shape
        n_k = eval_k.shape[1]
        n_tau = eval_tau.shape[1]

        # For each sample, do nearest-neighbor interpolation
        for b in range(B):
            pc = point_cloud[b]  # (N, 7)
            m = mask[b]  # (N,)
            valid = pc[m > 0.5]  # valid quotes

            if len(valid) == 0:
                all_rmse.append(0.2)  # dummy
                all_mae.append(0.2)
                continue

            # Nearest-neighbor for each grid point
            pred_surface = torch.zeros(n_k, n_tau, device=device)
            for i in range(n_k):
                for j in range(n_tau):
                    k_q = eval_k[b, i]
                    tau_q = eval_tau[b, j]
                    dists = (valid[:, 0] - k_q) ** 2 + (valid[:, 1] - tau_q) ** 2
                    nearest_idx = dists.argmin()
                    pred_surface[i, j] = valid[nearest_idx, 2]  # IV

            rmse = torch.sqrt(((pred_surface - gt_surface[b]) ** 2).mean()).item()
            mae = (pred_surface - gt_surface[b]).abs().mean().item()
            all_rmse.append(rmse)
            all_mae.append(mae)

    return {"rmse": np.mean(all_rmse), "mae": np.mean(all_mae)}


def run_demo(args: argparse.Namespace) -> None:
    """Quick demo showing SAM-Vol surface reconstruction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Small model for demo
    model = SAMVolModel(
        input_dim=7,
        d_model=64,
        n_encoder_layers=2,
        n_heads=4,
        hidden_dim=128,
        n_surface_layers=3,
        bandwidth=0.15,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    # Generate one surface
    rng = np.random.default_rng(42)
    k_grid, tau_grid, iv_surface = generate_heston_surface(
        n_strikes=20, n_maturities=10, S=100.0, rng=rng
    )
    quotes = sample_sparse_quotes(k_grid, tau_grid, iv_surface, n_quotes=30, S=100.0, rng=rng)

    logger.info("True surface shape: %s", iv_surface.shape)
    logger.info("Number of observed quotes: %d", len(quotes))
    logger.info("IV range: [%.4f, %.4f]", iv_surface.min(), iv_surface.max())

    # Build input tensors
    n_quotes = len(quotes)
    pc = np.zeros((n_quotes, 7))
    for i, q in enumerate(quotes):
        log_m = np.log(q.strike / 100.0)
        pc[i, 0] = log_m
        pc[i, 1] = q.maturity
        pc[i, 2] = q.implied_vol
        pc[i, 3] = q.bid_ask_spread
        pc[i, 4] = q.call_price / 100.0
        pc[i, 5] = 0.5  # delta placeholder
        pc[i, 6] = 0.01  # gamma placeholder

    # Pad to max
    max_q = 60
    pad = np.zeros((max_q - n_quotes, 7))
    pc_padded = np.concatenate([pc, pad], axis=0)
    mask = np.zeros(max_q)
    mask[:n_quotes] = 1.0

    eval_k = np.linspace(-0.3, 0.3, 20)
    eval_tau = tau_grid[:10]

    point_cloud = torch.tensor(pc_padded, dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    eval_k_t = torch.tensor(eval_k, dtype=torch.float32).unsqueeze(0).to(device)
    eval_tau_t = torch.tensor(eval_tau, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(point_cloud, mask_t, eval_k_t, eval_tau_t)

    iv_pred = output["iv_surface"][0].cpu().numpy()
    logger.info("Predicted surface shape: %s", iv_pred.shape)
    logger.info("Predicted IV range: [%.4f, %.4f]", iv_pred.min(), iv_pred.max())

    # Compare with ground truth
    rmse = np.sqrt(np.mean((iv_pred - iv_surface) ** 2))
    mae = np.mean(np.abs(iv_pred - iv_surface))
    logger.info("Untrained model results (random weights):")
    logger.info("  RMSE: %.4f", rmse)
    logger.info("  MAE:  %.4f", mae)
    logger.info("  (High error expected — model has not been trained)")

    # Print surface cross-section
    logger.info("Surface cross-section (ATM, k=0):")
    atm_idx = 10  # middle of grid
    for j in range(min(5, iv_pred.shape[1])):
        logger.info("  tau=%.3f: true=%.4f, pred=%.4f",
                    eval_tau[j], iv_surface[atm_idx, j], iv_pred[atm_idx, j])

    # Arbitrage check
    arb_rate = model.count_arbitrage_violations(
        output["iv_surface"], 20, min(10, iv_pred.shape[1])
    )
    logger.info("Arbitrage violation rate: %.2f%%", arb_rate * 100)

    logger.info("Demo complete. Run with --mode train for full training.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SAM-Vol: 波动率曲面重建训练与评估",
    )
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["train", "eval", "demo"],
                        help="运行模式: train=训练, eval=评估, demo=快速演示")
    parser.add_argument("--epochs", type=int, default=15,
                        help="训练轮数 (默认: 15)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小 (默认: 16)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率 (默认: 3e-4)")
    parser.add_argument("--d_model", type=int, default=64,
                        help="模型嵌入维度 (默认: 64)")
    parser.add_argument("--n_encoder_layers", type=int, default=2,
                        help="编码器层数 (默认: 2)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="注意力头数 (默认: 4)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="隐层维度 (默认: 128)")
    parser.add_argument("--n_surface_layers", type=int, default=4,
                        help="曲面解码器层数 (默认: 4)")
    parser.add_argument("--bandwidth", type=float, default=0.15,
                        help="局部插值核带宽 (默认: 0.15)")
    parser.add_argument("--arb_weight", type=float, default=10.0,
                        help="无套利正则化权重 (默认: 10.0)")
    parser.add_argument("--n_surfaces", type=int, default=500,
                        help="合成曲面数量 (默认: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.mode == "train":
        run_training(args)
    elif args.mode == "demo":
        run_demo(args)
    else:
        logger.warning("Eval mode requires a trained model checkpoint.")


if __name__ == "__main__":
    main()
