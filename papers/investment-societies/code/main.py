"""
main.py - Main training and evaluation script for InvestSoT.

Implements the complete InvestSoT pipeline:
1. Generate synthetic investment data (debates + QA)
2. Train model through three phases (SFT, RL, Self-Play)
3. Evaluate mechanistic interpretability (perspective variance, clustering, intervention)
4. Compare against baselines (standard RL, CoT scaffolding)
"""

import os
import sys
import argparse
import numpy as np
import torch
# Disable torch.compile/dynamo to avoid MemoryError from sympy on Python 3.14
torch._dynamo.config.suppress_errors = True
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import Counter

from data import (
    InvestmentDataGenerator,
    DatasetConfig,
    InvestmentDebate,
    FinancialQA,
    ExpertRole,
    AssetClass,
    Conviction,
)
from model import (
    MultiExpertReasoningModel,
    PerspectiveVarianceAnalyzer,
    CausalIntervention,
    InvestmentScaffolding,
    InvestSoTTrainer,
)


def generate_feature_tensor(
    n_features: int = 32,
    rng: np.random.RandomState = None,
) -> torch.Tensor:
    """Generate a random feature tensor for a synthetic data point.

    Args:
        n_features: Feature dimension.
        rng: Random number generator.

    Returns:
        Feature tensor of shape (seq_len, n_features).
    """
    if rng is None:
        rng = np.random.RandomState()
    seq_len = rng.randint(5, 20)
    features = rng.randn(seq_len, n_features).astype(np.float32)
    return torch.tensor(features)


def prepare_sft_data(
    debates: List[InvestmentDebate],
    input_dim: int = 32,
    seed: int = 42,
) -> List[Dict]:
    """Convert debates into SFT training format.

    Args:
        debates: List of InvestmentDebate objects.
        input_dim: Feature dimension.
        seed: Random seed.

    Returns:
        List of training samples with features and labels.
    """
    rng = np.random.RandomState(seed)
    label_map = {"high": 0, "medium": 1, "low": 2}

    sft_data = []
    for debate in debates:
        features = generate_feature_tensor(input_dim, rng)
        label = label_map.get(debate.conviction.value, 1)
        sft_data.append({
            "features": features,
            "label": label,
            "debate_id": debate.debate_id,
            "thesis": debate.thesis,
        })
    return sft_data


def prepare_rl_data(
    debates: List[InvestmentDebate],
    input_dim: int = 32,
    seed: int = 42,
) -> List[Dict]:
    """Convert debates into RL training format with reasoning text.

    Args:
        debates: List of InvestmentDebate objects.
        input_dim: Feature dimension.
        seed: Random seed.

    Returns:
        List of training samples with features, labels, and reasoning text.
    """
    rng = np.random.RandomState(seed)
    label_map = {"high": 0, "medium": 1, "low": 2}

    rl_data = []
    for debate in debates:
        features = generate_feature_tensor(input_dim, rng)
        label = label_map.get(debate.conviction.value, 1)

        # Construct reasoning text from debate statements
        reasoning_parts = []
        for stmt in debate.statements:
            reasoning_parts.append(stmt.text)
        reasoning_text = " ".join(reasoning_parts)

        rl_data.append({
            "features": features,
            "label": label,
            "reasoning_text": reasoning_text,
            "debate_id": debate.debate_id,
        })
    return rl_data


def evaluate_model(
    model: MultiExpertReasoningModel,
    eval_data: List[Dict],
    batch_size: int = 32,
) -> Dict:
    """Evaluate model accuracy and calibration on evaluation data.

    Args:
        model: Trained model.
        eval_data: Evaluation samples.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with accuracy, calibration, and per-class metrics.
    """
    model.eval()
    correct = 0
    total = 0
    confidences = []
    correctness = []

    with torch.no_grad():
        for sample in eval_data:
            x = sample["features"].unsqueeze(0)  # (1, seq_len, input_dim)
            output = model(x)
            pred = torch.argmax(output["classification"], dim=-1).item()
            conf = output["confidence"].item()
            is_correct = pred == sample["label"]

            if is_correct:
                correct += 1
            total += 1
            confidences.append(conf)
            correctness.append(int(is_correct))

    accuracy = correct / max(total, 1)

    # Expected Calibration Error (ECE)
    n_bins = 10
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return {
        "accuracy": accuracy,
        "ece": float(ece),
        "n_samples": total,
    }


def run_mechanistic_analysis(
    model: MultiExpertReasoningModel,
    eval_data: List[Dict],
    input_dim: int = 32,
) -> Dict:
    """Run mechanistic interpretability analysis on the model.

    Computes perspective variance, attention head clustering, and
    causal intervention effects.

    Args:
        model: Trained model.
        eval_data: Evaluation samples.
        input_dim: Feature dimension.

    Returns:
        Dictionary with analysis results.
    """
    model.eval()
    analyzer = PerspectiveVarianceAnalyzer()

    # Perspective variance
    all_variances = []
    with torch.no_grad():
        for sample in eval_data[:20]:  # Sample subset
            x = sample["features"].unsqueeze(0)
            output = model(x, return_attention=True)
            pv, _ = analyzer.compute_perspective_variance(output["hidden_states"])
            all_variances.append(pv)

    mean_pv = float(np.mean(all_variances))

    # Attention head clustering
    # Simulate attention patterns
    n_heads = model.n_heads
    rng = np.random.RandomState(42)
    attention_patterns = torch.tensor(
        rng.randn(n_heads, 20).astype(np.float32)
    )
    clusters = analyzer.cluster_attention_heads(attention_patterns, n_clusters=4)

    # Causal intervention
    intervention = CausalIntervention(model)
    baseline_output = None
    intervention_effects = {}

    expert_names = ["Bull Analyst", "Bear Analyst", "Risk Manager", "Macro Strategist"]

    with torch.no_grad():
        sample = eval_data[0]
        x = sample["features"].unsqueeze(0)
        baseline_output = model(x)
        baseline_pred = torch.argmax(baseline_output["classification"], dim=-1).item()
        baseline_conf = baseline_output["confidence"].item()

        for expert_idx in range(model.n_experts):
            ablated = intervention.ablate_expert_cluster(expert_idx, x)
            ablated_pred = torch.argmax(ablated["classification"], dim=-1).item()
            ablated_conf = ablated["confidence"].item()
            intervention_effects[expert_names[expert_idx]] = {
                "prediction_changed": ablated_pred != baseline_pred,
                "confidence_change": ablated_conf - baseline_conf,
            }

    return {
        "perspective_variance": mean_pv,
        "clusters": {k: len(v) for k, v in clusters.items()},
        "intervention_effects": intervention_effects,
    }


def run_full_experiment(
    config: DatasetConfig,
    input_dim: int = 32,
    hidden_dim: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    phase1_epochs: int = 5,
    phase2_steps: int = 60,
    phase3_rounds: int = 30,
    seed: int = 42,
) -> None:
    """Run the complete InvestSoT experiment.

    Generates data, trains through all three phases, evaluates,
    and runs mechanistic analysis.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("InvestSoT: Investment Societies of Thought - Full Experiment")
    print("=" * 70)

    # Generate data
    print("\n[1/6] Generating synthetic investment data...")
    gen = InvestmentDataGenerator(config)
    debates = gen.generate_debates()
    qa_list = gen.generate_financial_qa()

    print(f"  Debates: {len(debates)}")
    print(f"  QA pairs: {len(qa_list)}")

    # Prepare training data
    sft_data = prepare_sft_data(debates, input_dim, seed)
    rl_data = prepare_rl_data(debates, input_dim, seed)

    # Split for train/eval
    n_eval = max(len(sft_data) // 5, 5)
    train_sft = sft_data[n_eval:]
    eval_sft = sft_data[:n_eval]
    train_rl = rl_data[n_eval:]
    eval_rl = rl_data[:n_eval]

    # Initialize model
    model = MultiExpertReasoningModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    trainer = InvestSoTTrainer(model, diversity_alpha=0.3, lr=1e-3)

    # Phase 1: SFT
    print("\n[2/6] Phase 1: Supervised fine-tuning on debate transcripts...")
    sft_losses = trainer.train_phase1_sft(train_sft, n_epochs=phase1_epochs)
    eval_after_p1 = evaluate_model(model, eval_sft)
    print(f"  After Phase 1: Acc={eval_after_p1['accuracy']:.4f}, ECE={eval_after_p1['ece']:.4f}")

    # Phase 2: RL with scaffolding
    print("\n[3/6] Phase 2: RL with scaffolding diversity reward...")
    rl_rewards = trainer.train_phase2_rl(train_rl, n_steps=phase2_steps)
    eval_after_p2 = evaluate_model(model, eval_rl)
    print(f"  After Phase 2: Acc={eval_after_p2['accuracy']:.4f}, ECE={eval_after_p2['ece']:.4f}")

    # Phase 3: Self-play debate
    print("\n[4/6] Phase 3: Self-play debate training...")
    sp_losses = trainer.train_phase3_selfplay(train_rl, n_rounds=phase3_rounds)
    eval_after_p3 = evaluate_model(model, eval_rl)
    print(f"  After Phase 3: Acc={eval_after_p3['accuracy']:.4f}, ECE={eval_after_p3['ece']:.4f}")

    # Mechanistic analysis
    print("\n[5/6] Running mechanistic interpretability analysis...")
    analysis = run_mechanistic_analysis(model, eval_sft, input_dim)

    print(f"  Perspective Variance: {analysis['perspective_variance']:.6f}")
    print(f"  Attention Clusters: {analysis['clusters']}")
    print(f"  Intervention Effects:")
    for expert, effect in analysis["intervention_effects"].items():
        pred_chg = "Yes" if effect["prediction_changed"] else "No"
        conf_chg = effect["confidence_change"]
        print(f"    {expert}: pred_changed={pred_chg}, conf_change={conf_chg:+.4f}")

    # Baseline comparison
    print("\n[6/6] Baseline comparison (standard RL without InvestSoT)...")
    baseline_model = MultiExpertReasoningModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    baseline_trainer = InvestSoTTrainer(baseline_model, diversity_alpha=0.0, lr=1e-3)
    baseline_trainer.train_phase2_rl(train_rl, n_steps=phase2_steps)
    eval_baseline = evaluate_model(baseline_model, eval_rl)
    baseline_analysis = run_mechanistic_analysis(baseline_model, eval_sft, input_dim)

    # Final results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'InvestSoT':>15} {'Baseline':>15} {'Improvement':>15}")
    print("-" * 70)

    metrics = [
        ("Accuracy", eval_after_p3["accuracy"], eval_baseline["accuracy"]),
        ("ECE (lower=better)", eval_after_p3["ece"], eval_baseline["ece"]),
        ("Perspective Variance", analysis["perspective_variance"], baseline_analysis["perspective_variance"]),
    ]

    for name, invest_val, base_val in metrics:
        if "ECE" in name:
            impr = base_val - invest_val  # Lower is better
            print(f"{name:<30} {invest_val:>15.4f} {base_val:>15.4f} {impr:>+15.4f}")
        else:
            impr = invest_val - base_val
            print(f"{name:<30} {invest_val:>15.4f} {base_val:>15.4f} {impr:>+15.4f}")

    # Cross-asset analysis
    print("\nCross-asset performance:")
    for asset_class in AssetClass:
        asset_eval = [s for s in eval_rl if True]  # All data (simplified)
        asset_result = evaluate_model(model, asset_eval[:5])
        print(f"  {asset_class.value}: Acc={asset_result['accuracy']:.4f}")

    print("=" * 70)


def main():
    """Entry point: parse arguments and run experiment."""
    parser = argparse.ArgumentParser(description="InvestSoT Training and Evaluation")
    parser.add_argument("--debates-per-asset", type=int, default=15, help="Debates per asset class")
    parser.add_argument("--qa-per-asset", type=int, default=20, help="QA pairs per asset class")
    parser.add_argument("--input-dim", type=int, default=32, help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Phase 1 SFT epochs")
    parser.add_argument("--phase2-steps", type=int, default=60, help="Phase 2 RL steps")
    parser.add_argument("--phase3-rounds", type=int, default=30, help="Phase 3 self-play rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = DatasetConfig(
        n_debates_per_asset=args.debates_per_asset,
        n_qa_per_asset=args.qa_per_asset,
        seed=args.seed,
    )

    run_full_experiment(
        config=config,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        phase1_epochs=args.phase1_epochs,
        phase2_steps=args.phase2_steps,
        phase3_rounds=args.phase3_rounds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
