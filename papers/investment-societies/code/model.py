"""
model.py - Core model architecture for InvestSoT (Investment Societies of Thought).

Implements:
1. Multi-expert reasoning model with attention head clustering
2. Mechanistic interpretability analysis tools (perspective variance, causal intervention)
3. Investment scaffolding module
4. InvestSoT training framework with diversity reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data import ExpertRole, AssetClass, Conviction


class MultiExpertReasoningModel(nn.Module):
    """Simulated multi-expert reasoning model with attention head specialization.

    This model architecture simulates a transformer-based reasoning model
    where attention heads naturally specialize into expert roles (bull analyst,
    bear analyst, risk manager, macro strategist).

    Architecture:
    - Embedding layer for input features
    - Multi-head self-attention layers
    - Expert-specialized attention heads (simulated via learned routing)
    - Reasoning chain decoder
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_experts: int = 4,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ))

        # Expert routing heads (learned soft assignment)
        self.expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )

        # Reasoning chain decoder
        self.reasoning_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output heads
        self.classification_head = nn.Linear(hidden_dim // 2, 3)  # buy/hold/sell
        self.conviction_head = nn.Linear(hidden_dim // 2, 3)  # high/medium/low
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)  # continuous confidence

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the multi-expert reasoning model.

        Args:
            x: Input features of shape (batch, seq_len, input_dim).
            return_attention: Whether to return attention weights for analysis.

        Returns:
            Dictionary with classification, conviction, confidence, and
            optionally expert routing and hidden states.
        """
        batch_size, seq_len, _ = x.shape

        # Project and add positional encoding
        h = self.input_proj(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = h + self.pos_embedding(positions)

        # Transformer layers
        hidden_states = [h]
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)

        # Expert routing (simulated perspective switching)
        expert_logits = self.expert_router(h)  # (batch, seq, n_experts)
        expert_weights = F.softmax(expert_logits, dim=-1)

        # Weighted aggregation by expert
        expert_representations = []
        for e in range(self.n_experts):
            w = expert_weights[:, :, e:e+1]  # (batch, seq, 1)
            expert_rep = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-8)
            expert_representations.append(expert_rep)

        # Combine expert perspectives
        combined = torch.stack(expert_representations, dim=1)  # (batch, n_experts, hidden)
        pooled = combined.mean(dim=1)  # (batch, hidden)

        # Decode reasoning
        reasoning = self.reasoning_decoder(pooled)

        # Output predictions
        classification = self.classification_head(reasoning)
        conviction = self.conviction_head(reasoning)
        confidence = torch.sigmoid(self.confidence_head(reasoning))

        output = {
            "classification": classification,
            "conviction": conviction,
            "confidence": confidence,
            "expert_weights": expert_weights,
            "expert_representations": combined,
        }

        if return_attention:
            output["hidden_states"] = hidden_states

        return output


class PerspectiveVarianceAnalyzer:
    """Analyzes internal perspective variability of reasoning models.

    Computes the perspective variance score by measuring cosine distance
    between consecutive hidden states in the reasoning chain.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_perspective_variance(
        hidden_states: List[torch.Tensor],
    ) -> Tuple[float, np.ndarray]:
        """Compute perspective variance from hidden state sequence.

        Args:
            hidden_states: List of hidden state tensors from each layer,
                          each of shape (batch, seq_len, hidden_dim).

        Returns:
            Tuple of (variance score, array of pairwise distances).
        """
        # Use the final layer hidden states
        h = hidden_states[-1]  # (batch, seq_len, hidden_dim)

        if h.dim() == 3:
            h = h[0]  # Take first batch element: (seq_len, hidden_dim)

        seq_len = h.shape[0]
        if seq_len < 2:
            return 0.0, np.array([0.0])

        # Compute cosine distances between consecutive positions
        distances = []
        for t in range(seq_len - 1):
            h_t = h[t]
            h_t1 = h[t + 1]
            cos_sim = F.cosine_similarity(h_t.unsqueeze(0), h_t1.unsqueeze(0)).item()
            distances.append(1 - cos_sim)  # Convert similarity to distance

        distances = np.array(distances)
        variance = float(np.var(distances))

        return variance, distances

    @staticmethod
    def cluster_attention_heads(
        attention_patterns: torch.Tensor,
        n_clusters: int = 4,
    ) -> Dict[int, List[int]]:
        """Cluster attention heads by their activation patterns.

        Args:
            attention_patterns: Tensor of shape (n_heads, n_tokens) representing
                               average attention weights for each head.
            n_clusters: Number of clusters (default 4 for the 4 expert roles).

        Returns:
            Dictionary mapping cluster_id to list of head indices.
        """
        n_heads = attention_patterns.shape[0]

        # Compute correlation matrix between heads
        corr = torch.corrcoef(attention_patterns)

        # Spectral clustering via eigendecomposition of Laplacian
        # D - A where D is degree matrix and A is adjacency (correlation)
        adj = torch.clamp(corr, min=0)  # Non-negative correlations only
        degree = adj.sum(dim=1)
        laplacian = torch.diag(degree) - adj

        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        # Use first k eigenvectors for clustering
        features = eigenvectors[:, :n_clusters]

        # Simple k-means on spectral features
        clusters: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}

        # Initialize centroids randomly
        rng = np.random.RandomState(42)
        centroid_idx = rng.choice(n_heads, size=n_clusters, replace=False)
        centroids = features[centroid_idx]

        for _ in range(20):  # K-means iterations
            # Assign to nearest centroid
            dists = torch.cdist(features, centroids)
            assignments = torch.argmin(dists, dim=1)

            # Update centroids
            for c in range(n_clusters):
                mask = assignments == c
                if mask.any():
                    centroids[c] = features[mask].mean(dim=0)

        for head_idx in range(n_heads):
            cluster_id = assignments[head_idx].item()
            clusters[cluster_id].append(head_idx)

        return clusters


class CausalIntervention:
    """Performs causal interventions on attention head clusters.

    Zeros out specific attention head outputs to test whether
    clusters are functionally responsible for specific capabilities.
    """

    def __init__(self, model: MultiExpertReasoningModel):
        self.model = model

    def ablate_expert_cluster(
        self,
        expert_idx: int,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Ablate (zero out) a specific expert's contribution.

        Args:
            expert_idx: Index of the expert cluster to ablate (0-3).
            x: Input tensor.

        Returns:
            Model output with the specified expert ablated.
        """
        # Forward pass
        output = self.model(x)

        # Zero out the ablated expert's representation
        expert_reps = output["expert_representations"].clone()
        expert_reps[:, expert_idx, :] = 0.0

        # Recompute pooled representation without the ablated expert
        remaining = expert_reps.clone()
        remaining[:, expert_idx, :] = 0.0
        active_count = self.model.n_experts - 1
        pooled = remaining.sum(dim=1) / max(active_count, 1)

        # Re-decode
        reasoning = self.model.reasoning_decoder(pooled)
        classification = self.model.classification_head(reasoning)
        conviction = self.model.conviction_head(reasoning)
        confidence = torch.sigmoid(self.model.confidence_head(reasoning))

        return {
            "classification": classification,
            "conviction": conviction,
            "confidence": confidence,
            "ablated_expert": expert_idx,
        }


class InvestmentScaffolding:
    """Investment conversational scaffolding module.

    Structures model reasoning along bull/bear/risk/macro dimensions
    and computes a diversity reward for RL training.
    """

    def __init__(self):
        self.perspective_keywords = {
            ExpertRole.BULL_ANALYST: [
                "growth", "opportunity", "strength", "advantage", "catalyst",
                "undervalued", "momentum", "beat", "upgrade", "expansion",
            ],
            ExpertRole.BEAR_ANALYST: [
                "risk", "concern", "weakness", "threat", "headwind",
                "overvalued", "decline", "miss", "downgrade", "compression",
            ],
            ExpertRole.RISK_MANAGER: [
                "exposure", "drawdown", "concentration", "correlation", "var",
                "liquidity", "leverage", "volatility", "stress", "scenario",
            ],
            ExpertRole.MACRO_STRATEGIST: [
                "rates", "inflation", "gdp", "regime", "monetary",
                "fiscal", "geopolitical", "cross-asset", "cycle", "policy",
            ],
        }

    def compute_diversity_score(self, reasoning_text: str) -> float:
        """Compute the diversity reward based on perspective coverage.

        Args:
            reasoning_text: Model's reasoning text.

        Returns:
            Diversity score between 0 (single perspective) and 1 (all perspectives).
        """
        text_lower = reasoning_text.lower()
        perspectives_covered = 0

        for role, keywords in self.perspective_keywords.items():
            # Check if any keyword from this perspective appears
            if any(kw in text_lower for kw in keywords):
                perspectives_covered += 1

        return perspectives_covered / len(self.perspective_keywords)

    def compute_scaffolding_reward(
        self,
        reasoning_text: str,
        accuracy_reward: float,
        alpha: float = 0.3,
    ) -> float:
        """Compute combined reward with diversity bonus.

        R = R_accuracy + alpha * R_diversity

        Args:
            reasoning_text: Model's reasoning text.
            accuracy_reward: Task accuracy reward.
            alpha: Weight for diversity bonus.

        Returns:
            Combined reward.
        """
        diversity = self.compute_diversity_score(reasoning_text)
        return accuracy_reward + alpha * diversity


class InvestSoTTrainer:
    """InvestSoT training framework.

    Implements three-phase training:
    1. SFT on expert debate transcripts
    2. RL with scaffolding diversity reward
    3. Self-play debate
    """

    def __init__(
        self,
        model: MultiExpertReasoningModel,
        scaffolding: Optional[InvestmentScaffolding] = None,
        lr: float = 1e-4,
        diversity_alpha: float = 0.3,
    ):
        self.model = model
        self.scaffolding = scaffolding or InvestmentScaffolding()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.diversity_alpha = diversity_alpha
        self.training_log: List[Dict] = []

    def train_phase1_sft(
        self,
        sft_data: List[Dict],
        n_epochs: int = 5,
        batch_size: int = 16,
    ) -> List[float]:
        """Phase 1: Supervised fine-tuning on expert debate transcripts.

        Args:
            sft_data: List of dicts with 'features' and 'labels'.
            n_epochs: Number of training epochs.
            batch_size: Batch size.

        Returns:
            List of epoch losses.
        """
        self.model.train()
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            indices = np.random.permutation(len(sft_data))
            for start in range(0, len(sft_data), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_features = [sft_data[i]["features"] for i in batch_idx]

                # Pad variable-length sequences to max length in batch
                max_len = max(f.shape[0] for f in batch_features)
                feat_dim = batch_features[0].shape[-1]
                padded = []
                for f in batch_features:
                    if f.shape[0] < max_len:
                        pad = torch.zeros(max_len - f.shape[0], feat_dim)
                        padded.append(torch.cat([f, pad], dim=0))
                    else:
                        padded.append(f)
                batch_x = torch.stack(padded)
                batch_y = torch.tensor([sft_data[i]["label"] for i in batch_idx])

                output = self.model(batch_x)
                loss = F.cross_entropy(output["classification"], batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            print(f"  Phase 1 Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

        return losses

    def train_phase2_rl(
        self,
        rl_data: List[Dict],
        n_steps: int = 100,
    ) -> List[float]:
        """Phase 2: RL with scaffolding diversity reward.

        Args:
            rl_data: List of dicts with 'features', 'label', and 'reasoning_text'.
            n_steps: Number of RL training steps.

        Returns:
            List of step rewards.
        """
        self.model.train()
        rewards = []

        for step in range(n_steps):
            idx = np.random.randint(0, len(rl_data))
            sample = rl_data[idx]
            x = sample["features"].unsqueeze(0)
            label = sample["label"]

            output = self.model(x)

            # Accuracy reward
            pred = torch.argmax(output["classification"], dim=-1).item()
            accuracy_reward = 1.0 if pred == label else 0.0

            # Diversity reward (from simulated reasoning text)
            reasoning_text = sample.get("reasoning_text", "")
            total_reward = self.scaffolding.compute_scaffolding_reward(
                reasoning_text, accuracy_reward, self.diversity_alpha
            )

            # Policy gradient update (REINFORCE-style)
            log_prob = F.log_softmax(output["classification"], dim=-1)
            selected_log_prob = log_prob[0, label]
            loss = -selected_log_prob * total_reward

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            rewards.append(total_reward)

            if (step + 1) % 20 == 0:
                avg_r = np.mean(rewards[-20:])
                print(f"  Phase 2 Step {step+1}/{n_steps} | Avg Reward: {avg_r:.4f}")

        return rewards

    def train_phase3_selfplay(
        self,
        debate_theses: List[Dict],
        n_rounds: int = 50,
    ) -> List[float]:
        """Phase 3: Self-play debate training.

        The model generates both bull and bear arguments for investment theses
        and learns to synthesize them.

        Args:
            debate_theses: List of dicts with 'features' and 'label'.
            n_rounds: Number of self-play rounds.

        Returns:
            List of round losses.
        """
        self.model.train()
        losses = []

        for round_idx in range(n_rounds):
            idx = np.random.randint(0, len(debate_theses))
            sample = debate_theses[idx]
            x = sample["features"].unsqueeze(0)
            label = sample["label"]

            # Bull perspective: maximize classification accuracy
            output_bull = self.model(x)
            loss_bull = F.cross_entropy(output_bull["classification"], torch.tensor([label]))

            # Bear perspective: encourage different expert routing
            expert_weights = output_bull["expert_weights"]
            # Penalize if all experts agree (encourage diversity)
            expert_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=-1).mean()
            diversity_loss = -0.1 * expert_entropy  # Minimize negative entropy

            # Combined loss
            loss = loss_bull + diversity_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            losses.append(loss.item())

            if (round_idx + 1) % 10 == 0:
                avg_l = np.mean(losses[-10:])
                print(f"  Phase 3 Round {round_idx+1}/{n_rounds} | Avg Loss: {avg_l:.4f}")

        return losses


if __name__ == "__main__":
    # Quick test of model architecture
    model = MultiExpertReasoningModel(input_dim=32, hidden_dim=128, n_heads=4, n_layers=2)
    x = torch.randn(2, 10, 32)  # batch=2, seq=10, dim=32
    output = model(x, return_attention=True)

    print("Model output keys:", list(output.keys()))
    print("Classification shape:", output["classification"].shape)
    print("Conviction shape:", output["conviction"].shape)
    print("Confidence shape:", output["confidence"].shape)
    print("Expert weights shape:", output["expert_weights"].shape)
    print("Hidden states count:", len(output["hidden_states"]))

    # Test perspective variance
    analyzer = PerspectiveVarianceAnalyzer()
    pv, dists = analyzer.compute_perspective_variance(output["hidden_states"])
    print(f"Perspective variance: {pv:.6f}")
    print(f"Distance array length: {len(dists)}")

    # Test scaffolding
    scaffolding = InvestmentScaffolding()
    test_text = "The growth opportunity is compelling but risk exposure to rates and inflation is elevated."
    div_score = scaffolding.compute_diversity_score(test_text)
    print(f"Diversity score: {div_score:.2f}")
