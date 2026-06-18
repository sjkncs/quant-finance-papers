"""
model.py - PortESamp Core: Strategy Distiller and Novelty-Guided Sampling
=========================================================================
Implements the core PortESamp framework for diverse portfolio generation.

Key components:
- PortfolioEncoder: Shallow feature extractor for portfolio proposals
- PortfolioOptimizer: Deep model producing portfolio weights from features
- StrategyDistiller: Lightweight MLP predicting deep representations
- PortESampSampler: Novelty-guided portfolio sampling with reweighting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class PortfolioProposal:
    """A candidate portfolio with its internal representations."""
    weights: np.ndarray
    shallow_features: torch.Tensor
    deep_representation: torch.Tensor
    novelty_score: float = 0.0
    sharpe: float = 0.0


class PortfolioEncoder(nn.Module):
    """Shallow feature extractor for market conditions.

    Transforms raw market features into a compact feature vector
    that serves as input to both the deep optimizer and the distiller.

    Architecture: 2-layer MLP, 128 hidden, ReLU.
    """

    def __init__(self, input_dim: int = 50, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shallow features.

        Args:
            x: Market feature tensor of shape (batch, input_dim).

        Returns:
            Shallow feature tensor of shape (batch, output_dim).
        """
        return self.net(x)


class DeepPortfolioProcessor(nn.Module):
    """Deep processing network that transforms shallow features
    into rich internal representations.

    Architecture: 4-layer MLP with skip connections, 256 hidden.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process shallow features into deep representation.

        Args:
            x: Shallow feature tensor of shape (batch, input_dim).

        Returns:
            Tuple of (deep_representation, output_features).
        """
        h = self.act(self.layer1(x))
        h = self.act(self.norm(self.layer2(h))) + h  # skip connection
        h = self.act(self.layer3(h)) + h  # skip connection
        deep_repr = self.layer4(h)
        return deep_repr, h


class PortfolioHead(nn.Module):
    """Maps deep representations to portfolio weights via softmax.

    Architecture: 2-layer MLP with softmax output.
    """

    def __init__(self, input_dim: int = 128, num_assets: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_assets),
        )

    def forward(self, deep_repr: torch.Tensor) -> torch.Tensor:
        """Generate portfolio weights.

        Args:
            deep_repr: Deep representation tensor.

        Returns:
            Portfolio weights tensor (softmax-normalized).
        """
        logits = self.net(deep_repr)
        weights = F.softmax(logits, dim=-1)
        return weights


class PortfolioModel(nn.Module):
    """Complete portfolio optimization model with decomposable internals.

    Composed of: Encoder -> DeepProcessor -> Head
    Exposes shallow features and deep representations for distillation.
    """

    def __init__(
        self,
        input_dim: int = 50,
        num_assets: int = 100,
        feature_dim: int = 64,
        deep_dim: int = 128,
    ):
        super().__init__()
        self.encoder = PortfolioEncoder(input_dim, feature_dim)
        self.processor = DeepPortfolioProcessor(feature_dim, 256, deep_dim)
        self.head = PortfolioHead(deep_dim, num_assets)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass exposing all internal representations.

        Args:
            x: Market feature tensor.

        Returns:
            Tuple of (weights, shallow_features, deep_representation).
        """
        shallow = self.encoder(x)
        deep, _ = self.processor(shallow)
        weights = self.head(deep)
        return weights, shallow, deep

    def generate_with_noise(
        self, x: torch.Tensor, noise_scale: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate portfolio with injected noise for diverse proposals.

        Args:
            x: Market feature tensor.
            noise_scale: Standard deviation of Gaussian noise.

        Returns:
            Tuple of (weights, shallow_features, deep_representation).
        """
        shallow = self.encoder(x)
        noise = torch.randn_like(shallow) * noise_scale
        shallow_noisy = shallow + noise
        deep, _ = self.processor(shallow_noisy)
        weights = self.head(deep)
        return weights, shallow_noisy, deep


class StrategyDistiller(nn.Module):
    """Lightweight MLP that predicts deep representations from shallow features.

    The prediction error serves as a novelty signal: large errors
    indicate strategies in unexplored regions of the strategy space.

    Architecture: 2-layer MLP, 256 hidden, ReLU.
    """

    def __init__(self, input_dim: int = 64, output_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, shallow_features: torch.Tensor) -> torch.Tensor:
        """Predict deep representations from shallow features.

        Args:
            shallow_features: Tensor of shape (batch, input_dim).

        Returns:
            Predicted deep representation of shape (batch, output_dim).
        """
        return self.net(shallow_features)

    def compute_novelty(
        self,
        shallow_features: torch.Tensor,
        actual_deep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute novelty score as prediction error.

        Args:
            shallow_features: Shallow feature tensor.
            actual_deep: Actual deep representation from the portfolio model.

        Returns:
            Novelty scores (L2 norm of prediction error) per sample.
        """
        predicted = self.forward(shallow_features)
        error = torch.norm(predicted - actual_deep, dim=-1)
        return error


class PortESampSampler:
    """Novelty-guided portfolio sampling using the Strategy Distiller.

    Generates diverse portfolio proposals by reweighting candidate
    sampling probabilities based on novelty scores.
    """

    def __init__(
        self,
        portfolio_model: PortfolioModel,
        distiller: StrategyDistiller,
        beta: float = 0.25,
        num_proposals: int = 64,
        num_select: int = 8,
        noise_scale: float = 0.15,
    ):
        self.model = portfolio_model
        self.distiller = distiller
        self.beta = beta
        self.num_proposals = num_proposals
        self.num_select = num_select
        self.noise_scale = noise_scale

    @torch.no_grad()
    def generate_diverse_portfolios(
        self, market_features: torch.Tensor
    ) -> List[PortfolioProposal]:
        """Generate a diverse set of portfolio proposals.

        Args:
            market_features: Market feature tensor of shape (input_dim,).

        Returns:
            List of PortfolioProposal objects sorted by novelty-adjusted score.
        """
        self.model.eval()
        self.distiller.eval()

        proposals = []

        # Step 1: Generate P candidate proposals with noise
        for p in range(self.num_proposals):
            weights, shallow, deep = self.model.generate_with_noise(
                market_features.unsqueeze(0), self.noise_scale
            )
            proposals.append(PortfolioProposal(
                weights=weights.squeeze().numpy(),
                shallow_features=shallow.squeeze(),
                deep_representation=deep.squeeze(),
            ))

        # Step 2: Compute novelty scores
        shallow_batch = torch.stack([p.shallow_features for p in proposals])
        deep_batch = torch.stack([p.deep_representation for p in proposals])

        novelty_scores = self.distiller.compute_novelty(shallow_batch, deep_batch)
        novelty_std = novelty_scores.std() + 1e-8

        # Step 3: Reweight sampling probabilities by novelty
        log_probs = torch.zeros(self.num_proposals)
        for i in range(self.num_proposals):
            proposals[i].novelty_score = novelty_scores[i].item()
            log_probs[i] = self.beta * novelty_scores[i] / novelty_std

        # Convert to probabilities
        probs = F.softmax(log_probs, dim=0)

        # Step 4: Sample K portfolios with replacement
        indices = torch.multinomial(probs, self.num_select, replacement=False)

        selected = []
        for idx in indices:
            prop = proposals[idx.item()]
            selected.append(prop)

        return selected

    def train_distiller_step(
        self,
        market_features_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train the distiller on a batch of market features.

        Args:
            market_features_batch: Batch of market features.
            optimizer: Distiller optimizer.

        Returns:
            MSE loss value.
        """
        self.model.eval()
        self.distiller.train()

        with torch.no_grad():
            _, shallow, deep = self.model(market_features_batch)

        predicted = self.distiller(shallow)
        loss = F.mse_loss(predicted, deep)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class DistillerReplayBuffer:
    """Replay buffer for asynchronous distiller training.

    Stores (shallow_features, deep_representation) pairs collected
    during portfolio generation for background distiller training.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.shallow_buffer: List[torch.Tensor] = []
        self.deep_buffer: List[torch.Tensor] = []

    def add(self, shallow: torch.Tensor, deep: torch.Tensor):
        """Add a (shallow, deep) pair to the buffer."""
        self.shallow_buffer.append(shallow.detach().cpu())
        self.deep_buffer.append(deep.detach().cpu())
        if len(self.shallow_buffer) > self.max_size:
            self.shallow_buffer.pop(0)
            self.deep_buffer.pop(0)

    def sample_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples.

        Returns:
            Tuple of (shallow_batch, deep_batch).
        """
        n = len(self.shallow_buffer)
        if n == 0:
            raise ValueError("Buffer is empty")
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        shallow = torch.stack([self.shallow_buffer[i] for i in indices])
        deep = torch.stack([self.deep_buffer[i] for i in indices])
        return shallow, deep

    def __len__(self):
        return len(self.shallow_buffer)


if __name__ == "__main__":
    # Smoke test
    num_assets = 50
    input_dim = 50

    model = PortfolioModel(input_dim=input_dim, num_assets=num_assets)
    distiller = StrategyDistiller(input_dim=64, output_dim=128)
    sampler = PortESampSampler(model, distiller, beta=0.25, num_proposals=32, num_select=5)

    # Generate market features
    features = torch.randn(input_dim)
    proposals = sampler.generate_diverse_portfolios(features)

    print(f"Generated {len(proposals)} diverse portfolios:")
    for i, p in enumerate(proposals):
        w = p.weights
        entropy = -np.sum(w[w > 1e-8] * np.log(w[w > 1e-8]))
        print(f"  Portfolio {i}: novelty={p.novelty_score:.4f}, "
              f"entropy={entropy:.3f}, max_weight={w.max():.3f}")

    print("\nSmoke test passed!")
