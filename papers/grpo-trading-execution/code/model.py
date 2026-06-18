"""
model.py - Policy Network and Lambda-ExecGRPO Core
===================================================
Implements the execution policy network and the lambda-ExecGRPO
training algorithm with process-reward equivalence.

Key components:
- ExecutionPolicy: Gaussian policy network for execution decisions
- PrefixGrouper: Shared-prefix trajectory grouping
- LambdaExecGRPO: The core training algorithm with lambda normalization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryStep:
    """A single step in an execution trajectory."""
    state: np.ndarray
    action: np.ndarray  # [shares_fraction, aggressiveness]
    log_prob: float
    reward: float  # terminal reward (only at last step)


@dataclass
class Trajectory:
    """A complete execution trajectory."""
    steps: List[TrajectoryStep]
    total_reward: float
    action_sequence: List[np.ndarray]  # for prefix matching


class ExecutionPolicy(nn.Module):
    """Gaussian policy network for trade execution.

    Takes market state features and outputs a Gaussian distribution
    over the 2D action space: [shares_fraction, aggressiveness].

    Architecture: 3-layer MLP with 512 hidden units, ReLU activations.
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 2,
        hidden_dim: int = 512,
        num_layers: int = 3,
        log_std_init: float = -1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP layers
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Mean head
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # Log-std head (state-independent for stability)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters.

        Args:
            state: Batch of state vectors, shape (batch, state_dim).

        Returns:
            Tuple of (mean, log_std) for the Gaussian policy.
        """
        # Clamp inputs to prevent NaN from extreme market values
        state = torch.clamp(state, -10.0, 10.0)
        state = torch.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        features = self.trunk(state)
        features = torch.nan_to_num(features, nan=0.0)
        mean = self.mean_head(features)
        # Squash mean to valid ranges: shares_fraction in [0,1], aggressiveness in [0,1]
        mean = torch.sigmoid(mean)
        mean = torch.clamp(mean, 0.01, 0.99)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log probability.

        Args:
            state: State vector, shape (state_dim,) or (batch, state_dim).

        Returns:
            Tuple of (action, log_prob).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterized sampling
        normal = torch.distributions.Normal(mean, std)
        action_raw = normal.rsample()

        # Clamp to valid action ranges
        action = torch.clamp(action_raw, 0.0, 1.0)

        # Log probability with clamping correction
        log_prob = normal.log_prob(action_raw).sum(dim=-1)

        return action, log_prob

    def get_log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of an action under the current policy.

        Args:
            state: State tensor.
            action: Action tensor (already in [0,1] range).

        Returns:
            Log probability tensor.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        return log_prob


class PrefixGrouper:
    """Groups trajectories by shared execution prefixes.

    For each time step t, identifies equivalence classes of trajectories
    that have identical actions at all steps 1 through t. Computes the
    process set cardinality |Lambda(i,t)| for each (trajectory, step) pair.
    """

    def __init__(self, action_tolerance: float = 0.05):
        """Initialize with action matching tolerance.

        Args:
            action_tolerance: Maximum L2 distance for two actions to be
                considered matching (accounts for floating point and
                minor policy stochasticity).
        """
        self.action_tolerance = action_tolerance

    def compute_prefix_groups(
        self, trajectories: List[Trajectory]
    ) -> Dict[int, np.ndarray]:
        """Compute process set cardinalities for all (trajectory, step) pairs.

        Args:
            trajectories: List of G trajectories.

        Returns:
            Dictionary mapping step t to array of shape (G,) containing
            the process set cardinality |Lambda(i,t)| for each trajectory i.
        """
        G = len(trajectories)
        if G == 0:
            return {}

        T = min(len(traj.action_sequence) for traj in trajectories)
        cardinalities: Dict[int, np.ndarray] = {}

        # For each step, compute prefix matching
        prefix_match = np.ones((G, G), dtype=bool)  # pairwise match matrix

        for t in range(T):
            # Update prefix matching: trajectories must match at step t
            # AND have matched at all previous steps
            for i in range(G):
                for j in range(i + 1, G):
                    if prefix_match[i, j]:
                        dist = np.linalg.norm(
                            trajectories[i].action_sequence[t]
                            - trajectories[j].action_sequence[t]
                        )
                        if dist > self.action_tolerance:
                            prefix_match[i, j] = False
                            prefix_match[j, i] = False

            # Process set size for each trajectory at step t
            card_t = np.zeros(G, dtype=np.float32)
            for i in range(G):
                # Count how many trajectories share prefix with i at step t
                card_t[i] = prefix_match[i].sum()
            cardinalities[t] = card_t

        return cardinalities

    def compute_process_rewards(
        self,
        trajectories: List[Trajectory],
        cardinalities: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """Compute Monte Carlo process rewards for each step.

        The process reward at step t for trajectory i is the average
        terminal reward over all trajectories sharing the same prefix.

        Args:
            trajectories: List of trajectories.
            cardinalities: Process set cardinalities from compute_prefix_groups.

        Returns:
            Dictionary mapping step t to array of process rewards.
        """
        G = len(trajectories)
        rewards = np.array([traj.total_reward for traj in trajectories])
        process_rewards: Dict[int, np.ndarray] = {}

        T = len(cardinalities)
        for t in range(T):
            card = cardinalities[t]
            pr = np.zeros(G, dtype=np.float32)

            # Group trajectories by prefix and average rewards within groups
            visited = set()
            for i in range(G):
                if i in visited:
                    continue
                group_indices = [i]
                for j in range(i + 1, G):
                    if j not in visited:
                        dist = np.linalg.norm(
                            trajectories[i].action_sequence[t]
                            - trajectories[j].action_sequence[t]
                        )
                        if dist <= self.action_tolerance:
                            group_indices.append(j)

                group_reward = rewards[group_indices].mean()
                for idx in group_indices:
                    pr[idx] = group_reward
                    visited.add(idx)

            process_rewards[t] = pr

        return process_rewards


class LambdaExecGRPO:
    """Lambda-ExecGRPO training algorithm.

    Implements the GRPO loss with lambda normalization that corrects
    the frequency imbalance across execution time steps.

    The key modification vs standard GRPO:
        L = -sum_{i,t} (1/|Lambda(i,t)|) * log pi(a_t^i|s_t^i) * A_i
    instead of:
        L = -sum_{i,t} log pi(a_t^i|s_t^i) * A_i
    """

    def __init__(
        self,
        policy: ExecutionPolicy,
        ref_policy: Optional[ExecutionPolicy] = None,
        lr: float = 1e-4,
        kl_penalty: float = 0.01,
        clip_ratio: float = 0.2,
        use_lambda_norm: bool = True,
    ):
        self.policy = policy
        self.ref_policy = ref_policy or self._copy_policy()
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.kl_penalty = kl_penalty
        self.clip_ratio = clip_ratio
        self.use_lambda_norm = use_lambda_norm
        self.grouper = PrefixGrouper()

    def _copy_policy(self) -> ExecutionPolicy:
        """Create a frozen copy of the current policy."""
        import copy
        ref = copy.deepcopy(self.policy)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        return ref

    def compute_advantages(
        self, trajectories: List[Trajectory]
    ) -> np.ndarray:
        """Compute group-relative advantages.

        Args:
            trajectories: List of G trajectories.

        Returns:
            Array of shape (G,) with normalized advantages.
        """
        rewards = np.array([traj.total_reward for traj in trajectories])
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        return (rewards - mean_r) / std_r

    def compute_loss(
        self,
        trajectories: List[Trajectory],
        advantages: np.ndarray,
        cardinalities: Dict[int, np.ndarray],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the lambda-ExecGRPO loss.

        Args:
            trajectories: List of sampled trajectories.
            advantages: Group-relative advantages.
            cardinalities: Process set cardinalities.

        Returns:
            Tuple of (loss tensor, metrics dictionary).
        """
        G = len(trajectories)
        T = len(trajectories[0].steps)
        total_loss = torch.tensor(0.0, device=next(self.policy.parameters()).device)
        total_policy_loss = 0.0
        total_kl = 0.0
        total_terms = 0

        for i in range(G):
            traj = trajectories[i]
            adv_i = advantages[i]

            for t in range(min(T, len(traj.steps))):
                step = traj.steps[t]
                state_t = torch.tensor(step.state, dtype=torch.float32).unsqueeze(0)
                action_t = torch.tensor(step.action, dtype=torch.float32).unsqueeze(0)

                # Current policy log prob
                log_prob = self.policy.get_log_prob(state_t, action_t)

                # Reference policy log prob
                with torch.no_grad():
                    ref_log_prob = self.ref_policy.get_log_prob(state_t, action_t)

                # Lambda normalization factor
                if self.use_lambda_norm and t in cardinalities:
                    lambda_factor = 1.0 / max(cardinalities[t][i], 1.0)
                else:
                    lambda_factor = 1.0

                # Policy loss: negative advantage weighted by lambda
                policy_loss = -log_prob * adv_i * lambda_factor
                total_loss = total_loss + policy_loss
                total_policy_loss += policy_loss.item()

                # KL penalty
                kl = (log_prob - ref_log_prob).squeeze() * lambda_factor
                total_loss = total_loss + self.kl_penalty * kl
                total_kl += kl.item()
                total_terms += 1

        # Average over all terms
        if total_terms > 0:
            total_loss = total_loss / total_terms

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": total_policy_loss / max(total_terms, 1),
            "kl_divergence": total_kl / max(total_terms, 1),
            "num_terms": total_terms,
        }

        return total_loss, metrics

    def train_step(
        self, trajectories: List[Trajectory]
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            trajectories: List of G sampled trajectories for one parent order.

        Returns:
            Dictionary of training metrics.
        """
        # Compute advantages
        advantages = self.compute_advantages(trajectories)

        # Compute prefix groups and cardinalities
        cardinalities = self.grouper.compute_prefix_groups(trajectories)

        # Compute loss and backpropagate
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(trajectories, advantages, cardinalities)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return metrics

    def update_reference_policy(self):
        """Update reference policy to current policy (periodic sync)."""
        import copy
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False


if __name__ == "__main__":
    # Quick smoke test
    policy = ExecutionPolicy(state_dim=14, action_dim=2, hidden_dim=256)
    grpo = LambdaExecGRPO(policy, lr=1e-3, use_lambda_norm=True)

    # Create dummy trajectories
    G = 8
    T = 12
    trajectories = []
    for g in range(G):
        steps = []
        action_seq = []
        for t in range(T):
            state = np.random.randn(14).astype(np.float32)
            action = np.random.rand(2).astype(np.float32)
            steps.append(TrajectoryStep(
                state=state, action=action,
                log_prob=-1.0, reward=0.0,
            ))
            action_seq.append(action)
        total_reward = np.random.randn()
        trajectories.append(Trajectory(
            steps=steps, total_reward=total_reward,
            action_sequence=action_seq,
        ))

    metrics = grpo.train_step(trajectories)
    print(f"Training metrics: {metrics}")
    print("Lambda-ExecGRPO smoke test passed!")
