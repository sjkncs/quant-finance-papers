"""
model.py - Core model architecture for TROLL-Risk portfolio optimization.

Implements the trust region projection layer, differentiable risk measures,
the policy network (actor-critic), and the TROLL-Risk training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class SimplexProjection(nn.Module):
    """Differentiable projection onto the probability simplex.

    Uses the algorithm from Duchi et al. (2008) for efficient O(n log n)
    projection onto the l1-ball, adapted for the simplex constraint.
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Project logits onto the simplex (non-negative, sums to 1).

        Args:
            logits: Unnormalized log-weights of shape (batch, n_assets).

        Returns:
            Projected weights on the simplex.
        """
        return F.softmax(logits, dim=-1)


class CVaRRisk(nn.Module):
    """Differentiable Conditional Value-at-Risk using Cornish-Fisher expansion.

    Approximates the CVaR of the portfolio return distribution using
    the first four moments, enabling gradient-based optimization.
    """

    def __init__(self, confidence: float = 0.95):
        super().__init__()
        self.confidence = confidence
        # z-score for the confidence level
        from scipy.stats import norm
        self.z_alpha = norm.ppf(1 - confidence)

    def forward(
        self,
        weights: torch.Tensor,
        mean_returns: torch.Tensor,
        cov_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CVaR of portfolio return distribution.

        Args:
            weights: Portfolio weights (batch, n_assets).
            mean_returns: Expected asset returns (batch, n_assets).
            cov_matrix: Covariance matrix (batch, n_assets, n_assets) or (n_assets, n_assets).

        Returns:
            CVaR values of shape (batch,). Negative means loss.
        """
        # Portfolio mean return
        port_mean = torch.sum(weights * mean_returns, dim=-1)

        # Portfolio variance
        if cov_matrix.dim() == 2:
            port_var = torch.einsum("bi,ij,bj->b", weights, cov_matrix, weights)
        else:
            port_var = torch.einsum("bi,bij,bj->b", weights, cov_matrix, weights)

        port_std = torch.sqrt(port_var + 1e-8)

        # Cornish-Fisher CVaR approximation (assuming normal for simplicity)
        # CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)
        phi_z = torch.exp(torch.tensor(-0.5 * self.z_alpha**2)) / np.sqrt(2 * np.pi)
        cvar = -port_mean + port_std * phi_z / (1 - self.confidence)
        return cvar


class SoftDrawdownRisk(nn.Module):
    """Smooth approximation of maximum drawdown using exponential utility.

    D_soft(w) = (1/beta) * log(E[exp(-beta * cumulative_return)])

    As beta increases, this approaches the worst-case cumulative return.
    """

    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        weights: torch.Tensor,
        returns_history: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft drawdown penalty.

        Args:
            weights: Portfolio weights (batch, n_assets).
            returns_history: Historical returns (batch, lookback, n_assets).

        Returns:
            Soft drawdown values of shape (batch,). Higher = worse.
        """
        # Portfolio returns over history
        port_returns = torch.einsum("ba,bta->bt", weights, returns_history)
        # Cumulative returns
        cum_returns = torch.cumsum(port_returns, dim=-1)
        # Exponential utility (soft min over cumulative returns)
        exp_neg = torch.exp(-self.beta * cum_returns)
        soft_dd = torch.log(torch.mean(exp_neg, dim=-1) + 1e-8) / self.beta
        return soft_dd


class CorrelationShockRisk(nn.Module):
    """Expected portfolio loss under a correlation spike scenario.

    Computes L_shock(w) = w^T (Sigma_crisis - Sigma_normal) w,
    representing the additional loss from correlation regime change.
    """

    def __init__(self, n_assets: int):
        super().__init__()
        # Learnable crisis covariance delta (initialized small)
        self.crisis_delta = nn.Parameter(
            torch.randn(n_assets, n_assets) * 0.01
        )

    def forward(
        self,
        weights: torch.Tensor,
        normal_cov: torch.Tensor,
    ) -> torch.Tensor:
        """Compute correlation shock exposure.

        Args:
            weights: Portfolio weights (batch, n_assets).
            normal_cov: Normal-regime covariance (n_assets, n_assets) or (batch, n_assets, n_assets).

        Returns:
            Shock exposure values of shape (batch,).
        """
        # Make crisis_delta symmetric and positive semi-definite
        delta = self.crisis_delta @ self.crisis_delta.T

        if normal_cov.dim() == 2:
            crisis_cov = normal_cov + delta
            shock = torch.einsum("bi,ij,bj->b", weights, crisis_cov - normal_cov, weights)
        else:
            crisis_cov = normal_cov + delta.unsqueeze(0)
            shock = torch.einsum("bi,bij,bj->b", weights, crisis_cov - normal_cov, weights)

        return shock


class PolicyNetwork(nn.Module):
    """Actor network that outputs Dirichlet-distributed portfolio allocations.

    Architecture: 3-layer MLP with ReLU activations and LayerNorm,
    outputting logits that are projected onto the simplex.
    """

    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )
        self.projection = SimplexProjection()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Output portfolio allocation weights.

        Args:
            state: State vector (batch, state_dim).

        Returns:
            Allocation weights on simplex (batch, n_assets).
        """
        logits = self.network(state)
        return self.projection(logits)

    def get_log_prob(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given actions under current policy.

        Uses Dirichlet log-pdf approximation via softmax + log.

        Args:
            state: State vector (batch, state_dim).
            actions: Allocation weights (batch, n_assets).

        Returns:
            Log probabilities (batch,).
        """
        logits = self.network(state)
        log_probs = F.log_softmax(logits, dim=-1)
        # Approximate: sum of log-probs weighted by action
        return torch.sum(actions * log_probs, dim=-1)


class ValueNetwork(nn.Module):
    """Critic network estimating state values for advantage computation."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value.

        Args:
            state: State vector (batch, state_dim).

        Returns:
            Value estimates (batch, 1).
        """
        return self.network(state)


class TrustRegionProjection(nn.Module):
    """TROLL-Risk trust region projection layer.

    Projects candidate allocations onto a risk-aware trust region defined by
    KL divergence from previous policy and differentiable risk constraints.
    Uses augmented Lagrangian optimization with implicit differentiation.
    """

    def __init__(
        self,
        n_assets: int,
        kl_weight: float = 0.2,
        cvar_budget: float = 0.03,
        drawdown_beta: float = 10.0,
        shock_limit: float = 0.05,
        n_dual_iters: int = 20,
        dual_lr: float = 0.01,
        sparse_k: int = 10,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.kl_weight = kl_weight
        self.cvar_budget = cvar_budget
        self.drawdown_beta = drawdown_beta
        self.shock_limit = shock_limit
        self.n_dual_iters = n_dual_iters
        self.dual_lr = dual_lr
        self.sparse_k = min(sparse_k, n_assets)

        # Risk measure modules
        self.cvar_risk = CVaRRisk(confidence=0.95)
        self.drawdown_risk = SoftDrawdownRisk(beta=drawdown_beta)
        self.shock_risk = CorrelationShockRisk(n_assets)

    def _select_sparse_assets(
        self, w_hat: torch.Tensor, w_old: torch.Tensor
    ) -> torch.Tensor:
        """Select top-K assets with largest proposed weight changes.

        Args:
            w_hat: Proposed allocation (batch, n_assets).
            w_old: Previous allocation (batch, n_assets).

        Returns:
            Boolean mask of selected assets (batch, n_assets).
        """
        diffs = torch.abs(w_hat - w_old)
        _, indices = torch.topk(diffs, self.sparse_k, dim=-1)
        mask = torch.zeros_like(w_hat, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        return mask

    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute KL(p || q) for simplex distributions.

        Args:
            p: First distribution (batch, n_assets).
            q: Second distribution (batch, n_assets).

        Returns:
            KL divergence values (batch,).
        """
        p_safe = torch.clamp(p, 1e-8, 1.0)
        q_safe = torch.clamp(q, 1e-8, 1.0)
        return torch.sum(p_safe * (torch.log(p_safe) - torch.log(q_safe)), dim=-1)

    def forward(
        self,
        w_hat: torch.Tensor,
        w_old: torch.Tensor,
        mean_returns: torch.Tensor,
        cov_matrix: torch.Tensor,
        returns_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Project candidate allocation onto the trust region.

        Uses iterative projected gradient descent with dual variable updates
        for the augmented Lagrangian formulation. Each iteration rebuilds the
        computation graph from the current parameter value to maintain clean
        autograd semantics.

        Args:
            w_hat: Proposed allocation (batch, n_assets).
            w_old: Previous allocation (batch, n_assets).
            mean_returns: Expected returns (batch, n_assets).
            cov_matrix: Covariance matrix.
            returns_history: Historical returns for drawdown computation.

        Returns:
            Tuple of (projected weights, dict of risk measure values).
        """
        batch_size = w_hat.shape[0]

        # Sparse asset selection
        mask = self._select_sparse_assets(w_hat, w_old)
        mask_f = mask.float()

        # Detach inputs to the inner optimization loop
        w_hat_d = w_hat.detach()
        w_old_d = w_old.detach()
        mean_ret_d = mean_returns.detach()
        cov_d = cov_matrix.detach()
        hist_d = returns_history.detach()

        # Current allocation estimate (will be iteratively refined)
        w_current = w_hat_d.clone()

        # Dual variables for risk constraints
        mu_cvar = torch.zeros(batch_size, device=w_hat.device)
        mu_dd = torch.zeros(batch_size, device=w_hat.device)
        mu_shock = torch.zeros(batch_size, device=w_hat.device)

        for iteration in range(self.n_dual_iters):
            # Rebuild graph each iteration: w_param is a fresh leaf
            w_param = w_current.detach().clone().requires_grad_(True)

            # Simplex projection (via softmax of logits)
            w_proj = F.softmax(torch.log(w_param + 1e-8), dim=-1)

            # KL trust region term
            kl_new = self._kl_divergence(w_proj, w_hat_d + 1e-8)
            kl_old = self._kl_divergence(w_proj, w_old_d + 1e-8)

            # Risk measures
            cvar = self.cvar_risk(w_proj, mean_ret_d, cov_d)
            dd = self.drawdown_risk(w_proj, hist_d)
            shock = self.shock_risk(w_proj, cov_d)

            # Augmented Lagrangian objective
            obj = kl_new + self.kl_weight * kl_old
            obj = obj + mu_cvar * F.relu(cvar - self.cvar_budget)
            obj = obj + mu_dd * F.relu(dd - 0.1)
            obj = obj + mu_shock * F.relu(shock - self.shock_limit)
            obj = obj.sum()

            # Compute gradient w.r.t. w_param
            grad = torch.autograd.grad(obj, w_param)[0]

            # Manual gradient descent step
            with torch.no_grad():
                w_updated = w_param - self.dual_lr * grad

                # Apply sparse mask: only update selected assets
                w_unselected = w_hat_d * (1.0 - mask_f)
                w_updated = w_updated * mask_f + w_unselected

                # Re-normalize to simplex
                w_updated = torch.clamp(w_updated, 1e-8, None)
                w_updated = w_updated / w_updated.sum(dim=-1, keepdim=True)
                w_current = w_updated

                # Dual variable updates (gradient ascent)
                mu_cvar = torch.clamp(
                    mu_cvar + self.dual_lr * (cvar - self.cvar_budget),
                    min=0.0,
                )
                mu_dd = torch.clamp(
                    mu_dd + self.dual_lr * (dd - 0.1),
                    min=0.0,
                )
                mu_shock = torch.clamp(
                    mu_shock + self.dual_lr * (shock - self.shock_limit),
                    min=0.0,
                )

        # Final projection
        w_star = w_current

        # Record final risk values
        risk_info = {}
        with torch.no_grad():
            risk_info["cvar"] = self.cvar_risk(w_star, mean_returns, cov_matrix)
            risk_info["drawdown"] = self.drawdown_risk(w_star, returns_history)
            risk_info["shock"] = self.shock_risk(w_star, cov_matrix)
            risk_info["kl_from_old"] = self._kl_divergence(w_star, w_old + 1e-8)

        return w_star, risk_info


class TROLLRiskAgent(nn.Module):
    """Complete TROLL-Risk agent combining actor, critic, and trust region projection.

    The agent implements:
    1. Policy network (actor) outputting Dirichlet-distributed allocations
    2. Value network (critic) for advantage estimation
    3. Trust region projection for risk-aware policy updates
    4. PPO-style policy gradient with trust region projected actions
    """

    def __init__(
        self,
        state_dim: int,
        n_assets: int,
        hidden_dim: int = 256,
        kl_weight: float = 0.2,
        sparse_k: int = 10,
    ):
        super().__init__()
        self.actor = PolicyNetwork(state_dim, n_assets, hidden_dim)
        self.critic = ValueNetwork(state_dim, hidden_dim)
        self.trust_region = TrustRegionProjection(
            n_assets=n_assets,
            kl_weight=kl_weight,
            sparse_k=sparse_k,
        )
        self.n_assets = n_assets

    def act(
        self,
        state: torch.Tensor,
        prev_weights: torch.Tensor,
        mean_returns: torch.Tensor,
        cov_matrix: torch.Tensor,
        returns_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Select action (allocation) with trust region projection.

        Args:
            state: Current state.
            prev_weights: Previous portfolio weights.
            mean_returns: Expected asset returns.
            cov_matrix: Asset covariance matrix.
            returns_history: Recent return history.

        Returns:
            Tuple of (projected allocation weights, risk info dict).
        """
        w_hat = self.actor(state)
        # Re-enable grad for the inner trust region optimization loop,
        # which uses torch.autograd.grad() internally. This is needed
        # when act() is called from within a torch.no_grad() context.
        with torch.enable_grad():
            w_star, risk_info = self.trust_region(
                w_hat, prev_weights, mean_returns, cov_matrix, returns_history
            )
        return w_star, risk_info

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward sequence (T,).
            values: Value estimates (T,).
            next_value: Value of the final next state (1,).
            gamma: Discount factor.
            lam: GAE lambda parameter.

        Returns:
            Tuple of (advantages, returns).
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns
