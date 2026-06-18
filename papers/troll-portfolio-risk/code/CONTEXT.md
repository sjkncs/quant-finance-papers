# Domain Glossary: TROLL-Risk Portfolio Optimization
# 领域术语表：TROLL-Risk组合优化

## Key Financial Terms / 关键金融术语

| Term | English Definition | 中文定义 |
|------|-------------------|---------|
| Trust Region | A constrained region in parameter space within which policy updates are permitted, ensuring stable learning. | 参数空间中的约束区域，策略更新被限制在该区域内以确保稳定学习。 |
| Portfolio Risk | The probability and magnitude of financial loss in a portfolio, measured via metrics like CVaR, drawdown, and volatility. | 组合中财务损失的概率和幅度，通过CVaR、回撤和波动率等指标衡量。 |
| Max Drawdown (MDD) | The largest peak-to-trough decline in portfolio value over a specified period. | 在特定时期内组合价值从峰值到谷底的最大跌幅。 |
| CVaR (Conditional Value-at-Risk) | The expected loss given that the loss exceeds the Value-at-Risk threshold (e.g., 95th percentile). | 在损失超过风险价值阈值（如第95百分位）条件下的期望损失。 |
| Sharpe Ratio | Risk-adjusted return metric: (portfolio return - risk-free rate) / portfolio volatility. | 风险调整收益指标：（组合收益 - 无风险利率）/ 组合波动率。 |
| Calmar Ratio | Annualized return divided by maximum drawdown, measuring risk-adjusted performance. | 年化收益除以最大回撤，衡量风险调整后的表现。 |
| Portfolio Allocation Simplex | The set of all valid weight vectors: non-negative weights summing to 1. | 所有有效权重向量的集合：非负权重且总和为1。 |
| KL Divergence | Kullback-Leibler divergence measuring the difference between two probability distributions. | KL散度衡量两个概率分布之间的差异。 |
| Correlation Shock Exposure | Expected portfolio loss when asset correlations spike from normal to crisis levels. | 当资产相关性从正常水平飙升至危机水平时的预期组合损失。 |
| Soft Drawdown Penalty | A smooth (differentiable) approximation of maximum drawdown using exponential utility. | 使用指数效用函数对最大回撤的光滑（可微分）近似。 |
| Rebalancing | The process of adjusting portfolio weights at discrete time intervals. | 在离散时间间隔调整组合权重的过程。 |
| Transaction Cost | The cost incurred when executing trades, proportional to portfolio turnover. | 执行交易时产生的成本，与组合换手率成正比。 |

## Key ML Terms / 关键机器学习术语

| Term | English Definition | 中文定义 |
|------|-------------------|---------|
| TROLL | Trust Region Optimization with Learnable Limits — a framework replacing PPO clipping with differentiable trust region projections. | 可学习约束的信任区域优化——用可微分信任区域投影替代PPO裁剪的框架。 |
| Discrete Trust Regions | Trust region constraints formulated as differentiable optimization problems solvable in closed form. | 信任区域约束被表述为可闭式求解的可微分优化问题。 |
| PPO (Proximal Policy Optimization) | A policy gradient method using clipped surrogate objective to limit policy update magnitude. | 使用裁剪代理目标函数限制策略更新幅度的策略梯度方法。 |
| GAE (Generalized Advantage Estimation) | A method for computing advantage estimates with bias-variance trade-off controlled by lambda parameter. | 一种通过lambda参数控制偏差-方差权衡来计算优势估计的方法。 |
| Sparse Asset Selection | Restricting the trust region projection to only the K assets with the largest proposed weight changes. | 将信任区域投影限制在仅具有最大提议权重变化的K个资产上。 |
| Augmented Lagrangian | An optimization method combining Lagrangian multipliers with penalty terms for constrained optimization. | 结合拉格朗日乘子和惩罚项的约束优化方法。 |
| Policy Collapse | A training failure mode where the policy degenerates, producing near-zero Sharpe ratios. | 训练失败模式，策略退化导致夏普比率接近零。 |

## Variable Naming Conventions / 变量命名规范

| Variable | Meaning | Type |
|----------|---------|------|
| `w_hat` | Unconstrained candidate allocation from policy network | Tensor (batch, n_assets) |
| `w_old` | Previous period's allocation weights | Tensor (batch, n_assets) |
| `w_star` | Projected allocation after trust region optimization | Tensor (batch, n_assets) |
| `mu_cvar`, `mu_dd`, `mu_shock` | Dual variables for risk constraint Lagrangian | Tensor (batch,) |
| `n_assets` | Total number of assets in the portfolio | int |
| `sparse_k` | Number of assets selected for sparse projection | int |
| `kl_weight` | Weight (lambda) for the KL divergence penalty term | float |
| `state_dim` | Dimension of the state vector input to the policy | int |
| `hidden_dim` | Hidden layer dimension in MLP networks | int |
| `episode` | One complete training episode (one year of daily steps) | int |

## Data Format Descriptions / 数据格式描述

| Data Object | Shape | Description |
|-------------|-------|-------------|
| `returns` | (n_days, n_assets) | Daily asset returns, regime-switching with fat tails |
| `prices` | (n_days+1, n_assets) | Asset prices computed from returns, starting at 100.0 |
| `regimes` | (n_days,) | Binary regime indicators: 0=normal, 1=crisis |
| `state_vector` | (state_dim,) | Concatenated: lookback returns + rolling vols + weights + stats |
| `cov_matrix` | (n_assets, n_assets) | Rolling covariance matrix from 60-day window |
| `hist_tensor` | (1, lookback, n_assets) | Recent return history for drawdown computation |
