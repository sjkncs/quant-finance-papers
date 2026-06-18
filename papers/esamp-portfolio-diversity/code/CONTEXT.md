# CONTEXT.md - Domain Glossary and Code Conventions
# ESamp Portfolio Diversity Paper / ESamp投资组合多样性论文

---

## Key Financial Terms / 关键金融术语

| Term (EN) | Term (CN) | Definition |
|-----------|-----------|------------|
| Portfolio Diversification | 投资组合分散化 | Spreading investments across diverse assets/strategies to reduce risk concentration. |
| Mean-Variance Optimization | 均值-方差优化 | Classical Markowitz framework maximizing expected return for a given risk level. |
| Black-Litterman Model | Black-Litterman模型 | Bayesian extension of mean-variance incorporating investor views as tilts to equilibrium returns. |
| Risk Parity | 风险平价 | Allocation strategy equalizing risk contributions across assets rather than capital weights. |
| Sharpe Ratio | 夏普比率 | Risk-adjusted return: (annualized_return - risk_free_rate) / annualized_volatility. |
| Maximum Drawdown (MaxDD) | 最大回撤 | Largest peak-to-trough decline in portfolio value. Key downside risk metric. |
| Calmar Ratio | 卡尔玛比率 | Annualized return divided by absolute maximum drawdown. |
| Weight-Space Entropy | 权重空间熵 | Shannon entropy of portfolio weight distribution, measuring allocation diversification. Higher = more uniform. |
| Strategy Coverage | 策略覆盖度 | Number of distinct strategic clusters found by k-means on weight vectors. |
| Efficient Frontier | 有效前沿 | Set of portfolios offering maximum return for each risk level. |
| Factor Loading | 因子载荷 | Sensitivity of an asset's return to a systematic risk factor. |
| Covariance Shrinkage | 协方差收缩 | Ledoit-Wolf regularization pulling sample covariance toward a structured target (e.g., identity). |

## Key ML Terms / 关键机器学习术语

| Term (EN) | Term (CN) | Definition |
|-----------|-----------|------------|
| ESamp (Exploratory Sampling) | 探索性采样 | Framework using latent distiller prediction error as novelty signal to bias sampling toward unexplored regions. |
| PortESamp | PortESamp | This paper's adaptation of ESamp to portfolio strategy generation. |
| Strategy Distiller | 策略蒸馏器 | Lightweight 2-layer MLP (256 hidden) predicting deep representations from shallow features. |
| Latent Distilling | 潜空间蒸馏 | Training a small model to predict a large model's internal representations; prediction error = novelty signal. |
| Novelty Score | 新颖性分数 | L2 norm of prediction error between distiller output and actual deep representation. High = novel strategy. |
| Exploration Strength (beta) | 探索强度β | Hyperparameter controlling tradeoff between performance and novelty in sampling probabilities. Optimal at 0.25. |
| Shallow Features | 浅层特征 | Compact representation from the encoder layer (64-dim), input to both deep processor and distiller. |
| Deep Representation | 深层表示 | Rich internal representation (128-dim) from the deep processing network. |
| Replay Buffer | 经验回放缓冲区 | Storage for (shallow_features, deep_representation) pairs for asynchronous distiller training. |
| Softmax Normalization | Softmax归一化 | Converting portfolio logits to valid probability weights summing to 1. |

## Variable Naming Conventions / 变量命名约定

- `input_dim=50`: Dimension of market feature input (one per asset)
- `feature_dim=64`: Shallow feature dimension after encoder
- `deep_dim=128`: Deep representation dimension after processor
- `distiller_hidden=256`: Hidden dimension of the Strategy Distiller MLP
- `num_assets`: Number of assets in the portfolio universe (default 50 for experiments)
- `beta`: Exploration strength hyperparameter (default 0.25)
- `noise_scale`: Gaussian noise injected into shallow features for diverse proposals (default 0.15)
- `num_proposals=64`: Number of candidate proposals generated (P)
- `num_select=8`: Number of portfolios selected from proposals (K)
- `H`: Weight-space entropy (diversity metric)

## Data Format Descriptions / 数据格式说明

- **Returns DataFrame**: Shape (num_days, num_assets). Daily asset returns. Generated synthetically with multi-factor model + regime structure.
- **Market Features**: Shape (input_dim,). Concatenation of annualized return, volatility, momentum, short-term stats, and eigenvalue summary.
- **PortfolioProposal**: Dataclass with `weights` (ndarray), `shallow_features` (Tensor), `deep_representation` (Tensor), `novelty_score` (float).
- **MarketRegime**: Named regimes (bull, correction, bear, recovery, bull_late, stagnation, rally) with drift/volatility/correlation parameters.

---

## 中文补充说明

本文代码实现了PortESamp框架：(1) 训练一个组合优化模型（编码器→深度处理器→组合头），(2) 训练轻量级策略蒸馏器从浅层特征预测深层表示，(3) 使用蒸馏器预测误差作为新颖性信号重新加权候选组合的采样概率，(4) 选择新颖性最高的K个组合作为多样化策略输出。合成资产宇宙使用多因子模型生成，包含牛熊交替的市场制度结构。关键设计决策是使用softmax输出组合权重以保证权重非负且和为1，以及使用异步回放缓冲区训练蒸馏器以减少计算开销。
