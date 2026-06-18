# CONTEXT.md — Domain Glossary / 领域术语表
## SAM-Vol: Segment Anything for Volatility Surface Reconstruction

---

### Financial Terms / 金融术语

| Term | English | 中文 |
|------|---------|------|
| Implied volatility (IV) | The volatility implied by the market price of an option via Black-Scholes | 隐含波动率 — 通过Black-Scholes模型从期权市场价格反推的波动率 |
| Volatility surface | The mapping (strike, maturity) -> implied volatility for a given underlier | 波动率曲面 — 给定标的资产的(行权价, 到期日)到隐含波动率的映射 |
| Log-moneyness (k) | log(K/S) — log ratio of strike to spot price | 对数货币性 — 行权价与标的价格比值的对数 |
| SVI | Stochastic Volatility Inspired — 5-parameter parametric vol surface model | SVI参数化 — 5参数波动率曲面参数化模型 |
| SSVI | Surface SVI — full-surface extension of SVI with no-arbitrage constraints | 曲面SVI — SVI的全曲面扩展，含无套利约束 |
| Butterfly spread | d^2C/dK^2 >= 0 — ensures positive risk-neutral density (no static arb) | 蝶式价差 — 确保风险中性密度非负（无静态套利） |
| Calendar spread | d(T*sigma^2)/dT >= 0 — total variance increases with maturity | 日历价差 — 总方差随到期日增加 |
| Put-call parity | C - P = S*e^(-qT) - K*e^(-rT) — fundamental options pricing identity | 看涨看跌平价 — 期权定价基本恒等式 |
| Greeks (delta, gamma, vega) | Sensitivities of option price to underlying, volatility, etc. | 希腊字母 — 期权价格对标的、波动率等的敏感度 |
| Heston model | Stochastic volatility model with mean-reverting variance | Heston模型 — 具有均值回复方差的随机波动率模型 |
| SABR model | Stochastic Alpha Beta Rho — vol smile model popular in FX | SABR模型 — 外汇领域流行的波动率微笑模型 |
| Rough Bergomi | Rough volatility model using fractional Brownian motion | Rough Bergomi模型 — 使用分数布朗运动的粗糙波动率模型 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习 |
| Zero-shot transfer | Applying a model to unseen domains without fine-tuning | 零样本迁移 — 无需微调将模型应用于未见领域 |

### ML / Model Terms / 机器学习术语

| Term | English | 中文 |
|------|---------|------|
| Point-Transformer | Self-attention architecture for unordered point sets | Point-Transformer — 用于无序点集的自注意力架构 |
| Neural implicit surface | Continuous function mapping coordinates to values (here: IV) | 神经隐式曲面 — 将坐标映射到值的连续函数（此处为隐含波动率） |
| Fourier features | Sinusoidal encoding of input coordinates for better high-frequency learning | 傅里叶特征 — 输入坐标的正弦编码，改善高频学习 |
| Local feature interpolation | Attention-weighted interpolation of nearby point features at query locations | 局部特征插值 — 在查询位置对邻近点特征进行注意力加权插值 |
| No-arbitrage regularizer | Differentiable penalty enforcing butterfly + calendar spread constraints | 无套利正则化器 — 强制执行蝶式和日历价差约束的可微分惩罚 |

### Variable Naming Conventions / 变量命名约定

- `point_cloud`: `(B, N, 7)` — sparse option quotes: (log_m, tau, iv, spread, price, delta, gamma)
- `mask`: `(B, N)` — binary mask for valid (non-padded) quotes
- `eval_k`: `(B, n_k)` — log-moneyness evaluation grid
- `eval_tau`: `(B, n_tau)` — maturity evaluation grid
- `iv_surface`: `(B, n_k, n_tau)` — predicted implied volatility surface
- `gt_surface`: `(B, n_k, n_tau)` — ground-truth implied volatility surface
- `global_ctx`: `(B, D)` — global context vector from point encoder

### Data Format / 数据格式

- **Synthetic surfaces**: Generated using Heston, SABR, and SVI models with randomized parameters. Each surface is a `(n_strikes, n_maturities)` grid.
- **Sparse quotes**: Sampled from full surfaces with observation noise proportional to moneyness distance from ATM. Each quote is an `OptionQuote` dataclass with strike, maturity, implied_vol, bid_ask_spread, call_price, underlying_price.
- **Batch format**: Dict with keys `point_cloud`, `mask`, `eval_k`, `eval_tau`, `gt_surface`, `underlying_price`.
- **Split**: 70% train / 15% validation / 15% test by surface index (different random seeds for each split).
