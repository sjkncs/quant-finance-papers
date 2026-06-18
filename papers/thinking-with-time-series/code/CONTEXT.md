# CONTEXT.md — Domain Glossary / 领域术语表
## Thinking with Time-Series (TTS)

---

### Financial Terms / 金融术语

| Term | English | 中文 |
|------|---------|------|
| OHLCV | Open, High, Low, Close, Volume — standard bar data for a financial instrument | 开盘价、最高价、最低价、收盘价、成交量 — 金融工具的标准K线数据 |
| VIX | CBOE Volatility Index — measures implied volatility of S&P 500 options | CBOE波动率指数 — 衡量S&P 500期权的隐含波动率 |
| Regime (market) | A persistent market state (bull, bear, crisis) with distinct statistical properties | 市场制度 — 具有独特统计特征的持续市场状态（牛市、熊市、危机） |
| Cross-asset correlation | The correlation structure between returns of different assets | 跨资产相关性 — 不同资产收益之间的相关结构 |
| Tail risk | The probability and impact of extreme market moves (typically > 2 std deviations) | 尾部风险 — 极端市场波动的概率和影响（通常 > 2个标准差） |
| Calmar ratio | Annualized return / maximum drawdown — risk-adjusted performance metric | 卡尔马比率 — 年化收益/最大回撤，风险调整后的绩效指标 |
| Drawdown | Peak-to-trough decline in portfolio value | 回撤 — 投资组合价值从峰值到谷底的下降 |
| Self-consistency | Ensemble voting method: generate multiple reasoning paths, pick majority answer | 自一致性 — 集成投票方法：生成多个推理路径，选择多数答案 |

### ML / Model Terms / 机器学习术语

| Term | English | 中文 |
|------|---------|------|
| DDPM | Denoising Diffusion Probabilistic Model — iterative denoising for generation | 去噪扩散概率模型 — 通过迭代去噪进行生成 |
| Temporal U-Net | U-Net architecture with temporal self-attention for time-series processing | 时序U-Net — 具有时序自注意力的U-Net架构 |
| FiLM conditioning | Feature-wise Linear Modulation — injecting conditioning via scale and bias | FiLM条件化 — 通过缩放和偏置注入条件信息 |
| HMM | Hidden Markov Model — latent state model with observable emissions | 隐马尔可夫模型 — 具有可观测发射的隐状态模型 |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning method | 低秩适应 — 参数高效的微调方法 |
| Trajectory generation | Producing synthetic time-series paths via diffusion sampling | 轨迹生成 — 通过扩散采样生成合成时间序列路径 |

### Variable Naming Conventions / 变量命名约定

- `market_state`: `(B, lookback, n_assets)` — log-return series for the observation window
- `vix`: `(B,)` — VIX level per batch item
- `query_emb`: `(B, query_dim)` — encoded natural-language query
- `cond`: `(B, cond_dim)` — conditioning vector from the query encoder
- `trajectories` / `traj_stack`: `(B, N, T, A)` — N generated trajectories, T timesteps, A assets
- `logits`: `(B, 3)` — answer classification logits (negative/neutral/positive)
- `regime`: int in {0, 1, 2} — market regime (low-vol/bull, high-vol/bear, crisis)

### Data Format / 数据格式

- **Synthetic market data**: Generated via geometric Brownian motion with regime switching and GARCH-like volatility clustering. Stored as NumPy arrays `(n_days, n_assets)`.
- **MarketThinkBench scenarios**: JSON-serializable `MarketScenario` dataclass with query text, asset list, horizon, regime label, ground truth, difficulty, and category.
- **Batch format**: Dict with keys `market_state`, `future`, `vix`, `query_emb`, `ground_truth`, `category`, `horizon_days`.
