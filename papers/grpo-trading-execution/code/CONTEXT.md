# CONTEXT.md - Domain Glossary and Code Conventions
# GRPO Trading Execution Paper / GRPO交易执行论文

---

## Key Financial Terms / 关键金融术语

| Term (EN) | Term (CN) | Definition |
|-----------|-----------|------------|
| Implementation Shortfall (IS) | 实现缺口 | The difference between the actual execution price and the decision-price benchmark (arrival price). Measured in basis points (bps). The primary cost metric for trade execution. |
| TWAP (Time-Weighted Average Price) | 时间加权平均价格 | A baseline execution strategy that divides an order equally across time intervals. |
| VWAP (Volume-Weighted Average Price) | 成交量加权平均价格 | An execution strategy that follows the historical intraday volume profile. |
| Execution Cost | 执行成本 | The total cost of executing a parent order, including market impact, spread crossing, and slippage. |
| Market Impact | 市场冲击 | The price effect caused by the execution activity itself, split into temporary and permanent components. |
| Temporary Impact | 临时冲击 | Short-lived price displacement that decays after execution. Modeled as proportional to sqrt(participation_rate). |
| Permanent Impact | 永久冲击 | Persistent price change caused by information leakage from execution. Modeled as linear in participation rate. |
| Slippage | 滑点 | The difference between the expected execution price and the actual fill price. |
| Bid-Ask Spread | 买卖价差 | The difference between the best ask and best bid prices. Measured in bps. |
| Order Book Depth | 订单簿深度 | The quantity available at the best bid/ask price levels. |
| Participation Rate | 参与率 | The ratio of the order size to the available liquidity. Higher participation = more market impact. |
| Sharpe Ratio | 夏普比率 | Risk-adjusted return metric: (return - risk_free_rate) / volatility. Used here for execution alpha. |
| Maximum Drawdown | 最大回撤 | The worst peak-to-trough decline in cumulative execution cost across orders. |
| Fill Rate | 成交率 | The fraction of the parent order that is actually executed. |
| Aggressiveness | 激进度 | A parameter in [0,1] controlling execution style: 0 = passive limit order, 1 = immediate market order. |
| VIX | 恐慌指数 | CBOE Volatility Index; used as a proxy for market volatility regime. |

## Key ML Terms / 关键机器学习术语

| Term (EN) | Term (CN) | Definition |
|-----------|-----------|------------|
| GRPO (Group Relative Policy Optimization) | 群组相对策略优化 | An RL algorithm that normalizes returns within trajectory groups, eliminating the need for a learned value function. |
| Process Reward Model (PRM) | 过程奖励模型 | A model that assigns rewards to intermediate steps rather than only the final outcome. |
| ExecPRM | 执行过程奖励模型 | The formal equivalence between GRPO and PRM in the trade execution setting (this paper's contribution). |
| Lambda Normalization (λ-normalization) | λ归一化 | Dividing each advantage term by its process set cardinality to correct frequency imbalance. |
| Prefix Grouping | 前缀分组 | Grouping trajectories by shared action prefixes to compute step-level process rewards. |
| Process Set Λ(i,t) | 过程集 | The set of trajectories sharing the same action prefix as trajectory i up to step t. |
| Trajectory | 轨迹 | A complete execution schedule from order arrival to completion: sequence of (state, action, reward). |
| Child Order | 子订单 | A single execution decision within a parent order's execution trajectory. |
| KL Penalty | KL散度惩罚 | Regularization term preventing the policy from diverging too far from the reference policy. |
| Advantage | 优势 | Group-relative normalized return: (R_i - mean(R)) / std(R). |

## Variable Naming Conventions / 变量命名约定

- `state_dim=14`: Dimension of the state vector (remaining shares, time, price, volumes, spread, depth, vol, VIX)
- `action_dim=2`: Action space dimension (shares_fraction, aggressiveness)
- `G`: Number of sampled trajectories per parent order (default 16)
- `T`: Number of execution intervals per trading day (default 48, five-minute intervals)
- `Q`: Total parent order size in shares
- `q_t`: Remaining shares at time t
- `p_0`: Arrival mid-price (benchmark for implementation shortfall)
- `lambda_factor`: The 1/|Λ(i,t)| normalization weight
- `cardinalities[t][i]`: Process set size for trajectory i at step t
- `impl_shortfall` / `is_bps`: Implementation shortfall in basis points

## Data Format Descriptions / 数据格式说明

- **Orders DataFrame**: Columns: `order_id`, `asset_id`, `size_usd`, `is_buy`, `horizon_intervals`, `arrival_price`, `vix_level`, `daily_volume`
- **MarketState**: Dataclass with `mid_price`, `spread`, `bid_depth` (array[3]), `ask_depth` (array[3]), `recent_volume` (array[5]), `realized_vol`, `vix_level`
- **Trajectory**: List of `TrajectoryStep` objects, each containing `state` (ndarray[14]), `action` (ndarray[2]), `log_prob` (float), `reward` (float)
- **ExecutionResult**: `executed_shares`, `avg_price`, `market_impact_bps`, `slippage_bps`

---

## 中文补充说明

本文代码实现了一个合成市场模拟器，用于训练和评估λ-ExecGRPO交易执行策略。核心流程为：(1) 生成合成机构订单，(2) 在模拟市场环境中采样G条执行轨迹，(3) 按共享前缀分组计算过程集基数，(4) 使用λ归一化的GRPO损失更新策略网络。关键设计决策包括使用U型日内成交量分布模拟真实的开盘/收盘交易活跃度模式，以及使用平方根市场冲击模型（Almgren-Chriss风格）计算临时冲击。
