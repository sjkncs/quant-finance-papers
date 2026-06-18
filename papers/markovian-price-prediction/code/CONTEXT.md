# CONTEXT.md — Domain Glossary / 领域术语表
## Markovian Multi-Resolution Forecasting (MMRF)

---

### Financial Terms / 金融术语

| Term | English | 中文 |
|------|---------|------|
| Resolution level | A temporal granularity for financial data (tick, minute, hourly, daily, weekly) | 分辨率级别 — 金融数据的时间粒度（tick、分钟、小时、日、周） |
| Directional Accuracy (DA) | Percentage of correct sign predictions (up/down) | 方向准确率 — 正确预测涨跌方向的百分比 |
| MASE | Mean Absolute Scaled Error — forecast error normalized by naive forecast | 平均绝对标度误差 — 以朴素预测为基准归一化的预测误差 |
| Sharpe ratio | Excess return / volatility — risk-adjusted performance metric | 夏普比率 — 超额收益/波动率，风险调整后的绩效指标 |
| VWAP | Volume-Weighted Average Price — average price weighted by volume | 成交量加权平均价格 |
| Order flow imbalance | Difference between buy and sell order volumes | 订单流失衡 — 买卖订单量的差异 |
| GARCH | Generalized AutoRegressive Conditional Heteroskedasticity — volatility model | 广义自回归条件异方差 — 波动率模型 |
| Autocorrelation decay | How quickly serial correlation diminishes with lag | 自相关衰减 — 序列相关性随滞后阶数减弱的速度 |
| Cross-asset eigenvalues | Eigenvalues of the correlation matrix — measure of market co-movement | 跨资产特征值 — 相关矩阵的特征值，衡量市场联动程度 |

### ML / Model Terms / 机器学习术语

| Term | English | 中文 |
|------|---------|------|
| Markovian window | Sliding window over resolution levels (not tokens) — only the w most recent resolutions use full tokens | 马尔可夫窗口 — 仅在最近w个分辨率级别使用完整token序列 |
| Compressed summary | Fixed-size statistical vector (d_c=30) replacing full token sequences for distant resolutions | 压缩摘要 — 替代远距离分辨率完整token序列的固定大小统计向量 |
| Resolution tokenizer | Per-resolution linear projection + positional encoding + MLP for embedding raw features | 分辨率分词器 — 将原始特征嵌入为token的每分辨率独立模块 |
| Cross-resolution transformer | Shared transformer processing tokens from all resolutions within the Markovian window | 跨分辨率Transformer — 处理马尔可夫窗口内所有分辨率token的共享Transformer |
| Financial Markov property | The property that returns at resolution s depend mainly on the w nearest resolutions | 金融马尔可夫性质 — 分辨率s的收益主要取决于最近w个分辨率 |

### Variable Naming Conventions / 变量命名约定

- `tokens`: Dict[str, Tensor] — `{resolution_name: (B, seq_len, n_assets * n_features)}`
- `compressed_summaries`: `(B, n_resolutions, 30)` — statistical summaries for all resolutions
- `predictions`: Dict[str, Tensor] — `{resolution_name: (B,) predicted returns}`
- `res_name`: string — one of `"tick"`, `"minute"`, `"hourly"`, `"daily"`, `"weekly"`
- `w` / `window_width`: int — Markovian sliding window width (number of recent resolutions with full tokens)
- `d_model`: int — transformer hidden dimension

### Data Format / 数据格式

- **Multi-resolution data**: Generated from daily log-returns via aggregation. Each resolution level produces `(n_periods, n_assets, n_features)` arrays.
- **Compressed summaries**: 30-dimensional vectors with rolling statistics (8d), autocorrelation (10d), cross-asset eigenvalues (5d), volatility regime (4d), and trend features (3d).
- **Batch format**: Dict with keys `tokens_{res_name}` (per resolution), `compressed_summaries`, and `target_{res_name}` (next-period return).
- **Chronological split**: 70% train / 15% validation / 15% test by time (no future leakage).
