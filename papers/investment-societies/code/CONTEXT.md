# Domain Glossary: Investment Societies of Thought
# 领域术语表：投资思维社会

## Key Financial Terms / 关键金融术语

| Term | English Definition | 中文定义 |
|------|-------------------|---------|
| Investment Thesis | A structured argument supporting or opposing a specific investment position. | 支持或反对特定投资头寸的结构化论点。 |
| Bull Case | The optimistic analytical perspective emphasizing positive catalysts and growth opportunities. | 强调积极催化剂和增长机会的乐观分析视角。 |
| Bear Case | The pessimistic analytical perspective emphasizing risks, threats, and valuation concerns. | 强调风险、威胁和估值担忧的悲观分析视角。 |
| Conviction Level | The degree of confidence in an investment recommendation (high, medium, low). | 对投资建议的信心程度（高、中、低）。 |
| Expected Calibration Error (ECE) | Measures how well a model's confidence estimates align with actual correctness rates. | 衡量模型置信度估计与实际正确率的对齐程度。 |
| Cross-Asset Analysis | Evaluating reasoning quality across equities, fixed income, commodities, and foreign exchange. | 跨股票、固收、商品和外汇评估推理质量。 |
| Portfolio Drawdown | The decline in portfolio value from its peak, measured as a percentage. | 组合价值从峰值下降的幅度，以百分比衡量。 |
| CAPM | Capital Asset Pricing Model — estimates required return given systematic risk (beta). | 资本资产定价模型——根据系统性风险（beta）估计所需收益。 |
| Risk-Adjusted Return | Return metric accounting for the level of risk taken (e.g., Sharpe, Calmar ratios). | 考虑所承担风险水平的收益指标（如夏普比率、卡尔玛比率）。 |

## Key ML Terms / 关键机器学习术语

| Term | English Definition | 中文定义 |
|------|-------------------|---------|
| Societies of Thought | Framework showing reasoning models internally simulate multiple specialized perspectives. | 展示推理模型在内部模拟多个专业视角的框架。 |
| Multi-Expert Simulation | A single model's internal simulation of distinct analytical expert roles during reasoning. | 单个模型在推理过程中对不同的分析专家角色的内部模拟。 |
| Perspective Variance | Measure of internal hidden state variability across reasoning chain positions. | 推理链位置间内部隐藏状态变异性的度量。 |
| Attention Head Clustering | Grouping attention heads by activation correlation to identify specialized functional roles. | 通过激活相关性对注意力头进行分组以识别专业化功能角色。 |
| Causal Intervention | Zeroing out specific attention head outputs to test functional responsibility of clusters. | 将特定注意力头的输出置零以测试聚类的功能责任。 |
| Investment Scaffolding | Structured prompting that explicitly organizes reasoning along bull/bear/risk/macro dimensions. | 显式地沿多头/空头/风险/宏观维度组织推理的结构化提示。 |
| Self-Play Debate | Training phase where the model argues both sides of an investment thesis before synthesizing. | 模型在综合之前论证投资论点正反两面的训练阶段。 |
| Diversity Reward | RL bonus encouraging the model to address multiple analytical perspectives before converging. | 鼓励模型在收敛前考虑多个分析视角的强化学习奖励。 |
| SFT | Supervised Fine-Tuning — training on labeled examples (here: expert debate transcripts). | 有监督微调——在标注样本上训练（此处为专家辩论记录）。 |
| Expert Routing | Learned soft assignment of tokens to specialized expert representations. | 将token学习到的软分配到专业专家表示。 |

## Variable Naming Conventions / 变量命名规范

| Variable | Meaning | Type |
|----------|---------|------|
| `expert_weights` | Soft routing weights assigning tokens to expert roles | Tensor (batch, seq, n_experts) |
| `expert_representations` | Per-expert aggregated hidden representations | Tensor (batch, n_experts, hidden) |
| `perspective_variance` | Variance of cosine distances between consecutive hidden states | float |
| `n_experts` | Number of expert roles (default 4: Bull, Bear, Risk, Macro) | int |
| `n_heads` | Number of attention heads in transformer layers | int |
| `conviction` | Investment conviction level (high/medium/low) | Conviction enum |
| `asset_class` | Asset category: equity, fixed_income, commodity, foreign_exchange | AssetClass enum |
| `diversity_score` | Fraction of expert perspectives covered in reasoning text (0-1) | float |
| `alpha` | Weight for diversity bonus in RL reward (default 0.3) | float |

## Data Format Descriptions / 数据格式描述

| Data Object | Description |
|-------------|-------------|
| `InvestmentDebate` | Dataclass: debate_id, thesis, asset_class, ticker, statements (list of ExpertStatement), final_recommendation, conviction |
| `ExpertStatement` | Dataclass: role (ExpertRole), text, confidence (0-1), key_arguments (list of str) |
| `FinancialQA` | Dataclass: qa_id, question, reasoning, answer, difficulty (easy/medium/hard), asset_class |
| `THESIS_TEMPLATES` | Dict: AssetClass -> list of thesis template strings with {ticker} placeholders |
| `EXPERT_ARGUMENTS` | Dict: ExpertRole -> list of argument template strings for each expert perspective |
| `DatasetConfig` | Configuration: n_debates_per_asset, n_qa_per_asset, seed, tickers dict |
