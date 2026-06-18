# Domain Glossary: Financial Deep Ignorance
# 领域术语表：金融深度无知

## Key Financial Terms / 关键金融术语

| Term | English Definition | 中文定义 |
|------|-------------------|---------|
| Market Manipulation | Techniques for artificially distorting market prices or volumes (spoofing, layering, wash trading). | 人为扭曲市场价格或交易量的技术（幌骗、分层、对倒交易）。 |
| Insider Trading | Trading based on Material Non-Public Information (MNPI) before it becomes publicly available. | 基于重大非公开信息（MNPI）在信息公开前进行的交易。 |
| Sanctions Evasion | Techniques for circumventing international sanctions regimes (shell companies, jurisdiction arbitrage). | 规避国际制裁制度的技术（空壳公司、司法管辖区套利）。 |
| Spoofing | Placing large orders with no intention to execute, to create false supply/demand signals. | 下大额订单但无意执行，以制造虚假的供需信号。 |
| Layering | A graduated form of spoofing across multiple price levels in the order book. | 在订单簿多个价格层级进行的渐进式幌骗。 |
| Wash Trading | Executing trades between accounts under common control to create false volume. | 在共同控制的账户之间执行交易以制造虚假交易量。 |
| Front-Running | Trading ahead of known large client orders to profit from the anticipated price movement. | 在已知的大额客户订单之前交易，以从预期价格变动中获利。 |
| Structuring | Breaking transactions into smaller amounts to avoid regulatory reporting thresholds. | 将交易拆分为较小金额以规避监管报告阈值。 |
| KYC / AML | Know-Your-Customer / Anti-Money-Laundering regulatory requirements. | 了解你的客户 / 反洗钱监管要求。 |
| Dual-Use Knowledge | Information that can serve both legitimate analytical and dangerous exploitation purposes. | 既可用于合法分析又可用于危险目的的信息。 |

## Key ML Terms / 关键机器学习术语

| Term | English Definition | 中文定义 |
|------|-------------------|---------|
| Deep Ignorance | A pretraining data filtering paradigm that prevents models from ever learning dangerous capabilities. | 一种预训练数据筛选范式，防止模型学习到危险能力。 |
| Tamper Resistance | The model's ability to resist adversarial fine-tuning and jailbreak attacks that attempt to elicit filtered knowledge. | 模型抵抗对抗性微调和越狱攻击以提取被筛选知识的能力。 |
| Three-Stage Pipeline | Filtering pipeline: (1) keyword matching, (2) semantic classification, (3) human expert review. | 筛选流水线：（1）关键词匹配，（2）语义分类，（3）人工专家审核。 |
| RLHF | Reinforcement Learning from Human Feedback — post-training alignment via human preference signals. | 人类反馈强化学习——通过人类偏好信号进行后训练对齐。 |
| Constitutional AI | Automated alignment using a set of principles for self-critique rather than human feedback. | 使用一组原则进行自我批评的自动对齐方法，而非人类反馈。 |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning method using low-rank weight matrices. | 低秩适应——使用低秩权重矩阵的参数高效微调方法。 |
| Attack Vector | A specific method used to probe or bypass a model's safety mechanisms. | 用于探测或绕过模型安全机制的特定方法。 |
| False Positive Rate | The fraction of safe documents incorrectly flagged as dangerous by the filter. | 被过滤器错误标记为危险的安全文档比例。 |

## Variable Naming Conventions / 变量命名规范

| Variable | Meaning | Type |
|----------|---------|------|
| `doc_id` | Unique document identifier (e.g., "doc_000042") | str |
| `category` | Danger taxonomy category (6 categories) | DangerCategory enum |
| `label` | Document classification: safe, dangerous, or ambiguous | DocumentLabel enum |
| `severity` | Filtering strictness: aggressive, moderate, or conservative | SeverityLevel enum |
| `stage1_flagged` | Whether keyword matching flagged this document | bool |
| `stage2_score` | Probability of being dangerous from semantic classifier | float [0,1] |
| `final_decision` | Pipeline outcome: "keep", "remove", or "redact" | str |
| `danger_score` | Classifier's probability for the "dangerous" class | float [0,1] |
| `n_classes` | Number of classification labels (3: safe, dangerous, ambiguous) | int |

## Data Format Descriptions / 数据格式描述

| Data Object | Description |
|-------------|-------------|
| `FinancialDocument` | Dataclass with doc_id, text, category, label, source_type, severity, keywords_found |
| `FilterResult` | Dataclass with per-stage filtering decisions and final outcome |
| `DOCUMENT_TEMPLATES` | Nested dict: DangerCategory -> {"safe": [templates], "dangerous": [templates]} |
| `CATEGORY_SEVERITY` | Dict mapping DangerCategory to SeverityLevel for filtering strictness |
| `DatasetConfig` | Configuration for synthetic dataset generation (doc counts, augment factor, seed) |
