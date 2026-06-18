# Quantitative Finance Paper Adaptations / 量化金融领域论文创作

> **Adapting cutting-edge AI conference methodologies to quantitative finance**
> **将前沿AI顶会方法论迁移至量化金融领域**

---

## Overview / 概述

This repository contains 9 full-length conference papers that adapt core methodologies from top-tier AI/ML conferences (CVPR, ICML, ICLR) to quantitative finance problems. Each paper is accompanied by a complete, runnable Python implementation.

本仓库包含9篇完整的会议论文，每篇论文将顶会AI/ML核心方法论迁移至量化金融问题。每篇论文均附有完整的、可运行的Python实现代码。

## Paper Collection / 论文列表

| # | Source Paper / 源论文 | Venue / 会议 | Core Method / 核心方法 | Quant Finance Paper / 量化金融论文 | Quant Area / 量化方向 |
|---|---|---|---|---|---|
| 1 | Thinking with Video | CVPR 2026 | Video generation as reasoning / 视频生成推理 | [Thinking with Time-Series](papers/thinking-with-time-series/paper.md) | Multi-horizon market reasoning / 多周期市场推理 |
| 2 | Markovian Scale Prediction | CVPR 2026 | Sliding window multi-scale / 滑动窗口多尺度 | [Markovian Multi-Resolution Forecasting](papers/markovian-price-prediction/paper.md) | Multi-resolution price prediction / 多分辨率价格预测 |
| 3 | SAM 3D | CVPR 2026 | Single-image 3D reconstruction / 单图3D重建 | [SAM-Vol: Segment Anything for Volatility](papers/sam-volatility/paper.md) | Volatility surface reconstruction / 波动率曲面重建 |
| 4 | GRPO is Secretly a PRM | ICML 2026 | Implicit process reward / 隐式过程奖励 | [GRPO for Trading Execution](papers/grpo-trading-execution/paper.md) | Optimal trade execution / 交易执行优化 |
| 5 | ESamp: Latent Distilling | ICML 2026 | Latent exploration sampling / 潜空间探索采样 | [ESamp for Portfolio Diversification](papers/esamp-portfolio-diversity/paper.md) | Portfolio strategy diversity / 组合多样性探索 |
| 6 | CEO-Bench | arXiv 2026 | Long-horizon agent evaluation / 长程Agent评测 | [QuantBench: Long-Horizon Market Regimes](papers/quantbench-long-horizon/paper.md) | Long-horizon trading evaluation / 长周期交易评测 |
| 7 | TROLL Trust Regions | ICLR 2026 | Differentiable trust region / 离散可微信任区域 | [TROLL for Portfolio Risk](papers/troll-portfolio-risk/paper.md) | Portfolio risk management / 组合风险管理 |
| 8 | Deep Ignorance | ICLR 2026 | Pretraining data filtering / 预训练数据防篡改 | [Financial Deep Ignorance](papers/financial-deep-ignorance/paper.md) | Financial model safety / 金融模型安全 |
| 9 | Societies of Thought | Google 2026 | Internal multi-agent reasoning / 内部多智能体推理 | [Investment Societies of Thought](papers/investment-societies/paper.md) | Multi-expert investment reasoning / 多专家投资决策 |

## Repository Structure / 仓库结构

```
quant-finance-papers/
├── README.md                              # This file / 本文件
├── papers/
│   ├── thinking-with-time-series/         # Thinking with Video → Multi-horizon market reasoning
│   │   ├── paper.md                       # Full paper (~7,400 words)
│   │   └── code/
│   │       ├── main.py                    # Training & evaluation script
│   │       ├── model.py                   # Diffusion-based trajectory generator
│   │       ├── data.py                    # MarketThinkBench synthetic data
│   │       └── requirements.txt
│   ├── markovian-price-prediction/        # Markovian Scale Prediction → Price forecasting
│   │   ├── paper.md
│   │   └── code/ (main.py, model.py, data.py, requirements.txt)
│   ├── sam-volatility/                    # SAM 3D → Volatility surface
│   │   ├── paper.md
│   │   └── code/ (main.py, model.py, data.py, requirements.txt)
│   ├── grpo-trading-execution/            # GRPO-PRM → Trade execution
│   │   ├── paper.md
│   │   └── code/ (main.py, model.py, data.py, requirements.txt)
│   ├── esamp-portfolio-diversity/         # ESamp → Portfolio diversification
│   │   ├── paper.md
│   │   └── code/ (main.py, model.py, data.py, requirements.txt)
│   ├── quantbench-long-horizon/           # CEO-Bench → Trading agent evaluation
│   │   ├── paper.md
│   │   └── code/ (main.py, model.py, data.py, requirements.txt)
│   ├── troll-portfolio-risk/              # TROLL → Portfolio risk
│   │   ├── paper.md
│   │   └── code/ (main.py, model.py, data.py, requirements.txt)
│   ├── financial-deep-ignorance/          # Deep Ignorance → Financial model safety
│   │   ├── paper.md
│   │   └── code/ (main.py, model.py, data.py, requirements.txt)
│   └── investment-societies/              # Societies of Thought → Investment reasoning
│       ├── paper.md
│       └── code/ (main.py, model.py, data.py, requirements.txt)
└── LICENSE
```

## Domain Adaptation Methodology / 领域迁移方法论

Each paper follows a systematic adaptation pipeline:

每篇论文遵循系统化的迁移流程：

1. **Extract Core Mechanism / 提取核心机制** — Abstract the mathematical/algorithmic essence from the source paper, stripping away domain-specific details.
2. **Map to Quant Finance Problem / 映射量化金融问题** — Identify structurally analogous unsolved or under-explored problems in quantitative finance.
3. **Domain-Specific Constraints / 领域特定约束** — Introduce finance-specific constraints: liquidity constraints, transaction costs, regulatory requirements, non-stationarity of financial time series.
4. **Finance Datasets & Metrics / 金融数据集与指标** — Replace generic benchmarks with financial datasets (S&P 500, options chains, institutional orders) and evaluation metrics (Sharpe ratio, maximum drawdown, Calmar ratio, implementation shortfall).
5. **Bilingual Presentation / 中英双语撰写** — Bridge the ML community and quantitative finance community terminology systems.

## Paper Specifications / 论文规格

Each full paper includes:

- **Bilingual Abstract / 双语摘要** (~300 words each language)
- **Introduction / 引言** with problem motivation, gap analysis, 4-5 contributions
- **Related Work / 相关工作** with 15-35 citations across 2-3 subsections
- **Method / 方法** with formal mathematical definitions, theorems with proof sketches, algorithm pseudocode
- **Experiments / 实验** with main results tables, ablation studies, hyperparameter sensitivity analysis
- **Discussion & Conclusion / 讨论与结论** including limitations and broader impact
- **References / 参考文献** (25-35 per paper)

## Code Specifications / 代码规格

Each implementation includes:

- `data.py` — Synthetic data generation with realistic financial properties (regime switching, intraday volume patterns, multi-factor returns)
- `model.py` — Core PyTorch model architecture with proper docstrings and type hints
- `main.py` — End-to-end training and evaluation script with argparse CLI (supports train/eval/ablation modes)
- `requirements.txt` — Dependencies (numpy, torch, pandas, scipy, scikit-learn)

All code generates synthetic data and runs end-to-end without requiring external datasets.

## Statistics / 统计

| Metric / 指标 | Value / 数值 |
|---|---|
| Total Papers / 论文总数 | 9 |
| Total Words / 总字数 | ~57,800 |
| Total Code Lines / 总代码行数 | ~10,228 |
| Python Files / Python文件数 | 27 (+ 9 requirements.txt) |
| Avg Words per Paper / 每篇均字数 | ~6,400 |
| Avg Citations per Paper / 每篇均引用 | ~28 |

## Quick Start / 快速开始

```bash
# Clone the repository / 克隆仓库
git clone https://github.com/sjkncs/quant-finance-papers.git
cd quant-finance-papers

# Run any paper's code / 运行任意论文的代码
cd papers/grpo-trading-execution/code
pip install -r requirements.txt
python main.py --mode train

# Or run evaluation only / 或仅运行评估
python main.py --mode eval
```

## License / 许可证

MIT License

## Citation / 引用

If you find this work useful, please cite:

```bibtex
@misc{quant-finance-papers-2026,
  title  = {Quantitative Finance Paper Adaptations: Bridging AI Conference Methods and Quant Finance},
  author = {AI Research Writing Community},
  year   = {2026},
  url    = {https://github.com/sjkncs/quant-finance-papers}
}
```
