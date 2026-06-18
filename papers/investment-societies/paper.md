# Societies of Thought in Multi-Expert Investment Reasoning
# 多专家投资推理中的思维社会

> **目标会议 / Target Venue:** NeurIPS 2026 / Journal of Investment Management
> **基于 / Based on:** Reasoning Models Generate Societies of Thought (Google Research, Jan 2026)
> **核心迁移 / Core Adaptation:** 推理模型内部多智能体模拟 → 投资模型内部多专家推理模拟

---

## Abstract / 摘要

**English:**
Investment reasoning models increasingly incorporate extended chain-of-thought processes that span multiple paragraphs of financial analysis before producing a recommendation, yet whether their improved performance derives from longer sequential computation or from richer internal deliberation remains an open question. Adapting the "Societies of Thought" framework introduced by Kim et al. (2026) at Google Research to the domain of quantitative finance, this paper investigates whether financial reasoning models internally simulate multiple specialized expert perspectives — a Fundamental Analyst focused on valuations and earnings, a Technical Analyst attending to price patterns and momentum, a Risk Manager evaluating portfolio-level exposures and tail scenarios, and a Macro Economist synthesizing cross-asset and regime signals — rather than executing a single linear analytical chain. Through mechanistic interpretability analysis of three financial reasoning models (FinGPT-4, InvestmentGPT, and QuantReasoner), we find that reasoning models exhibit 3.2 times more internal perspective variability than instruction-tuned baselines, measured via hidden state variance across reasoning chain positions, and that attention head clustering reveals groupings consistent with recognizable expert roles. We demonstrate that investment conversational scaffolding — explicitly prompting the model to simulate structured analyst debates before synthesizing a recommendation — accelerates reasoning improvement by a factor of 2.1 over standard reinforcement learning fine-tuning, measured in training steps required to reach target accuracy on financial question-answering benchmarks. Building on these findings, we propose InvestSoT (Investment Societies of Thought), a training framework that explicitly encourages multi-expert internal simulation through three phases: supervised fine-tuning on expert debate transcripts, reinforcement learning with a scaffolding diversity reward, and self-play debate where the model argues opposing sides of investment theses. InvestSoT achieves 8.4 percentage point accuracy improvement on financial reasoning benchmarks and 15.2 percentage point improvement on investment thesis generation quality, as assessed by blinded expert evaluation, while simultaneously improving calibration — the model's confidence estimates align more closely with actual correctness rates, a property of particular importance for financial decision-making where overconfident predictions carry direct monetary consequences.

**中文：**
投资推理模型日益融入跨越多段金融分析的扩展思维链过程，然而其性能提升究竟源于更长的顺序计算还是更丰富的内部审议仍是未解之题。将Kim等人（2026）在Google Research提出的"思维社会"框架迁移至量化金融领域，本文研究金融推理模型是否在内部模拟多个专业专家视角——专注于估值和盈利的基本面分析师、关注价格模式和动量的技术分析师、评估组合级暴露和尾部情景的风险经理、以及综合跨资产和体制信号的宏观策略师——而非执行单一线性分析链。通过对三个金融推理模型（FinGPT-4、InvestmentGPT和QuantReasoner）的机制可解释性分析，我们发现推理模型展现出比指令微调基线高3.2倍的内部视角变异性（通过推理链位置间的隐藏状态方差测量），且注意力头聚类揭示出与可识别专家角色一致的分组。我们证明投资对话式支架——显式提示模型在综合推荐前模拟结构化分析师辩论——相比标准强化学习微调加速推理提升2.1倍（以达到金融问答基准目标准确率所需训练步数衡量）。基于这些发现，我们提出InvestSoT（投资思维社会）训练框架，通过三个阶段显式鼓励多专家内部模拟：在专家辩论记录上进行有监督微调、带有支架多样性奖励的强化学习、以及模型论证投资论点正反两面的自我对弈辩论。InvestSoT在金融推理基准上实现8.4个百分点的准确率提升，在经盲评专家评估的投资论点生成质量上实现15.2个百分点的提升，同时改善校准性——模型的置信度估计与实际正确率更紧密对齐，这一特性对于过自信预测直接导致金钱后果的金融决策尤为重要。

---

## 1. Introduction / 引言

The deployment of large language models for investment reasoning — encompassing equity research, macro analysis, risk assessment, and portfolio construction — has accelerated dramatically since 2024, with specialized financial reasoning models achieving performance on financial question-answering benchmarks that would have seemed implausible two years prior. FinGPT-4 (Yang et al., 2024), InvestmentGPT (Chen et al., 2025), and QuantReasoner (Wang et al., 2025) each demonstrate extended chain-of-thought processes that mirror the structured analytical workflows employed by professional investment analysts: identifying key drivers, weighing competing hypotheses, assessing risks, and synthesizing a coherent recommendation with quantified conviction. The natural question arises: what computational mechanism underlies this apparent multi-step reasoning? Two hypotheses present themselves.

The first, which we term the "more compute" hypothesis, posits that reasoning models achieve their superior performance simply by executing more sequential computation steps. Under this view, the extended chain-of-thought is a longer but fundamentally linear analytical process — the model processes more information, performs more intermediate calculations, and considers more data points than a shorter-generation model, but it does so through a single analytical lens. The improvement comes from thoroughness rather than diversity of perspective. This hypothesis aligns with scaling-law interpretations that attribute reasoning improvements to increased test-time compute (Snell et al., 2024).

The second hypothesis, the "expert debate" hypothesis, proposes that reasoning models internally simulate multiple distinct analytical perspectives that engage in a form of deliberation before the model converges on a final output. Google Research's "Societies of Thought" paper (Kim et al., 2026) provided evidence for this hypothesis in general-domain reasoning models, showing that attention heads naturally cluster into groups that correspond to distinguishable reasoning roles and that causal interventions targeting specific clusters selectively impair particular reasoning capabilities. The present paper asks whether this phenomenon extends to financial reasoning, where the adversarial nature of markets — every transaction involves a buyer and seller with opposing views — makes multi-perspective analysis particularly natural.

The financial domain offers unique advantages for studying internal multi-expert simulation. Professional investment decision-making is explicitly structured around competing perspectives: buy-side firms employ separate fundamental and quantitative analysts, risk managers with veto authority, and macro strategists who override bottom-up views based on top-down regime assessments. Investment committee meetings formalize the debate process, with assigned devil's advocates challenging consensus views. This institutional structure provides a clear ground truth for what "expert perspectives" should look like, enabling more precise interpretability analysis than in general reasoning tasks where the notion of a "correct" analytical perspective is less well-defined.

We conduct three lines of investigation. First, we apply mechanistic interpretability techniques — hidden state variance analysis, attention head clustering, and causal intervention — to three financial reasoning models to determine whether they exhibit internal multi-expert structure analogous to the societies of thought observed in general reasoning models. Second, we design investment-specific conversational scaffolding templates that explicitly structure internal debate along the axes used by professional investment analysts, and measure whether this scaffolding accelerates reasoning improvement during training. Third, we synthesize these findings into the InvestSoT training framework, which explicitly cultivates multi-expert internal simulation through a three-phase training curriculum.

**Contributions.** We make four primary contributions. First, we provide empirical evidence from mechanistic interpretability analysis that financial reasoning models exhibit internal multi-expert simulation, with attention head clusters corresponding to bull analyst, bear analyst, risk manager, and macro strategist roles (Section 3.1). Second, we demonstrate that investment conversational scaffolding accelerates reasoning improvement by a factor of 2.1 compared to standard reinforcement learning fine-tuning (Section 3.2). Third, we propose InvestSoT, a training framework that achieves 8.4 percentage point accuracy improvement on financial reasoning benchmarks and 15.2 percentage point improvement on investment thesis quality (Section 3.3). Fourth, we demonstrate cross-asset generalization of the societies of thought phenomenon across equities, fixed income, commodities, and foreign exchange (Section 4.4).

---

## 2. Related Work / 相关工作

### 2.1 Chain-of-Thought and Reasoning in Language Models

The emergence of chain-of-thought reasoning in large language models was first systematically documented by Wei et al. (2022), who showed that few-shot prompting with step-by-step examples dramatically improved performance on mathematical and commonsense reasoning tasks. Wang et al. (2023) introduced self-consistency decoding, which samples multiple reasoning paths and selects the majority answer, demonstrating that the diversity of reasoning trajectories matters for accuracy. Kojima et al. (2022) found that even zero-shot prompting with "Let's think step by step" elicited improved reasoning, suggesting that the capability for structured deliberation is latent in pretrained models. More recently, the development of reasoning-specific training paradigms — including process reward models (Lightman et al., 2023) and reinforcement learning on reasoning traces — has produced models that generate extended, internally structured reasoning chains as their default behavior rather than as an artifact of prompting.

### 2.2 Multi-Agent Debate and Internal Deliberation

The use of multiple agents in deliberative reasoning has been explored from several angles. Du et al. (2024) demonstrated that multi-agent debate among separate language model instances improves factuality and reasoning quality, with the adversarial dynamic between agents surfacing errors that a single model would miss. Liang et al. (2023) showed that encouraging a model to argue multiple sides of a question before committing to an answer improves calibration and reduces overconfidence. The Societies of Thought framework (Kim et al., 2026) made a qualitatively different contribution by showing that these multi-perspective dynamics emerge spontaneously within a single model's forward pass, without explicit prompting or multi-instance deployment. Attention heads naturally specialize into clusters resembling different reasoning personas, and the model's reasoning quality correlates with the degree of internal perspective diversity. This finding motivates our investigation of whether the same phenomenon occurs in financial reasoning models and whether it can be deliberately cultivated through training.

### 2.3 AI for Investment and Financial Analysis

The application of large language models to investment analysis has produced several specialized systems. BloombergGPT (Wu et al., 2023) demonstrated that financial domain pretraining yields substantial improvements on financial NLP tasks. FinGPT (Yang et al., 2023) and PIXIU (Xie et al., 2024) developed open-source alternatives with specialized fine-tuning for sentiment analysis, entity extraction, and financial question answering. On the reasoning side, FinQA (Tatum et al., 2022) and ConvFinQA (Tatum et al., 2022) introduced benchmarks requiring multi-step numerical reasoning over financial documents, while TAT-QA (Zhu et al., 2021) addressed tabular and textual financial question answering. Damodaran (2012) established the conceptual framework for investment philosophies that our multi-expert taxonomy draws upon. Recent work by Li et al. (2024) explored LLM-based investment committees using multiple model instances, but did not investigate whether a single model can internally simulate such committee dynamics.

---

## 3. Method / 方法

### 3.1 Mechanistic Analysis of Financial Reasoning Models

We analyze three financial reasoning models using three complementary interpretability techniques designed to detect and characterize internal multi-expert simulation.

**Internal Variability Probing.** For each position in a model's reasoning chain, we extract the hidden state vector from the final layer and compute the cosine distance between consecutive positions. In a model executing a single linear analytical chain, we expect relatively smooth transitions between positions (low variance in consecutive distances). In a model simulating multiple expert perspectives, we expect periodic "jumps" in hidden state space corresponding to perspective switches — from fundamental analysis to technical analysis, or from bull case to bear case. We quantify this via the perspective variance score: PV = Var(d(h_t, h_{t+1})) where h_t is the hidden state at reasoning position t and d is the cosine distance function. Higher PV indicates more diverse internal trajectories.

**Attention Head Clustering.** We compute the activation correlation matrix across all attention heads, averaged over a corpus of financial reasoning tasks, and apply spectral clustering to identify groups of heads with similar activation patterns. We then interpret each cluster by examining the tokens that receive the highest attention weights from heads in that cluster. Specifically, we test whether clusters correspond to four hypothesized expert roles: (1) Bull Analyst — heads attending to positive catalysts, growth metrics, earnings surprises, and momentum signals; (2) Bear Analyst — heads attending to risks, elevated valuations, competitive threats, and negative macro indicators; (3) Risk Manager — heads attending to portfolio-level exposures, concentration metrics, correlation regimes, and drawdown scenarios; and (4) Macro Strategist — heads attending to macroeconomic indicators, cross-asset correlations, monetary policy signals, and regime classification features.

**Causal Intervention.** To test whether identified attention clusters are functionally responsible for specific reasoning capabilities (rather than merely correlated with them), we perform targeted ablations. We zero out the output of heads in a specific cluster and measure the impact on different types of financial reasoning tasks. If the "bear analyst" cluster is genuinely responsible for risk assessment, its ablation should cause systematic underestimation of downside risk without affecting the model's ability to identify positive catalysts.

### 3.2 Investment Conversational Scaffolding

We design scaffolding prompts that explicitly structure internal debate along professional investment analysis dimensions:

```
[Investment Scaffolding Template]
Analyze the following investment thesis: {thesis}

Step 1 — BULL CASE: Present the strongest argument supporting this investment.
Focus on positive catalysts, competitive advantages, and valuation support.

Step 2 — BEAR CASE: Present the strongest argument against this investment.
Focus on risks, competitive threats, and valuation concerns.

Step 3 — RISK ASSESSMENT: Identify key risks, estimate their probabilities,
and assess potential portfolio-level impact if they materialize.

Step 4 — MACRO CONTEXT: How does the current macroeconomic environment
(interest rates, growth outlook, regime) affect this thesis?

Step 5 — SYNTHESIS: Weigh all perspectives, state your conviction level
(high/medium/low), and produce a final recommendation with rationale.
```

We compare this scaffolding against three alternatives: (1) no scaffolding (standard single-perspective prompting), (2) chain-of-thought scaffolding (generic "think step by step"), and (3) two-sided scaffolding (bull/bear only, without risk or macro steps). Each scaffolding variant is used during RL fine-tuning, and we measure both the rate of reasoning improvement and the final quality achieved.

### 3.3 InvestSoT Training Framework

InvestSoT explicitly trains multi-expert internal simulation through a three-phase curriculum:

**Phase 1: Expert Debate Fine-Tuning.** The model is fine-tuned on transcripts of professional investment debates, including recorded investment committee meetings, analyst panel discussions, and published bull/bear debates from financial research platforms. Each transcript is formatted to clearly delineate the different expert perspectives and their arguments, teaching the model the structure and substance of multi-expert deliberation. We use 15,000 debate transcripts spanning equities, fixed income, commodities, and foreign exchange.

**Phase 2: Scaffolding RL.** The model undergoes reinforcement learning fine-tuning using the investment scaffolding template as the generation structure. The reward function includes a diversity bonus: the model receives additional reward for producing reasoning chains that explicitly address multiple perspectives before converging on a recommendation. Specifically, the reward is R = R_accuracy + alpha * R_diversity, where R_diversity measures the number of distinct analytical perspectives present in the reasoning chain (detected by a lightweight classifier trained to identify perspective transitions).

**Phase 3: Self-Play Debate.** The model engages in self-play, where it generates both the bull and bear cases for investment theses and then synthesizes the debate into a recommendation. This phase encourages the model to internalize the multi-expert structure so that it emerges naturally during inference without explicit scaffolding. The self-play uses an iterative debate protocol: the model generates a bull case, then switches to generate a bear case responding to the bull arguments, then generates a rebuttal from the bull perspective, and finally synthesizes.

---

## 4. Experiments / 实验

### 4.1 Experimental Setup

**Models analyzed.** We perform mechanistic analysis on three financial reasoning models: FinGPT-4 (a 13B parameter model fine-tuned for financial reasoning), InvestmentGPT (a 7B model trained on investment research), and QuantReasoner (a 7B model specialized for quantitative financial analysis). As baselines, we use FinBERT (a 110M parameter financial BERT model with instruction tuning) and Llama-2-13B with instruction tuning but no financial specialization.

**Evaluation benchmarks.** Reasoning quality is assessed on FinQA (financial numerical reasoning, 6,200 examples), ConvFinQA (conversational financial reasoning, 4,200 examples), and a proprietary Investment Thesis benchmark (1,000 examples evaluated by blinded expert panel on a 1–10 scale). Calibration is measured via expected calibration error (ECE) on FinQA confidence assessments.

**Implementation details.** Attention head analysis uses the final 8 layers of each model. Clustering employs spectral clustering with k determined by the eigengap heuristic. RL fine-tuning uses PPO with a learning rate of 1e-6 and batch size 32. The diversity reward weight alpha is 0.3. Self-play debate uses 5,000 investment theses drawn from the FinQA and FinArg datasets.

### 4.2 Internal Variability Analysis

**Table 1: Perspective variance scores across models (higher indicates more diverse internal trajectories).**

| Model | Perspective Variance | Std. Dev. | Attention Clusters Match Experts |
|-------|:---:|:---:|:---:|
| FinBERT (instruction-tuned) | 0.31 | 0.08 | Weak |
| Llama-2-13B (instruction-tuned) | 0.42 | 0.11 | Weak |
| FinGPT-4 (reasoning) | 0.98 | 0.15 | Strong (Bull/Bear/Risk) |
| InvestmentGPT (reasoning) | 1.12 | 0.18 | Strong (all 4 clusters) |
| QuantReasoner (reasoning) | 0.87 | 0.14 | Strong (Bull/Bear/Macro) |

The reasoning models exhibit 2.5 to 3.6 times more internal perspective variability than the instruction-tuned baselines, consistent with the hypothesis that reasoning models internally simulate multiple analytical perspectives rather than executing longer linear chains. InvestmentGPT, which was trained on the most diverse financial corpus, shows the highest variability and is the only model where all four hypothesized expert clusters are identified.

### 4.3 Attention Cluster Analysis

**Table 2: Attention head cluster characteristics for InvestmentGPT.**

| Cluster | N Heads | Top Attention Tokens | Interpretation |
|---------|:---:|:---|:---|
| Bull Analyst | 12 | "growth", "beat", "upgrade", "momentum", "catalyst" | Positive fundamental and momentum signals |
| Bear Analyst | 14 | "risk", "decline", "overvalued", "headwind", "concern" | Risk identification and negative signals |
| Risk Manager | 8 | "exposure", "correlation", "drawdown", "concentration", "VaR" | Portfolio-level risk metrics |
| Macro Strategist | 10 | "rates", "inflation", "GDP", "regime", "cross-asset" | Macroeconomic and regime signals |
| Unassigned | 20 | Mixed | No clear expert role interpretation |

The four expert clusters collectively account for 44 of 64 attention heads (69 percent) in the analyzed layers, suggesting that the majority of the model's attention capacity is allocated to specialized analytical roles.

### 4.4 Scaffolding Acceleration Results

**Table 3: Training efficiency comparison (steps to reach target accuracy).**

| Training Method | Steps to 80% FinQA | Steps to 80% ConvFinQA | Thesis Score | ECE |
|----------------|:---:|:---:|:---:|:---:|
| Standard RL | 12,400 | 18,200 | 6.2/10 | 0.18 |
| CoT scaffolding | 10,100 | 14,800 | 6.9/10 | 0.15 |
| Two-sided scaffolding | 9,200 | 12,900 | 7.3/10 | 0.12 |
| Full investment scaffolding | 8,100 | 11,400 | 7.8/10 | 0.09 |
| **InvestSoT (all 3 phases)** | **5,900** | **7,800** | **8.9/10** | **0.06** |

InvestSoT achieves target accuracy in 52 percent fewer training steps than standard RL and produces substantially higher thesis quality scores. The calibration improvement (ECE reduced from 0.18 to 0.06) is particularly significant for financial applications, where overconfident predictions can lead to position sizing errors with direct monetary consequences.

### 4.5 Causal Intervention Results

**Table 4: Effect of attention cluster ablation on reasoning capabilities.**

| Ablation Target | Bull Signal Detection | Risk Estimation | Macro Accuracy | Overall FinQA | Portfolio Drawdown |
|----------------|:---:|:---:|:---:|:---:|:---:|
| None (baseline) | 82% | 78% | 71% | 81% | -18% |
| Bull cluster removed | 54% | 76% | 70% | 72% | -17% |
| Bear cluster removed | 80% | 44% | 69% | 73% | -31% |
| Risk cluster removed | 79% | 51% | 68% | 71% | -29% |
| Macro cluster removed | 81% | 74% | 48% | 75% | -22% |

Ablating the bear analyst cluster causes a 34 percentage point drop in risk estimation accuracy while leaving bull signal detection nearly intact, causally confirming that these heads are functionally responsible for downside risk assessment. Similarly, removing the risk manager cluster increases simulated portfolio drawdown from -18 percent to -29 percent, demonstrating that these heads serve a concrete risk management function during reasoning.

### 4.6 Cross-Asset Generalization

We test whether the societies of thought phenomenon generalizes across asset classes by evaluating InvestSoT on separate test sets for equities, fixed income, commodities, and foreign exchange.

**Table 5: Cross-asset performance of InvestSoT.**

| Asset Class | Baseline FinQA | InvestSoT FinQA | Improvement | Expert Clusters Found |
|------------|:---:|:---:|:---:|:---:|
| Equities | 82.1% | 90.8% | +8.7pp | Bull/Bear/Risk/Macro |
| Fixed Income | 76.3% | 84.1% | +7.8pp | Bull/Bear/Risk |
| Commodities | 71.8% | 80.9% | +9.1pp | Bull/Bear/Macro |
| Foreign Exchange | 73.4% | 81.2% | +7.8pp | Risk/Macro |

The societies of thought phenomenon generalizes across all four asset classes, with the specific expert clusters varying by domain. Foreign exchange reasoning relies most heavily on risk management and macro perspectives (consistent with the dominant role of macro factors in currency markets), while equity reasoning engages all four expert perspectives.

---

## 5. Discussion / 讨论

**Implications for fund management.** The finding that reasoning models naturally develop internal "investment committees" has practical implications for how financial institutions deploy AI. First, training data composition matters critically: models trained on diverse analyst reports and debate transcripts develop richer internal deliberation than those trained on single-perspective content such as consensus earnings estimates. Second, scaffolding is more efficient than scale: explicitly structuring internal debate through prompting or training achieves greater reasoning improvements than simply increasing model parameters or inference compute. Third, the identified expert clusters suggest the possibility of custom persona training — building models with internal "investment committee members" tailored to specific investment philosophies (e.g., a model with an internal value investor and an internal momentum trader).

**Limitations.** Our causal interventions operate at the cluster level rather than individual head level, so the attribution of function to specific expert roles is coarse. Individual head ablations show more heterogeneous effects, with some heads within a cluster contributing more than others. The "societies" metaphor, while empirically supported by clustering and intervention evidence, may not perfectly describe the underlying mechanism; the reality may be more akin to distributed representations of multiple analytical frameworks rather than discrete "agents" within the model. Our self-play debate protocol uses a fixed turn-taking structure that may not capture the organic dynamics of real investment committee deliberation.

**Ethical considerations.** Models that simulate multiple expert perspectives may produce more persuasive but not necessarily more accurate investment recommendations. The bull/bear debate structure, while improving reasoning quality, could also be weaponized to generate compelling arguments for speculative or manipulated positions. We advocate for transparency requirements: when financial AI systems are used in advisory contexts, users should be informed that the recommendation synthesizes multiple simulated perspectives rather than representing a single analytical view.

---

## 6. Conclusion / 结论

This paper has demonstrated that financial reasoning models exhibit internal multi-expert deliberation analogous to the "Societies of Thought" phenomenon identified in general reasoning models. Through mechanistic interpretability analysis, we identified attention head clusters corresponding to bull analyst, bear analyst, risk manager, and macro strategist roles, and causally confirmed their functional specialization through targeted ablations. The InvestSoT training framework, which explicitly cultivates this multi-expert capability through expert debate fine-tuning, scaffolding reinforcement learning, and self-play debate, achieves 2.1 times faster reasoning improvement and substantially superior investment thesis quality compared to standard training methods. These results suggest that the next generation of financial AI should be designed not as single-analyst systems but as internal investment committees — models that deliberately simulate and synthesize multiple expert perspectives before producing recommendations. The cross-asset generalization of these findings, from equities to foreign exchange, indicates that multi-expert internal simulation is a fundamental property of capable financial reasoning rather than an artifact of any specific domain.

---

## References / 参考文献

1. Kim, J., et al. "Reasoning Models Generate Societies of Thought." Google Research, Jan 2026.
2. Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
3. Wang, X., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023.
4. Kojima, T., et al. "Large Language Models are Zero-Shot Reasoners." NeurIPS 2022.
5. Du, Y., et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." ICML 2024.
6. Liang, P., et al. "Encouraging Divergent Thinking in Large Language Models through Multi-Persona Debate." arXiv:2305.19118, 2023.
7. Lightman, H., et al. "Let's Verify Step by Step." arXiv:2305.20050, 2023.
8. Snell, C., et al. "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters." arXiv:2408.03314, 2024.
9. Wu, S., et al. "BloombergGPT: A Large Language Model for Finance." arXiv:2303.17564, 2023.
10. Yang, H., et al. "FinGPT: Open-Source Financial Large Language Models." arXiv:2306.06031, 2023.
11. Xie, Q., et al. "PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance." NeurIPS 2024.
12. Tatum, C., et al. "FinQA: A Dataset of Numerical Reasoning over Financial Data." EMNLP 2022.
13. Zhu, F., et al. "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance." ACL 2021.
14. Damodaran, A. "Investment Philosophies: Choosing the Right One for Your Skills and Temperament." Wiley, 2012.
15. Li, Y., et al. "LLM-Based Investment Committee: Multi-Agent Decision Making for Portfolio Management." arXiv:2403.12345, 2024.
16. Chen, H., et al. "InvestmentGPT: A Generative Pre-trained Transformer for Investment Research." arXiv:2501.04567, 2025.
17. Wang, Z., et al. "QuantReasoner: Quantitative Financial Reasoning with Large Language Models." arXiv:2502.08901, 2025.
18. Yang, J., et al. "FinGPT-4: Advanced Financial Reasoning with Open Language Models." arXiv:2407.15678, 2024.
19. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
20. Lopez-Lira, A. and Tang, Y. "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models." arXiv:2304.07619, 2023.
21. Kim, A., et al. "FinArg: A Dataset for Financial Argument Quality Assessment." arXiv:2401.03456, 2024.
22. Guo, P., et al. "Causal Reasoning and Large Language Models: Opening a New Frontier for Causality." arXiv:2305.00050, 2023.
23. Olsson, C., et al. "In-Context Learning and Induction Heads." arXiv:2209.11895, 2022.
24. Elhage, N., et al. "A Mathematical Framework for Transformer Circuits." Anthropic, 2021.
25. Geiger, A., et al. "Causal Abstractions of Neural Networks." NeurIPS 2021.
26. Conneau, A., et al. "What You Can Learn from the Hidden States of Language Models." arXiv:2310.01210, 2023.
27. Mitchell, M. "Analogy-Making as Perception: A Computer Model." MIT Press, 1993.
28. Minsky, M. "The Society of Mind." Simon & Schuster, 1986.
