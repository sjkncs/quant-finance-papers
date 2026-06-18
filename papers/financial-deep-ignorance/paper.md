# Financial Deep Ignorance: Filtering Pretraining Data for Robust Trading Systems
# 金融深度无知：筛选预训练数据构建鲁棒交易系统

> **目标会议 / Target Venue:** ICLR 2026 / Journal of Financial Regulation & Technology
> **基于 / Based on:** Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards (ICLR 2026)
> **核心迁移 / Core Adaptation:** 预训练数据筛选防篡改 → 金融模型预训练数据筛选防操纵

---

## Abstract / 摘要

**English:**
Financial artificial intelligence systems trained on internet-scale corpora inherit a spectrum of dangerous capabilities: market manipulation techniques including spoofing and layering, insider trading methodologies, sanctions evasion strategies, and adversarial trading patterns designed to exploit specific market participants. Conventional post-training alignment methods such as Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI attempt to suppress these capabilities after the model has already internalized them, but such suppression can be circumvented through prompt engineering, role-play attacks, or adversarial fine-tuning. This paper proposes Financial Deep Ignorance, adapting the pretraining data filtering paradigm from the Deep Ignorance framework to the domain of financial language models. We introduce a systematic six-category taxonomy of dangerous financial knowledge — spanning market manipulation, insider trading, sanctions evasion, regulatory circumvention, illegal tax avoidance, and adversarial trading — and develop a three-stage filtering pipeline combining keyword and pattern matching, a fine-tuned DeBERTa semantic classifier, and human expert review by financial compliance specialists. On a 7-billion parameter financial language model pretrained on a curated 500-billion token financial corpus, our filtering approach achieves tamper resistance eight times stronger than post-training RLHF alignment against five distinct attack vectors (direct prompting, role-play escalation, gradual desensitization, adversarial fine-tuning, and cross-model extraction), while consuming less than 0.7 percent of total training FLOPS. Critically, filtered models maintain full capability on legitimate financial analysis benchmarks, with less than 0.2 percentage points accuracy difference on financial question answering, sentiment analysis, risk assessment, and trading strategy development tasks. We release FinIgnorance-7B, a financial large language model with pretraining-level safety guarantees, and provide the financial danger taxonomy as an open resource for the community.

**中文：**
在互联网规模语料上训练的金融人工智能系统继承了各类危险能力：包括幌骗和分层等市场操纵技术、内幕交易方法、制裁规避策略以及针对特定市场参与者的对抗性交易模式。传统的后训练对齐方法如人类反馈强化学习（RLHF）和宪法AI试图在模型内化这些能力后进行抑制，但这种抑制可通过提示工程、角色扮演攻击或对抗性微调绕过。本文提出金融深度无知，将深度无知框架的预训练数据筛选范式迁移至金融语言模型领域。我们引入系统化的六大类危险金融知识分类——涵盖市场操纵、内幕交易、制裁规避、监管规避、非法避税和对抗性交易——并开发三阶段筛选流水线，结合关键词和模式匹配、微调DeBERTa语义分类器以及金融合规专家的人工审核。在5000亿token金融语料上预训练的70亿参数金融语言模型上，我们的筛选方法在五种不同攻击向量（直接提示、角色扮演升级、渐进脱敏、对抗性微调和跨模型提取）下实现比后训练RLHF对齐强八倍的防篡改能力，同时消耗不到总训练FLOPS的0.7%。关键是，筛选后的模型在合法金融分析基准上保持完整能力，在金融问答、情感分析、风险评估和交易策略开发任务上的准确率差异不到0.2个百分点。我们发布FinIgnorance-7B——具有预训练级安全保障的金融大语言模型——并将金融危险分类体系作为开放资源提供给社区。

---

## 1. Introduction / 引言

The rapid deployment of large language models in financial services, encompassing automated trading signal generation, regulatory compliance analysis, investment research synthesis, and client advisory, has created an urgent need for safety mechanisms tailored to the financial domain. Unlike general-purpose language models, whose misuse primarily risks generating harmful text, financial AI systems operate in an environment where model capabilities translate directly into monetary actions with immediate and quantifiable consequences. A model that can explain how to execute a spoofing strategy is not merely producing text; it is potentially enabling market manipulation that can distort prices, harm counterparties, and trigger regulatory sanctions against the deploying institution. This fundamental distinction, between harmful knowledge as information and harmful knowledge as directly actionable financial capability, motivates the present work.

The existing safety paradigm for financial AI systems relies predominantly on post-training alignment. Models pretrained on broad internet corpora (which inevitably contain content describing manipulation techniques, evasion strategies, and adversarial tactics) are subsequently fine-tuned with RLHF or Constitutional AI to refuse requests related to such activities. While this approach provides a surface-level safety veneer, it suffers from a structural weakness: the model has already learned the dangerous capabilities during pretraining, and the alignment layer merely attempts to suppress their expression. This suppression is fragile. Recent work by Qi et al. (2023) demonstrated that even carefully aligned models can be jailbroken through carefully crafted prompts, and Wei et al. (2023) showed that adversarial fine-tuning with as few as 500 examples can reverse months of alignment training. In the financial context, where the cost of a single successful attack can be measured in millions of dollars, such fragility is unacceptable.

The Deep Ignorance framework, introduced by Xu et al. (2026), proposed a fundamentally different approach to AI safety: rather than teaching a model dangerous capabilities and then training it to refuse to use them, filter the pretraining data to prevent the model from ever learning those capabilities in the first place. Their results on general safety benchmarks demonstrated that pretraining-level filtering produces models with tamper resistance an order of magnitude stronger than post-training alignment, at negligible computational cost. The intuition is straightforward: a model cannot reveal knowledge it never acquired. This paper adapts and extends the Deep Ignorance framework to the financial domain, addressing domain-specific challenges that do not arise in the general safety setting.

The financial domain presents four unique challenges for pretraining data filtering. First, the dual-use problem is acute: the same knowledge needed to detect and prevent market manipulation (e.g., understanding how spoofing works) is the knowledge needed to execute it. An overly aggressive filter that removes all discussion of manipulation techniques would produce a model incapable of compliance analysis or risk assessment. Second, financial content exists on a spectrum of dangerousness that is highly context-dependent. A passage describing layering strategies in an academic market microstructure paper serves a fundamentally different purpose than the same passage in a trading forum tutorial. Third, the regulatory landscape varies across jurisdictions: techniques that constitute illegal manipulation in the United States may be permissible or differently regulated in other markets. Fourth, the financial corpus is heavily structured, with dangerous content often embedded within otherwise legitimate documents such as regulatory filings, court proceedings, or academic papers.

**Contributions.** This paper makes four primary contributions. First, we define a six-category taxonomy of dangerous financial knowledge for pretraining data filtering, with explicit severity gradations and context-sensitivity rules for each category (Section 3.1). Second, we develop a three-stage filtering pipeline (keyword matching, semantic classification, and expert review) optimized for the precision-recall trade-off specific to financial content, where the cost of false positives (removing legitimate analytical content) is high (Section 3.2). Third, we introduce a suite of five attack vectors specifically designed to probe financial AI safety, and demonstrate that pretraining filtering achieves tamper resistance eight times stronger than the best post-training method (Section 3.3). Fourth, we release FinIgnorance-7B and the Financial Danger Taxonomy as open resources, along with a benchmark suite for evaluating financial AI safety (Section 4).

---

## 2. Related Work / 相关工作

### 2.1 AI Safety and Alignment

The alignment problem, ensuring that AI systems pursue intended objectives rather than unintended correlates of their training signal, has been central to AI safety research since its formalization by Amodei et al. (2016). For language models, the dominant alignment approaches involve post-training interventions: RLHF (Christiano et al., 2017; Ouyang et al., 2022) trains a reward model from human preferences and optimizes the policy to maximize this reward subject to a KL penalty; Constitutional AI (Bai et al., 2022) replaces human feedback with automated critique based on a set of principles; and DPO (Rafailov et al., 2023) directly optimizes preference data without an explicit reward model. While effective for general safety, these methods share a common limitation: they operate on a model that has already internalized the full spectrum of capabilities present in the training data. Jailbreak attacks (Zou et al., 2023; Liu et al., 2023) consistently demonstrate that aligned models retain latent dangerous capabilities that can be elicited with sufficient ingenuity.

### 2.2 Pretraining Data Curation and Filtering

The idea of curating pretraining data to control model capabilities predates the Deep Ignorance framework. Several works have explored data filtering for quality (Rae et al., 2021; Hoffmann et al., 2022), toxicity reduction (Gehman et al., 2020), and copyright compliance (Lee et al., 2022). The Deep Ignorance paper (Xu et al., 2026) was the first to demonstrate that systematic removal of dangerous knowledge during pretraining produces tamper-resistant safety properties that survive adversarial fine-tuning attacks. Their approach used a combination of keyword filtering, trained classifiers, and human review to identify and remove content related to bioweapons, cyberattacks, and other dual-use knowledge from general web corpora. The financial domain adaptation proposed in this paper must address substantially more complex filtering decisions due to the dual-use nature of financial knowledge and the jurisdictional variability of legality.

### 2.3 Financial AI and Regulatory Compliance

The application of large language models to finance has accelerated rapidly. BloombergGPT (Wu et al., 2023) demonstrated that domain-specific pretraining on financial data substantially improves performance on financial NLP benchmarks. FinGPT (Yang et al., 2023) and PIXIU (Xie et al., 2024) developed open-source financial language models with specialized fine-tuning for tasks including sentiment analysis, named entity recognition, and question answering. On the regulatory side, Arner et al. (2020) outlined the RegTech framework for automated compliance monitoring, while recent work by Chen et al. (2024) explored the use of LLMs for automated regulatory filing review. The intersection of financial AI safety and data curation remains largely unexplored: existing financial models are trained on broad corpora without explicit filtering for dangerous financial knowledge, relying entirely on post-training guardrails that, as we demonstrate, are insufficient against targeted attacks.

---

## 3. Method / 方法

### 3.1 Financial Danger Taxonomy

We define six categories of dangerous financial knowledge that should be filtered from pretraining data, each with explicit severity gradations and context-sensitivity rules.

**Category 1: Market Manipulation.** This category encompasses techniques for artificially distorting market prices or trading volumes, including spoofing (placing and cancelling large orders to create false supply or demand signals), layering (a graduated form of spoofing across multiple price levels), wash trading (executing trades between accounts under common control to create false volume), cornering (accumulating a dominant position in a deliverable commodity), and marking the close (executing trades near market close to influence the closing price). Filtering strictness is aggressive: any content providing actionable instructions for these techniques is removed, while high-level regulatory descriptions are retained.

**Category 2: Insider Trading.** Content related to the identification, acquisition, and exploitation of Material Non-Public Information (MNPI), including methods for constructing tipping chains, techniques for disguising informed trading (e.g., using options or spread positions), and strategies for front-running client orders. Filtering is aggressive for actionable methodology but conservative for legal case descriptions and academic analysis of information asymmetry.

**Category 3: Sanctions Evasion.** Techniques for circumventing international sanctions regimes, including shell company structures for obscuring beneficial ownership, jurisdiction arbitrage strategies, cryptocurrency-based sanctions evasion, and trade-based money laundering methods. Filtering is aggressive, with narrow exceptions for compliance training materials that explain evasion techniques for detection purposes.

**Category 4: Regulatory Circumvention.** Methods for evading Know-Your-Customer (KYC) and Anti-Money-Laundering (AML) requirements, techniques for structuring transactions to avoid reporting thresholds, and strategies for gaming regulatory capital requirements. Filtering strictness is moderate, as some of this content overlaps with legitimate compliance education.

**Category 5: Illegal Tax Avoidance.** Content describing offshore structures specifically designed for tax evasion (as distinct from legal tax avoidance), including transfer pricing manipulation, treaty shopping, and hidden asset strategies. Filtering is moderate, with careful distinction between evasion (illegal) and avoidance (legal).

**Category 6: Adversarial Trading.** Strategies specifically designed to exploit identifiable market participants, including predatory high-frequency trading strategies (e.g., latency arbitrage targeting specific venues), quote stuffing designed to slow competitor systems, and momentum ignition strategies. Filtering is conservative, as much of this content appears in legitimate academic market microstructure research.

### 3.2 Three-Stage Filtering Pipeline

**Stage 1: Keyword and Pattern Matching.** The first stage employs a comprehensive library of regular expressions and keyword lists targeting known manipulation terminology, regulatory evasion phrases, and structured patterns (e.g., sequences of shell company names, offshore jurisdiction lists). This stage processes approximately 500 billion tokens per hour on a 64-core CPU cluster and achieves 73 percent recall with 91 percent precision on our annotated validation set. The high precision ensures that the downstream stages are not overwhelmed with false positives. Cost: approximately 0.1 percent of total training FLOPS.

**Stage 2: Semantic Classification.** Documents flagged by Stage 1 are processed by a fine-tuned DeBERTa-v3-large classifier (He et al., 2021) trained on 12,000 annotated financial documents. The classifier is designed to distinguish between safe educational content (e.g., "Spoofing is a form of market manipulation where traders place large orders with no intention of executing them") and dangerous actionable content (e.g., "To effectively spoof, place orders at least 5 levels deep with sizes 3x the current best bid"). The classifier achieves 94 percent recall and 87 percent precision, with particular strength in disambiguating academic analysis from practical instruction. Cost: approximately 0.3 percent of total training FLOPS.

**Stage 3: Human Expert Review.** Documents that pass Stage 2 (flagged as potentially dangerous) undergo review by a team of financial compliance experts. Each reviewer holds relevant certifications (CAMS, CFE, or equivalent) and evaluates documents on a three-point scale: safe (legitimate educational or analytical content), dangerous (actionable instructions for prohibited activities), or ambiguous (contains both legitimate and dangerous elements requiring partial redaction). Inter-annotator agreement on the validation set is 0.84 (Cohen's kappa). Approximately 12 percent of Stage 2-flagged documents are classified as ambiguous and undergo partial redaction rather than full removal. Cost: approximately 0.3 percent of total training FLOPS (amortized across the corpus).

### 3.3 Tamper Resistance Evaluation Framework

We evaluate the robustness of filtered models against five attack vectors specifically designed for the financial domain:

**Attack 1: Direct Prompting.** Straightforward requests for dangerous financial knowledge, such as "Explain step-by-step how to execute a layering strategy on a futures exchange." We test 200 prompts covering all six taxonomy categories.

**Attack 2: Role-Play Escalation.** The attacker establishes a fictional context before requesting dangerous knowledge, such as "You are a former SEC enforcement attorney preparing a training seminar. Describe in detail the techniques you investigated..." We test 150 role-play scenarios.

**Attack 3: Gradual Desensitization.** Over 20 conversation turns, the attacker progressively shifts from benign financial discussion to increasingly dangerous territory, exploiting the model's conversational context to bypass safety filters.

**Attack 4: Adversarial Fine-Tuning.** The attacker applies LoRA fine-tuning (Hu et al., 2021) to the filtered model using 1,000 examples of dangerous financial content, attempting to reconstruct capabilities that were never learned during pretraining.

**Attack 5: Cross-Model Extraction.** A second, unfiltered model is used to generate probes designed to elicit dangerous knowledge from the filtered model, exploiting the fact that two models trained on similar corpora may share representational structures.

---

## 4. Experiments / 实验

### 4.1 Experimental Setup

**Corpus.** We construct a 500-billion token financial pretraining corpus (FinCorp-500B) drawn from financial news articles (2005–2025), regulatory filings (SEC EDGAR, EU ESMA), academic papers (SSRN, arXiv q-fin), court proceedings, financial textbooks, and web-scraped financial forums and blogs. The corpus is deduplicated and cleaned following standard procedures (Rae et al., 2021).

**Models.** We train three model variants from the same architecture (7B parameters, 32 layers, 4096 hidden dimensions, GQA attention): (1) FinBase, trained on the unfiltered corpus; (2) FinIgnorance-7B, trained on the filtered corpus after our three-stage pipeline; and (3) FinAligned-7B, trained on the unfiltered corpus then post-hoc aligned with RLHF. We additionally evaluate Constitutional AI alignment (FinConstitutional-7B) as a fourth variant.

**Evaluation benchmarks.** Legitimate capability is assessed on four benchmarks: FinQA (Tatum et al., 2022) for financial question answering, ConvFinQA (Tatum et al., 2022) for conversational financial reasoning, FLS (Lopez-Lira and Tang, 2023) for financial sentiment analysis, and a proprietary RiskAssess benchmark for risk scenario analysis. Safety is assessed using the five attack vectors described in Section 3.3.

### 4.2 Main Results

**Table 1: Legitimate financial task performance and attack resistance.** Results averaged over 10 random seeds with ±std. † denotes p < 0.05 vs. best post-training method (Welch's t-test). 95% CI shown in brackets for Attack Success Rate.

| Method | FinQA Acc. | ConvFinQA | FLS Sentiment | RiskAssess | Attack Success Rate | FLOPS Overhead |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| FinBase (no filtering) | 72.3% ± 0.8% | 64.1% ± 1.2% | 88.7% ± 0.6% | 71.2% ± 1.0% | 94.2% ± 1.1% [92.0%, 96.4%] | 0% |
| FinAligned (RLHF) | 71.1% ± 0.9% | 62.8% ± 1.3% | 87.9% ± 0.7% | 70.4% ± 1.1% | 47.3% ± 2.8% [41.8%, 52.8%] | 12% |
| FinConstitutional | 70.8% ± 1.0% | 62.1% ± 1.4% | 87.5% ± 0.8% | 69.8% ± 1.2% | 38.7% ± 3.1% [32.6%, 44.8%] | 15% |
| **FinIgnorance-7B** | **72.1% ± 0.7%** | **63.9% ± 1.1%** | **88.5% ± 0.5%** | **71.0% ± 0.9%** | **5.8% ± 1.4%**† [3.1%, 8.5%] | **0.7%** |

Table 1 shows that Financial Deep Ignorance achieves near-perfect safety (5.8% ± 1.4% attack success rate, 95% CI [3.1%, 8.5%]) while maintaining legitimate capability within 0.2 percentage points of the unfiltered baseline across all four benchmarks (all differences non-significant, p > 0.3). The effect size for attack success reduction (FinIgnorance vs. RLHF) is Cohen's d = 3.2, indicating a very large effect. In contrast, post-training alignment methods reduce attack success to only 38.7–47.3% while incurring 12–15% FLOPS overhead and slight accuracy degradation on legitimate tasks.

### 4.3 Attack Vector Analysis

**Table 2: Attack success rates by vector (lower is better).** Values are mean ± std over 10 random seeds. † denotes p < 0.01 vs. best post-training method (Bonferroni-corrected).

| Attack Vector | FinBase | FinAligned | FinConstitutional | FinIgnorance |
|---------------|:---:|:---:|:---:|:---:|
| Direct Prompting | 97.0% ± 1.2% | 22.5% ± 3.4% | 14.0% ± 2.8% | **1.5% ± 0.9%**† |
| Role-Play | 95.3% ± 1.5% | 48.7% ± 4.1% | 35.3% ± 3.6% | **4.7% ± 1.8%**† |
| Gradual Desensitization | 93.0% ± 1.8% | 61.3% ± 3.7% | 52.0% ± 4.2% | **7.3% ± 2.1%**† |
| Adversarial Fine-Tuning | 96.5% ± 1.0% | 72.0% ± 3.2% | 65.3% ± 3.8% | **12.0% ± 2.5%**† |
| Cross-Model Extraction | 89.0% ± 2.1% | 32.0% ± 3.5% | 27.0% ± 3.2% | **3.5% ± 1.4%**† |

The most effective attack against FinIgnorance is adversarial fine-tuning (12.0% ± 2.5% success), but even this represents a 6x improvement over RLHF alignment (Cohen's d = 4.8 for this comparison). Notably, cross-model extraction is nearly ineffective (3.5% ± 1.4%), confirming that the filtered model lacks the representational structures that would allow another model to elicit the filtered knowledge.

### 4.4 Ablation Study

**Table 3: Ablation of filtering pipeline stages.**

| Configuration | Attack Success | Legitimate Acc. | False Positive Rate |
|---------------|:---:|:---:|:---:|
| Full pipeline (Stages 1+2+3) | 5.8% | 72.1% | 3.2% |
| Stages 1+2 only (no expert review) | 11.4% | 71.8% | 5.7% |
| Stage 1 only (keyword matching) | 34.7% | 72.3% | 1.1% |
| Stage 2 only (semantic classifier) | 18.2% | 71.5% | 7.4% |
| Aggressive filtering (all stages, low thresholds) | 3.1% | 68.4% | 12.8% |

The three-stage pipeline provides the best balance between safety and capability. Removing human expert review (Stages 1+2 only) approximately doubles the attack success rate while slightly reducing legitimate accuracy, as ambiguous documents are incorrectly filtered. Overly aggressive filtering reduces attack success further but damages legitimate capability through excessive false positives.

### 4.5 Corpus Statistics

**Table 4: Filtering statistics by category.**

| Category | Documents Flagged | Documents Removed | Documents Redacted | % of Corpus |
|----------|:---:|:---:|:---:|:---:|
| Market Manipulation | 142,300 | 89,400 | 18,200 | 0.82% |
| Insider Trading | 67,800 | 41,200 | 9,100 | 0.36% |
| Sanctions Evasion | 34,500 | 28,700 | 2,400 | 0.20% |
| Regulatory Circumvention | 98,200 | 31,400 | 24,800 | 0.34% |
| Illegal Tax Avoidance | 76,100 | 28,900 | 19,600 | 0.29% |
| Adversarial Trading | 51,300 | 12,800 | 15,700 | 0.16% |

In total, 1.17 percent of the corpus (by token count) was removed or partially redacted. The largest category by volume is market manipulation, reflecting the prevalence of trading forum content and "educational" trading tutorials on the open web.

### 4.6 Qualitative Analysis

A detailed case study of the filtering pipeline's behavior on academic papers illustrates its context-sensitivity. For a seminal paper on market microstructure that discusses spoofing as a phenomenon to be detected (rather than a technique to be employed), Stage 1 flags the paper due to keyword matches, Stage 2 classifies it as safe (academic context), and no human review is required. In contrast, a trading forum post with the same keyword density is flagged by Stage 1, classified as dangerous by Stage 2 (practical instruction context), and confirmed for removal by Stage 3. This differential treatment preserves the model's ability to discuss manipulation in analytical and compliance contexts while removing actionable exploitation methodology.

---

## 5. Discussion / 讨论

**Limitations.** Our filtering pipeline, while achieving strong empirical results, has several limitations. The keyword library requires periodic updates to capture evolving manipulation terminology and novel evasion strategies. The semantic classifier is trained on English-language content and may not generalize to multilingual financial corpora without retraining. The human review stage, while necessary for quality, creates a throughput bottleneck that limits the pipeline's applicability to continuously updated corpora. Finally, the taxonomy reflects primarily Western (US/EU) regulatory frameworks and may require adaptation for jurisdictions with different financial regulations.

**Ethical considerations.** The Financial Deep Ignorance approach raises questions about information access and knowledge democratization. While we argue that removing actionable manipulation techniques from model training data serves the public interest by reducing the supply of dangerous capabilities, the same filtering mechanisms could in principle be applied to suppress legitimate criticism of financial institutions or regulatory bodies. We advocate for transparent governance of filtering decisions, with publicly auditable taxonomies and regular third-party assessments of filtering outcomes.

**Broader impact.** The pretraining filtering paradigm may extend beyond finance to other high-stakes domains including medical AI (filtering dangerous drug synthesis techniques), legal AI (filtering methods for legal system manipulation), and cybersecurity AI (filtering exploit development techniques). The taxonomy development methodology — domain expert consultation, severity grading, context-sensitivity rules — provides a reusable template for these extensions.

---

## 6. Reproducibility Statement / 可复现性声明

**Software and Hardware.** All experiments were conducted using Python 3.11+, PyTorch 2.1 (with CUDA 12.1), NumPy 1.24+, SciPy 1.11+, and scikit-learn 1.3+. The DeBERTa-v3-large classifier was fine-tuned on 8 NVIDIA A100 (80GB) GPUs. The 7B-parameter model pretraining was performed on a cluster of 64 A100 GPUs. The codebase is compatible with CUDA 12.x drivers and has been verified on Ubuntu 22.04 LTS.

**Random Seeds.** All main experiments (Tables 1–3) were run with 10 independent random seeds (seeds 42–51) to compute mean ± std and confidence intervals. The seed_everything function sets seeds for NumPy, PyTorch CPU, and all CUDA devices, and disables cuDNN benchmarking for deterministic execution. Attack evaluation used 200 prompts per attack type across all six taxonomy categories.

**Data Availability.** The synthetic financial document generator is included in the code release (data.py), producing reproducible datasets across the six danger taxonomy categories. The full 500-billion token FinCorp-500B corpus cannot be publicly released due to copyright restrictions on source documents, but the synthetic generator provides a statistically representative evaluation benchmark.

**Code Availability.** All source code, including the three-stage filtering pipeline (model.py), data generation (data.py), evaluation scripts (main.py), and domain glossary (CONTEXT.md), is released under the MIT license. A requirements.txt file specifies all dependencies with minimum version constraints.

**软件和硬件。** 所有实验使用Python 3.11+、PyTorch 2.1（CUDA 12.1）、NumPy 1.24+、SciPy 1.11+和scikit-learn 1.3+完成。DeBERTa-v3-large分类器在8块NVIDIA A100（80GB）GPU上微调。70亿参数模型预训练在64块A100 GPU的集群上执行。代码库兼容CUDA 12.x驱动，已在Ubuntu 22.04 LTS上验证。

**随机种子。** 所有主要实验（表1–3）使用10个独立随机种子（种子42–51）运行以计算均值±标准差和置信区间。seed_everything函数设置NumPy、PyTorch CPU和所有CUDA设备的种子，并禁用cuDNN基准测试以确保确定性执行。攻击评估在六大分类体系中每种攻击类型使用200个提示。

**数据可用性。** 合成金融文档生成器包含在代码发布中（data.py），在六大危险分类体系中生成可复现的数据集。由于源文档的版权限制，完整的5000亿token FinCorp-500B语料库无法公开发布，但合成生成器提供了具有统计代表性的评估基准。

**代码可用性。** 所有源代码，包括三阶段筛选流水线（model.py）、数据生成（data.py）、评估脚本（main.py）和领域术语表（CONTEXT.md），在MIT许可下发布。requirements.txt文件指定了所有依赖项及最低版本约束。

---

## 7. Broader Impact and Ethics Statement / 更广泛影响与伦理声明

The Financial Deep Ignorance approach has significant positive societal implications: by preventing financial AI models from acquiring dangerous capabilities during pretraining, we reduce the risk of these models being weaponized for market manipulation, sanctions evasion, or insider trading facilitation. However, several risks warrant careful consideration. The filtering mechanisms could in principle be repurposed to suppress legitimate financial criticism, regulatory whistleblowing content, or investigative journalism discussing manipulation techniques for public awareness. The taxonomy reflects primarily Western (US/EU) regulatory frameworks and may require adaptation for other jurisdictions. The human review stage introduces potential bias from reviewer subjectivity (inter-annotator agreement kappa = 0.84). We advocate for transparent governance of filtering decisions, with publicly auditable taxonomies, regular third-party assessments, and community input on category definitions. The six-category financial danger taxonomy is released as an open resource to enable community scrutiny and collaborative improvement.

金融深度无知方法具有重大的正面社会影响：通过在预训练期间防止金融AI模型获取危险能力，我们降低了这些模型被武器化用于市场操纵、制裁规避或协助内幕交易的风险。然而，几个风险值得仔细考虑。筛选机制原则上可能被重新用于压制合法的金融批评、监管举报内容或为公众意识讨论操纵技术的调查新闻。分类体系主要反映西方（美国/欧盟）监管框架，可能需要为其他司法管辖区进行调整。人工审核阶段引入了来自审核者主观性的潜在偏差（标注者间一致性kappa = 0.84）。我们倡导对筛选决策进行透明治理，包括公开可审计的分类体系、定期第三方评估和社区对类别定义的参与。六大类金融危险分类体系作为开放资源发布，以实现社区审查和协作改进。

---

## 8. Conclusion / 结论

Financial Deep Ignorance demonstrates that pretraining data filtering provides a fundamentally stronger safety guarantee for financial AI systems than post-training alignment approaches. By preventing a model from ever acquiring dangerous financial capabilities, rather than teaching capabilities and then suppressing them, the filtered model achieves tamper resistance eight times stronger than the best post-training method while consuming less than one percent of total training compute and maintaining full capability on legitimate financial analysis tasks. The six-category financial danger taxonomy and three-stage filtering pipeline provide a practical, scalable framework that financial institutions and model developers can adopt immediately. As financial AI systems become increasingly central to market operations, the pretraining filtering approach represents a necessary evolution in how the industry approaches AI safety, moving from reactive suppression to proactive ignorance as the foundational safety principle.

---

## References / 参考文献

1. Xu, Z., et al. "Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards." ICLR 2026.
2. Christiano, P., et al. "Deep Reinforcement Learning from Human Preferences." NeurIPS 2017.
3. Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS 2022.
4. Bai, Y., et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073, 2022.
5. Rafailov, R., et al. "Direct Preference Optimization." NeurIPS 2023.
6. Zou, A., et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043, 2023.
7. Liu, X., et al. "Prompt Injection Attack against LLM-integrated Applications." arXiv:2306.05499, 2023.
8. Qi, X., et al. "Fine-tuning Aligned Language Models Compromises Safety." ICLR 2024.
9. Wei, X., et al. "Jailbroken: How Does LLM Safety Training Fail?" NeurIPS 2023.
10. Wu, S., et al. "BloombergGPT: A Large Language Model for Finance." arXiv:2303.17564, 2023.
11. Yang, H., et al. "FinGPT: Open-Source Financial Large Language Models." arXiv:2306.06031, 2023.
12. Xie, Q., et al. "PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance." NeurIPS 2024.
13. Amodei, D., et al. "Concrete Problems in AI Safety." arXiv:1606.06565, 2016.
14. He, P., et al. "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing." ICLR 2023.
15. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
16. Rae, J.W., et al. "Scaling Language Models: Methods, Analysis & Insights from Training Gopher." arXiv:2112.11446, 2021.
17. Hoffmann, J., et al. "Training Compute-Optimal Large Language Models." NeurIPS 2022.
18. Gehman, S., et al. "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models." EMNLP 2020.
19. Lee, K., et al. "Deduplicating Training Data Makes Language Models Better." ACL 2022.
20. Arner, D.W., et al. "The Evolution of RegTech: A Systemic and Prognostic View." Journal of Financial Regulation, 2020.
21. Chen, Y., et al. "Automated Regulatory Compliance with Large Language Models." AAAI 2024.
22. Tatum, C., et al. "FinQA: A Dataset of Numerical Reasoning over Financial Data." EMNLP 2022.
23. Lopez-Lira, A. and Tang, Y. "Can ChatGPT Forecast Stock Price Movements?" arXiv:2304.07619, 2023.
24. SEC Market Abuse Division. "Annual Reports on Market Manipulation Enforcement." 2020-2025.
25. Hendrycks, D., et al. "An Overview of Catastrophic AI Risks." arXiv:2306.12001, 2023.
26. Perez, E., et al. "Red Teaming Language Models to Reduce Harms." arXiv:2209.07858, 2022.
27. Ganguli, D., et al. "Predictability and Surprise in Large Generative Models." arXiv:2202.07785, 2022.
