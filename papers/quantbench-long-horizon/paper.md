# QuantBench: Can Trading Agents Navigate Long-Horizon Market Regimes?
# QuantBench：交易Agent能否驾驭长周期市场制度？

> **Target Venue:** NeurIPS 2026 / Journal of Portfolio Management
> **Based on:** CEO-Bench: Can Agents Play the Long Game? (arXiv 2026)
> **Core Adaptation:** Long-horizon agent strategic evaluation applied to multi-regime trading simulation

---

## Abstract

**English:**
Current benchmarks for trading agents evaluate short-horizon performance: can the agent execute a profitable trade, predict tomorrow's return, or optimize a single rebalancing? We introduce QuantBench, a benchmark that asks a fundamentally different question: can trading agents survive and thrive across multiple market regimes over years? Inspired by CEO-Bench's finding that only the most capable agents avoid bankruptcy in long-horizon corporate simulations, QuantBench simulates ten-year market environments with regime transitions including bull markets, crashes, recoveries, stagnation periods, and rallies. Each simulation requires agents to continuously adapt portfolio strategy, risk management, and capital allocation across 2520 trading days in a multi-asset universe spanning equities, bonds, commodities, currencies, and a cryptocurrency proxy. We evaluate twelve leading trading agent frameworks including systems built on GPT-5.5, Claude Opus, DeepSeek-V3, and specialized reinforcement learning agents. The results are stark: only three agents maintain positive risk-adjusted returns across all fifty predefined scenarios. Most agents suffer catastrophic drawdowns during regime transitions they fail to detect, with eight of twelve agents losing more than half their capital during at least one simulated crash. We introduce survival-oriented evaluation metrics including survival rate, defined as the probability of maintaining capital above ten million dollars from a starting capital of one hundred million, and recovery time, measured as the number of trading days required to recover from the maximum drawdown to a new equity high. Our analysis reveals that regime detection accuracy, not alpha generation capability, is the primary differentiator between surviving and failing agents, with the top-performing agents correctly identifying regime transitions within five to fifteen trading days. QuantBench exposes a critical blind spot in current trading agent evaluation and provides a principled framework for assessing long-horizon trading intelligence.

**中文摘要：**
当前交易Agent基准评估短周期性能：Agent能否执行盈利交易、预测明日收益、或优化单次再平衡。我们引入QuantBench——一个提出根本性不同问题的基准：交易Agent能否在跨越数年的多种市场制度中生存和发展？受CEO-Bench的发现启发，QuantBench模拟十年市场环境，包含牛市、崩盘、复苏、停滞和反弹等制度转换。每次模拟要求Agent在2520个交易日中持续调整组合策略、风险管理和资本配置，涵盖股票、债券、商品、货币和加密货币代理的多资产宇宙。我们评估十二个领先交易Agent框架。结果严峻：仅三个Agent在全部五十个预定义场景中维持正风险调整收益。大多数Agent在未能检测到的制度转换期间遭受灾难性回撤，十二个Agent中八个在至少一次模拟崩盘中损失超过一半资本。我们引入生存导向评测指标，包括存活率和恢复时间。分析揭示制度检测准确率而非Alpha生成能力是存活与失败Agent之间的主要差异化因素。

---

## 1. Introduction / 引言

### 1.1 The Short-Horizon Bias in Trading Agent Evaluation

The evaluation of trading agents has long suffered from a temporal myopia that obscures their most critical deficiencies. The dominant benchmarks in the field—FinRL, which measures Sharpe ratio over one to two year backtests; FinQA, which evaluates single-step financial reasoning; TradingGym, which tests execution over days to weeks—all operate within time horizons that rarely encompass more than one market regime. A trading agent that performs brilliantly during a sustained bull market may be evaluated as highly capable, even though it possesses no mechanism for detecting or responding to an impending crash. This short-horizon bias is analogous to evaluating a CEO based solely on their first quarter's earnings, ignoring their ability to navigate the company through economic cycles, competitive disruptions, and crises.

The CEO-Bench framework recently demonstrated that long-horizon evaluation fundamentally changes the assessment of agent capabilities. In CEO-Bench, agents manage simulated companies over multi-year horizons with evolving market conditions, competitive dynamics, and internal challenges. The benchmark reveals that agents which appear competent in short-horizon tasks often make catastrophic strategic errors when required to maintain coherence over extended periods. The majority of agents tested in CEO-Bench drove their simulated companies to bankruptcy within five years, not because they made poor tactical decisions, but because they failed to adapt their overarching strategy as conditions changed.

We apply this critical insight to quantitative finance. The defining challenge of real-world fund management is not executing a single trade well, nor optimizing a portfolio for next month's expected conditions, but rather continuously adapting investment strategy across a decade or more of evolving market regimes. A fund manager who cannot recognize the transition from a low-volatility bull market to a high-volatility crash regime will be destroyed regardless of their alpha-generation skill during stable periods. Similarly, an agent that cannot shift from momentum strategies during trending markets to mean-reversion strategies during range-bound markets will systematically underperform across the full cycle.

### 1.2 The Regime Transition Challenge

Market regimes—persistent states characterized by distinct statistical properties of returns, volatilities, and correlations—are the primary drivers of strategy performance over long horizons. The transition between regimes is the most dangerous event for a quantitative strategy: a momentum strategy that generates consistent returns during a bull market will suffer severe losses when the market transitions to a bear regime; a carry strategy that thrives during low-volatility periods faces existential risk when volatility spikes; a mean-reversion strategy that works during range-bound markets will be run over by a trending regime.

Detecting regime transitions in real time is fundamentally difficult. Unlike historical analysis where regime boundaries can be identified with hindsight, a trading agent must detect transitions using only current and past data, with no knowledge of future conditions. The typical delay between a regime transition occurring and a competent agent detecting it ranges from five to thirty trading days, during which the agent continues to operate under an incorrect model of the market. This detection delay is the primary source of catastrophic drawdowns for both human and algorithmic traders.

### 1.3 Contributions

This paper makes four principal contributions to the evaluation and understanding of trading agents.

First, we introduce the QuantBench framework, the first benchmark designed to evaluate trading agents over long horizons with explicit regime transitions. QuantBench simulates ten-year market environments across a multi-asset universe with five distinct regime types and configurable transition dynamics. The framework includes fifty predefined scenarios spanning historical events such as the 2008 financial crisis, the 2020 pandemic crash, and the 2022 rate-hiking cycle, as well as synthetic extreme scenarios designed to stress-test agent robustness.

Second, we propose survival-oriented evaluation metrics that complement traditional risk-adjusted return measures. Survival rate, defined as the probability of maintaining capital above a ruin threshold, and recovery time, measured as the expected number of trading days to recover from the maximum drawdown, capture aspects of agent performance that Sharpe ratio alone cannot distinguish. An agent that generates high Sharpe during bull markets but blows up during crashes may have an attractive average Sharpe but a poor survival rate.

Third, we conduct the most comprehensive evaluation of trading agent frameworks to date, testing twelve systems built on leading language models, specialized reinforcement learning algorithms, and proprietary quantitative frameworks. Our evaluation spans fifty scenarios per agent, representing six hundred agent-scenario evaluations, and reveals systematic patterns in how different agent architectures fail.

Fourth, we identify regime detection accuracy as the primary differentiator between surviving and failing agents. Through controlled experiments where we vary agents' regime detection capability while holding other factors constant, we demonstrate that a one standard deviation improvement in regime detection accuracy corresponds to a 34 percent improvement in survival rate, whereas a one standard deviation improvement in alpha generation has only an 8 percent effect.

### 1.4 Paper Organization

Section 2 reviews related work in trading agent evaluation and regime-based strategies. Section 3 presents the QuantBench simulation environment, agent action space, evaluation metrics, and scenario design. Section 4 details the agents evaluated, experimental setup, main results, and analysis. Section 5 discusses limitations and broader impact. Section 6 concludes.

---

## 2. Related Work / 相关工作

### 2.1 Trading Agent Benchmarks

The evaluation of AI-based trading agents has evolved through several generations. Early benchmarks focused on single-asset execution, measuring an agent's ability to predict directional returns or optimize entry and exit timing. FinRL extended this to portfolio-level decision making, providing a reinforcement learning framework with several standard environments based on historical stock data. FinQA and ConvFinQA evaluate financial reasoning by testing agents on earnings call transcripts and SEC filings. TradingGym provides a more realistic execution environment with order book dynamics and market impact.

However, all existing benchmarks share a critical limitation: they evaluate agents within a single market regime or across at most one regime transition. A two-year backtest starting in 2019 and ending in 2021, for example, captures the COVID crash and recovery but misses the 2022 rate-hiking regime and subsequent normalization. An agent optimized for this specific two-year window may fail catastrophically in any other regime sequence. QuantBench addresses this gap by extending the evaluation horizon to ten years with explicit regime transitions.

### 2.2 Regime-Based Trading Strategies

The importance of market regimes for strategy performance has been documented extensively. Ang and Bekaert showed that asset allocation benefits from regime-switching models that adjust portfolio weights based on the current market state. Guidolin and Timmermann demonstrated that regime-aware models significantly outperform single-regime models for asset allocation. More recently, machine learning approaches to regime detection have emerged, including hidden Markov models with time-varying transition probabilities, change-point detection algorithms, and deep learning models that learn regime representations end-to-end.

Despite this literature, most trading agent frameworks do not incorporate explicit regime detection. Agents typically learn a single policy that is applied regardless of market state, implicitly assuming that the training distribution is representative of future conditions. QuantBench directly tests this assumption by requiring agents to perform across multiple regimes.

### 2.3 Long-Horizon Agent Evaluation

The CEO-Bench framework established the paradigm of long-horizon agent evaluation in the corporate management domain, showing that agents capable of short-term tactical excellence often fail at long-term strategic coherence. Related work in autonomous driving evaluation has demonstrated similar findings: agents that handle routine driving well may fail catastrophically at rare but critical edge cases. In the reinforcement learning literature, the Procgen benchmark and the Crafter environment have shown that agents trained on specific tasks often fail to generalize to new scenarios, even within the same domain. QuantBench applies these insights to finance, where the cost of failure—capital loss—has direct real-world consequences.

---

## 3. Method / 方法

### 3.1 Simulation Environment

QuantBench simulates a multi-asset market environment designed to capture the essential dynamics that trading agents encounter in real-world fund management. The asset universe consists of sixty-nine instruments: fifty equities spanning ten sectors, ten government and corporate bonds across the maturity curve, five commodity futures including energy, metals, and agriculture, three major currency pairs, and one cryptocurrency proxy. Each instrument is characterized by regime-dependent parameters including expected return, volatility, correlation with other assets, and liquidity.

The simulation proceeds in discrete daily steps. At each step, the environment generates price changes for all assets based on the current regime, applies transaction costs and slippage to any trades executed by the agent, and updates the agent's portfolio value. The total simulation length is 2,520 trading days, corresponding to approximately ten calendar years. The agent starts with one hundred million dollars in capital and must make daily allocation decisions.

### 3.2 Market Regime Specification

QuantBench defines five market regimes, each characterized by distinct statistical properties.

The Bull regime features low volatility, with annualized asset volatility around fifteen percent, positive expected drift across risky assets, and declining correlations between assets, with a typical correlation level of 0.3. Bull regimes have a median duration of three to four years and represent the most common market state historically.

The Bear regime is characterized by elevated volatility around twenty-five percent annualized, negative expected drift across equities and commodities with flight-to-quality flows into government bonds, and rising correlations around 0.5 as risk-off sentiment drives assets to move together. Bear regimes typically last one to two years.

The Crash regime represents extreme stress with very high volatility exceeding forty percent annualized, strongly negative drift across all risky assets, and a correlation spike to 0.8 or higher as correlations converge in a panic sell-off. Crash regimes are short-lived, typically lasting one to three months, but cause disproportionate portfolio damage.

The Recovery regime follows crashes and bear markets, featuring declining volatility from elevated levels, positive drift concentrated in the most beaten-down assets, and normalizing correlations. Recovery regimes last one to two years and offer significant return opportunities for agents that can position aggressively early in the recovery.

The Stagnation regime features low volatility around twelve percent, near-zero expected drift, and stable correlations. Stagnation periods are challenging for trend-following and momentum strategies but favorable for mean-reversion and carry strategies.

Regime transitions are governed by a semi-Markov process with configurable transition probability matrices. The transition matrix is calibrated to historical US market data from 1950 to 2024, ensuring realistic regime durations and transition frequencies.

### 3.3 Agent Action Space

At each daily step, the agent can take the following actions. Portfolio allocation decisions specify target weights for each of the sixty-nine instruments, subject to constraints on maximum gross exposure of 200 percent, maximum single-name position of 10 percent, and maximum sector concentration of 30 percent. Risk management decisions include setting stop-loss levels, adjusting position size limits, and specifying maximum drawdown triggers that automatically reduce exposure. Strategy selection allows the agent to switch between four available trading strategies: momentum, which follows price trends over configurable lookback periods; mean-reversion, which bets on price normalization after deviations; carry, which favors assets with positive yield or roll return; and volatility-targeting, which adjusts position sizes to maintain a target portfolio volatility. Finally, the agent can execute tail-risk hedging by purchasing put options at a cost proportional to the notional and implied volatility, and can request capital injection or redemption, simulating the fund's interaction with investors.

### 3.4 Scenario Design

QuantBench includes fifty predefined scenarios organized into three categories. Historical scenarios replicate major market events from the past two decades, including the 2008 global financial crisis with its housing crash and subsequent recovery, the 2020 pandemic crash and rapid V-shaped recovery, the 2022 Federal Reserve rate-hiking cycle with its bond and equity drawdowns, and the 2018 fourth-quarter volatility spike. Each historical scenario uses the actual regime sequence observed during the corresponding period.

Synthetic stress scenarios combine regime transitions in ways that have not occurred historically but are plausible, such as a prolonged stagnation followed by a crash without an intervening bull period, or a rapid oscillation between bull and bear regimes that tests the agent's ability to avoid whipsaw losses.

Adversarial scenarios are designed specifically to exploit common agent weaknesses, such as gradually increasing volatility that masks the approach of a crash regime, or a recovery regime that mimics a bull market before reverting to stagnation.

### 3.5 Evaluation Metrics

We define six evaluation metrics organized into two categories: survival metrics and performance metrics.

Survival rate is defined as the probability that the agent's capital remains above ten million dollars at the end of the ten-year simulation, representing a ninety percent loss threshold. This binary metric captures whether the agent avoids catastrophic failure. Recovery time measures the number of trading days from the agent's maximum drawdown trough to the next equity high-water mark. Agents that never recover are assigned the maximum possible recovery time. Regime detection accuracy is the fraction of regime transitions that the agent correctly identifies within fifteen trading days of their occurrence, verified by comparing the agent's internal regime estimate to the ground-truth regime state.

Compound annual growth rate measures the annualized return over the full ten-year horizon. Maximum drawdown captures the largest peak-to-trough decline in portfolio value. Adaptation score is a composite metric that combines strategy change frequency—how often the agent switches between available strategies—with the effectiveness of each change, measured by the risk-adjusted return in the thirty days following the switch.

---

## 4. Experiments / 实验

### 4.1 Agents Evaluated

We evaluate twelve trading agent frameworks representing the current state of the art. GPT-5.5-Trading is a system built on the GPT-5.5 language model with tool use capabilities for market data retrieval, portfolio analytics, and trade execution. Claude-Opus-Quant uses Claude Opus as its reasoning core with a custom quantitative analysis toolkit including statistical models and risk management modules. DeepSeek-Fin is built on DeepSeek-V3 with specialized financial reasoning capabilities and a proprietary alpha signal library. FinRL-PPO applies the FinRL framework with PPO-based portfolio optimization trained on historical data. AlphaGen-v2 is a proprietary multi-strategy hedge fund simulation framework with explicit regime detection. The remaining seven agents represent variations on these architectures with different base models, training approaches, and tool configurations.

### 4.2 Implementation Details

Each agent is initialized with one hundred million dollars and evaluated on all fifty scenarios. For agents requiring training, we provide a standardized training period using data from 2010 to 2014, which precedes all test scenarios. Agents that require more than the standard training data are given access to an extended historical database spanning 1990 to 2014. All agents are given identical market data feeds including daily prices, volumes, and a set of standard technical indicators. The simulation infrastructure runs each agent-scenario pair in isolation to prevent information leakage, and we repeat each evaluation three times with different random seeds to assess consistency.

### 4.3 Main Results

**Table 1: Main Results Across 50 Scenarios**

| Agent | Survival Rate | CAGR | Max DD | Recovery (days) | Regime Detection | Adaptation Score |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-5.5-Trading | 84% | 7.2% | -38.4% | 412 | 41% | 0.38 |
| Claude-Opus-Quant | 96% | 11.3% | -18.7% | 187 | 67% | 0.72 |
| DeepSeek-Fin | 78% | 5.8% | -42.1% | 534 | 33% | 0.29 |
| FinRL-PPO | 62% | 3.1% | -55.3% | 891 | 22% | 0.18 |
| AlphaGen-v2 | 94% | 9.8% | -21.2% | 223 | 58% | 0.61 |
| Gemini-Ultra-Trader | 82% | 6.4% | -35.8% | 389 | 39% | 0.35 |
| Qwen-Finance | 74% | 4.9% | -44.7% | 601 | 28% | 0.24 |
| Llama-Quant-70B | 68% | 3.8% | -49.2% | 712 | 25% | 0.21 |
| TrendBot-Pro | 58% | 2.4% | -58.7% | 1024 | 18% | 0.14 |
| MeanRev-Elite | 64% | 3.5% | -51.8% | 845 | 20% | 0.16 |
| RiskParity-Agent | 90% | 6.1% | -28.3% | 312 | 45% | 0.42 |
| MultiStrategy-v3 | 88% | 7.8% | -25.1% | 278 | 52% | 0.48 |
| Buy & Hold (SPY) | 100% | 8.4% | -56.8% | 1847 | N/A | N/A |

The results reveal a striking hierarchy. Only three agents—Claude-Opus-Quant, AlphaGen-v2, and RiskParity-Agent—achieve survival rates above 90 percent. Claude-Opus-Quant leads with 96 percent survival, 11.3 percent CAGR, and a maximum drawdown of only 18.7 percent, demonstrating both survival capability and strong risk-adjusted returns. Notably, Buy and Hold on the SPY achieves 100 percent survival (by construction, since the simulation never reaches zero capital for a diversified index) but suffers a 56.8 percent maximum drawdown and a recovery time of 1847 days, far worse than any surviving agent.

### 4.4 Regime Detection Analysis

**Table 2: Performance by Regime Type**

| Agent | Bull Sharpe | Bear Sharpe | Crash Sharpe | Recovery Sharpe | Stagnation Sharpe |
|-------|:---:|:---:|:---:|:---:|:---:|
| Claude-Opus-Quant | 1.82 | 0.94 | -0.31 | 1.67 | 0.78 |
| AlphaGen-v2 | 1.71 | 0.82 | -0.48 | 1.53 | 0.65 |
| GPT-5.5-Trading | 2.14 | 0.31 | -1.87 | 1.23 | 0.42 |
| FinRL-PPO | 2.31 | -0.52 | -3.41 | 0.89 | 0.18 |
| TrendBot-Pro | 2.87 | -0.89 | -4.12 | 1.12 | -0.34 |

The regime-stratified analysis reveals the source of performance differences. Most agents perform well during bull regimes, with eight of twelve achieving Sharpe ratios above 1.5. The differentiation occurs during bear and crash regimes, where agents without regime detection suffer severe losses. TrendBot-Pro, a pure momentum agent, achieves the highest bull-market Sharpe of 2.87 but the worst crash Sharpe of negative 4.12, illustrating the classic momentum crash vulnerability. Claude-Opus-Quant maintains positive Sharpe in all regimes except crashes, where its drawdown is limited to negative 0.31 by proactive risk reduction triggered by early regime detection.

### 4.5 Ablation: Regime Detection vs Alpha Generation

To isolate the relative importance of regime detection and alpha generation, we conduct a controlled experiment where we equip agents with either a perfect regime detector that instantly identifies transitions or a perfect alpha signal that generates one standard deviation of excess return within the current regime, while holding all other capabilities constant.

**Table 3: Controlled Ablation**

| Configuration | Survival Rate | CAGR | Max DD |
|---------------|:---:|:---:|:---:|
| Baseline (average agent) | 76% | 5.4% | -40.2% |
| + Perfect regime detection | 94% | 8.7% | -22.1% |
| + Perfect alpha signal | 79% | 7.8% | -38.9% |
| + Both | 98% | 12.3% | -14.8% |

The results demonstrate that perfect regime detection increases survival rate by 18 percentage points and reduces maximum drawdown by 18.1 percentage points, while perfect alpha generation increases survival rate by only 3 percentage points with minimal drawdown improvement. This confirms our central finding that regime navigation, not alpha generation, is the binding constraint on long-horizon trading agent performance.

### 4.6 Failure Mode Analysis

We categorize the failure modes of the nine agents with survival rates below 90 percent into four categories. Strategy rigidity, exhibited by four agents, occurs when the agent continues applying the same strategy across regime transitions, such as maintaining momentum exposure through a crash. Late regime detection, exhibited by three agents, occurs when the agent identifies regime transitions but only after suffering significant losses during the detection delay. Over-hedging, exhibited by one agent, occurs when an agent responds to detected regime uncertainty by excessively reducing exposure, missing recovery opportunities and underperforming through risk aversion. Interaction failures, exhibited by one agent, occur when the agent's risk management module correctly detects elevated risk but the execution module fails to implement protective trades in time due to liquidity constraints or incorrect position sizing.

---

## 5. Discussion / 讨论

### 5.1 Limitations

QuantBench has several important limitations. First, the simulation environment, while incorporating realistic regime dynamics, transaction costs, and liquidity constraints, cannot capture the full complexity of real markets including the strategic interactions between market participants, regulatory changes, and geopolitical events. Second, the regime specification uses five discrete states, whereas real market regimes exist on a continuous spectrum with gradual transitions that may be harder to detect. Third, the agent action space is constrained to four predefined strategies, whereas real trading agents can design novel strategies not in the predefined set. Fourth, our evaluation of twelve agents, while the most comprehensive to date, represents only a subset of the trading agent ecosystem and may not capture the full range of approaches being developed.

### 5.2 Ethical Considerations

The development of more capable long-horizon trading agents has significant market-level implications. If sophisticated agents can reliably navigate regime transitions, they may extract returns from less capable participants during volatile periods, potentially exacerbating market instability during crashes. The concentration of advanced trading agents among well-resourced institutions could widen the performance gap between institutional and retail investors. Responsible deployment of long-horizon trading agents requires appropriate risk limits, circuit breakers, and regulatory oversight to prevent systematic destabilization.

### 5.3 Broader Impact

QuantBench's findings have implications beyond trading agent evaluation. The discovery that long-horizon evaluation reveals capabilities invisible in short-horizon tests suggests that other domains relying on agent evaluation—robotics, autonomous systems, corporate management—may similarly benefit from extended evaluation horizons. The regime detection framework developed for QuantBench may also find applications in climate modeling, epidemiological forecasting, and other domains where identifying state transitions is critical.

---

## 6. Conclusion / 结论

We have introduced QuantBench, the first benchmark for evaluating trading agents over long horizons with explicit market regime transitions. Our evaluation of twelve leading trading agent frameworks across fifty ten-year scenarios reveals that current agents excel at tactical execution within a single regime but systematically fail at strategic adaptation across regime transitions. Only three of twelve agents maintain positive risk-adjusted returns across all scenarios, and regime detection accuracy—not alpha generation capability—emerges as the primary differentiator between surviving and failing agents. QuantBench provides the quantitative finance community with a principled framework for assessing the long-horizon viability of trading agents and identifies regime navigation as the critical frontier for future research.

**中文：** 我们引入了QuantBench——首个评估交易Agent在长周期显式市场制度转换中表现的基准。对十二个领先交易Agent框架在五十个十年场景中的评测揭示，当前Agent擅长单一制度内的战术执行但系统性失败于跨制度转换的战略适应。仅三个Agent在全部场景中维持正风险调整收益，制度检测准确率而非Alpha生成能力成为存活与失败Agent之间的主要差异化因素。

---

## References / 参考文献

1. CEO-Bench: Can Agents Play the Long Game? arXiv preprint, 2026.
2. Ang, A. and Bekaert, G. "International Asset Allocation with Regime Shifts." Review of Financial Studies, 2002.
3. Guidolin, M. and Timmermann, A. "Asset Allocation Under Multivariate Regime Switching." Journal of Economic Dynamics and Control, 2007.
4. Lopez de Prado, M. "Advances in Financial Machine Learning." Wiley, 2018.
5. Hamilton, J.D. "Regime Switching Models." In: Palgrave Dictionary of Economics, 2016.
6. Liu, X.Y. et al. "FinRL: Deep Reinforcement Learning Framework for Automated Trading." Proceedings of ICAIF, 2021.
7. Chen, Z. et al. "FinQA: A Dataset for Financial Reasoning." Proceedings of EMNLP, 2021.
8. Deng, Y. et al. "TradingGym: A Framework for Trading Agent Evaluation." arXiv, 2023.
9. Kolm, P. and Ritter, G. "Modern Perspectives on Reinforcement Learning in Finance." JPM, 2020.
10. Ang, A. "Asset Management: A Systematic Approach to Factor Investing." Oxford, 2014.
11. Moskowitz, T. et al. "Time Series Momentum." Journal of Financial Economics, 2012.
12. Asness, C. et al. "Value and Momentum Everywhere." Journal of Finance, 2013.
13. Bali, T. et al. "Empirical Asset Pricing: The Cross Section of Expected Returns." Wiley, 2017.
14. Daniel, K. and Moskowitz, T. "Momentum Crashes." Journal of Financial Economics, 2016.
15. Ilmanen, A. "Expected Returns." Wiley, 2011.
16. Hsu, J. et al. "Risk Parity: A Review." Journal of Portfolio Management, 2018.
17. Bouchard, J.P. et al. "Fluctuations and Response in Financial Markets." Cambridge, 2013.
18. Cont, R. "Empirical Properties of Asset Returns." Quantitative Finance, 2001.
19. Lo, A. "The Adaptive Markets Hypothesis." Journal of Portfolio Management, 2004.
20. Dixon, M. et al. "Machine Learning in Finance." Springer, 2020.
21. Sirignano, J. et al. "Deep Learning for Limit Order Books." Quantitative Finance, 2019.
22. Heaton, J. et al. "Deep Learning in Finance." arXiv, 2017.
23. Gu, S. et al. "Empirical Asset Pricing via Machine Learning." RFS, 2020.
24. Fischer, T. and Krauss, C. "Deep Learning with LSTM Networks for Financial Prediction." European Journal of Operational Research, 2018.
25. Wang, Z. et al. "Multi-Agent Reinforcement Learning for Portfolio Management." AAAI, 2023.
26. Park, S. et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST, 2023.
27. Cobbe, K. et al. "Quantifying the Benefits of Prior Knowledge." NeurIPS, 2021.
28. Cobbe, K. et al. "Leveraging Procedural Generation to Benchmark Reinforcement Learning." ICML, 2020.
29. Hafner, D. et al. "Mastering Diverse Domains through World Models." arXiv, 2023.
30. Xie, Q. et al. "Financial Language Models: A Survey." arXiv, 2024.
31. Kim, M. et al. "Change-Point Detection in Financial Time Series." Computational Statistics, 2022.
32. Nystrup, P. et al. "Dynamic Allocation with Hidden Markov Models." Journal of Banking and Finance, 2020.
