"""
data.py - Data generation for Investment Societies of Thought.

Generates synthetic investment reasoning datasets including:
- Expert debate transcripts for Phase 1 SFT
- Financial QA pairs for evaluation
- Investment theses with bull/bear/risk/macro annotations
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class ExpertRole(Enum):
    """Expert roles in the investment committee simulation."""
    BULL_ANALYST = "bull_analyst"
    BEAR_ANALYST = "bear_analyst"
    RISK_MANAGER = "risk_manager"
    MACRO_STRATEGIST = "macro_strategist"
    SYNTHESIZER = "synthesizer"


class AssetClass(Enum):
    """Asset classes for cross-asset evaluation."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FOREIGN_EXCHANGE = "foreign_exchange"


class Conviction(Enum):
    """Investment conviction levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExpertStatement:
    """A single statement by an expert in a debate."""
    role: ExpertRole
    text: str
    confidence: float  # 0-1
    key_arguments: List[str] = field(default_factory=list)


@dataclass
class InvestmentDebate:
    """A complete multi-expert debate on an investment thesis."""
    debate_id: str
    thesis: str
    asset_class: AssetClass
    ticker: str
    statements: List[ExpertStatement] = field(default_factory=list)
    final_recommendation: str = ""
    conviction: Conviction = Conviction.MEDIUM

    def to_dict(self) -> Dict:
        return {
            "debate_id": self.debate_id,
            "thesis": self.thesis,
            "asset_class": self.asset_class.value,
            "ticker": self.ticker,
            "statements": [
                {"role": s.role.value, "text": s.text, "confidence": s.confidence}
                for s in self.statements
            ],
            "final_recommendation": self.final_recommendation,
            "conviction": self.conviction.value,
        }


@dataclass
class FinancialQA:
    """A financial question-answering pair for evaluation."""
    qa_id: str
    question: str
    reasoning: str
    answer: str
    difficulty: str  # "easy", "medium", "hard"
    asset_class: AssetClass

    def to_dict(self) -> Dict:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "asset_class": self.asset_class.value,
        }


# Templates for synthetic data generation
THESIS_TEMPLATES: Dict[AssetClass, List[str]] = {
    AssetClass.EQUITY: [
        "{ticker} is undervalued given its 15x forward P/E ratio, well below the sector median of 22x, with earnings growth projected at 18% CAGR over the next three years.",
        "{ticker}'s recent acquisition of a competitor positions it as the market leader in cloud infrastructure, with expected margin expansion of 300bps over the next two fiscal years.",
        "{ticker} faces significant headwinds from rising input costs and intensifying competition, suggesting a defensive position is warranted despite strong historical returns.",
    ],
    AssetClass.FIXED_INCOME: [
        "Duration extension in investment-grade corporates offers attractive risk-adjusted returns as the Fed nears the end of its tightening cycle.",
        "High-yield credit spreads are at historical tights (320bps), suggesting limited compensation for default risk in a potential downturn scenario.",
        "{ticker}'s senior secured bonds at 6.2% yield offer a compelling risk-reward given the company's improving cash flow trajectory and asset coverage.",
    ],
    AssetClass.COMMODITY: [
        "Copper futures present an asymmetric opportunity as global electrification demand outpaces supply additions, with a structural deficit expected by 2026.",
        "Crude oil faces near-term demand destruction risk from accelerating EV adoption, despite OPEC+ supply discipline maintaining a floor near $70/bbl.",
        "Gold allocation serves as effective portfolio insurance given elevated geopolitical uncertainty and central bank buying at record levels.",
    ],
    AssetClass.FOREIGN_EXCHANGE: [
        "JPY weakness is overextended relative to real interest rate differentials, suggesting a mean-reversion opportunity over the next quarter.",
        "EUR/USD faces downside pressure from widening growth differentials and potential energy crisis recurrence in the eurozone.",
        "EM currencies broadly offer carry attractiveness with real yields 200-400bps above DM, though idiosyncratic risk requires careful selection.",
    ],
}

EXPERT_ARGUMENTS: Dict[ExpertRole, List[str]] = {
    ExpertRole.BULL_ANALYST: [
        "The company's competitive moat is widening, as evidenced by increasing market share and pricing power over the past four quarters.",
        "Valuation is compelling at current levels, trading at a 30% discount to our DCF estimate with significant optionality from new product launches.",
        "Management's capital allocation track record is excellent, with ROIC consistently above 15% and disciplined M&A execution.",
        "Earnings momentum is accelerating, with the last three quarters showing sequential revenue growth improvement.",
    ],
    ExpertRole.BEAR_ANALYST: [
        "The company's debt load is concerning at 4.5x EBITDA, leaving limited financial flexibility in a rising rate environment.",
        "Competitive threats from emerging players are underappreciated, with market share losses of 200bps over the trailing twelve months.",
        "Valuation assumes continued growth that is unsustainable given market saturation and demographic headwinds.",
        "Regulatory risk is elevated, with pending legislation that could compress margins by 500-800bps if enacted.",
    ],
    ExpertRole.RISK_MANAGER: [
        "Portfolio concentration in this sector exceeds our 15% limit, requiring position sizing that caps upside participation.",
        "Correlation with existing holdings is 0.78, meaning this position adds limited diversification benefit.",
        "Maximum drawdown scenario analysis shows a potential 35% loss in a stress case, exceeding our per-position risk budget.",
        "Liquidity profile is adequate for our expected holding period, with 20-day average volume supporting full exit in 3 trading days.",
    ],
    ExpertRole.MACRO_STRATEGIST: [
        "Current monetary policy stance is supportive, with real rates declining and credit conditions easing for investment-grade issuers.",
        "The macro regime is transitioning from late-cycle to recession, historically unfavorable for this sector's performance.",
        "Cross-asset signals are mixed: equity risk premium is compressed while credit spreads are widening, suggesting caution.",
        "Geopolitical risk premium is elevated, with trade tensions creating binary outcomes that are difficult to price into positions.",
    ],
}


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    n_debates_per_asset: int = 50
    n_qa_per_asset: int = 80
    seed: int = 42
    tickers: Dict[str, List[str]] = field(default_factory=lambda: {
        "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "UNH", "XOM", "PG"],
        "fixed_income": ["IG_CORP", "HY_INDEX", "TREASURY_10Y", "MUNI_AA", "EMB_INDEX"],
        "commodity": ["COPPER", "CRUDE_WTI", "GOLD", "NAT_GAS", "WHEAT"],
        "foreign_exchange": ["EUR/USD", "USD/JPY", "GBP/USD", "USD/EM", "AUD/USD"],
    })


class InvestmentDataGenerator:
    """Generates synthetic investment reasoning datasets.

    Creates debate transcripts, financial QA pairs, and investment theses
    with multi-expert annotations for training and evaluation.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def _generate_expert_statement(
        self, role: ExpertRole, ticker: str, asset_class: AssetClass
    ) -> ExpertStatement:
        """Generate a synthetic expert statement."""
        args = EXPERT_ARGUMENTS[role]
        selected_args = list(self.rng.choice(args, size=min(2, len(args)), replace=False))
        confidence = float(self.rng.uniform(0.4, 0.95))

        text = f"[{role.value.replace('_', ' ').title()}] "
        text += f"Regarding {ticker} ({asset_class.value}): "
        text += " ".join(selected_args)

        return ExpertStatement(
            role=role,
            text=text,
            confidence=confidence,
            key_arguments=selected_args,
        )

    def generate_debates(self) -> List[InvestmentDebate]:
        """Generate synthetic multi-expert debate transcripts.

        Returns:
            List of InvestmentDebate objects.
        """
        debates: List[InvestmentDebate] = []
        debate_counter = 0

        for asset_class in AssetClass:
            tickers = self.config.tickers.get(asset_class.value, ["GENERIC"])
            templates = THESIS_TEMPLATES[asset_class]

            for i in range(self.config.n_debates_per_asset):
                ticker = tickers[i % len(tickers)]
                template = templates[i % len(templates)]
                thesis = template.format(ticker=ticker)

                debate = InvestmentDebate(
                    debate_id=f"debate_{debate_counter:05d}",
                    thesis=thesis,
                    asset_class=asset_class,
                    ticker=ticker,
                )

                # Generate statements from each expert
                roles = [
                    ExpertRole.BULL_ANALYST,
                    ExpertRole.BEAR_ANALYST,
                    ExpertRole.RISK_MANAGER,
                    ExpertRole.MACRO_STRATEGIST,
                ]
                for role in roles:
                    statement = self._generate_expert_statement(role, ticker, asset_class)
                    debate.statements.append(statement)

                # Synthesis
                bull_strength = debate.statements[0].confidence
                bear_strength = debate.statements[1].confidence
                risk_concern = debate.statements[2].confidence

                net_score = bull_strength - bear_strength - 0.3 * risk_concern

                if net_score > 0.3:
                    debate.conviction = Conviction.HIGH
                    debate.final_recommendation = f"BUY {ticker}: Bull case dominates with manageable risks."
                elif net_score > -0.1:
                    debate.conviction = Conviction.MEDIUM
                    debate.final_recommendation = f"HOLD {ticker}: Mixed signals warrant cautious positioning."
                else:
                    debate.conviction = Conviction.LOW
                    debate.final_recommendation = f"AVOID {ticker}: Risk-reward unfavorable at current levels."

                debates.append(debate)
                debate_counter += 1

        return debates

    def generate_financial_qa(self) -> List[FinancialQA]:
        """Generate synthetic financial QA pairs for evaluation.

        Returns:
            List of FinancialQA objects.
        """
        qa_templates = [
            {
                "question": "What is the forward P/E ratio of {ticker} given current price of ${price} and estimated EPS of ${eps}?",
                "reasoning": "Forward P/E = Price / Estimated EPS = ${price} / ${eps} = {result}x",
                "difficulty": "easy",
            },
            {
                "question": "If {ticker}'s revenue grows at {growth}% CAGR for 3 years from ${rev}B base, what will be the projected revenue?",
                "reasoning": "Projected revenue = ${rev}B * (1 + {growth}/100)^3 = ${projected}B",
                "difficulty": "medium",
            },
            {
                "question": "Given {ticker}'s beta of {beta} and market return expectation of {mkt_ret}%, what is the required return using CAPM with risk-free rate of {rf}%?",
                "reasoning": "Required return = {rf}% + {beta} * ({mkt_ret}% - {rf}%) = {capm}%",
                "difficulty": "medium",
            },
            {
                "question": "What is the maximum portfolio allocation to {ticker} given a 5% VaR constraint, current volatility of {vol}%, and portfolio size of $1M?",
                "reasoning": "Max allocation = VaR_budget / (z_95 * vol * sqrt(10/252)) = 50000 / (1.645 * {vol}/100 * 0.199) = ${max_alloc}",
                "difficulty": "hard",
            },
        ]

        qa_list: List[FinancialQA] = []
        qa_counter = 0

        for asset_class in AssetClass:
            tickers = self.config.tickers.get(asset_class.value, ["GENERIC"])

            for i in range(self.config.n_qa_per_asset):
                ticker = tickers[i % len(tickers)]
                template = qa_templates[i % len(qa_templates)]

                price = float(self.rng.uniform(50, 500))
                eps = float(self.rng.uniform(2, 30))
                growth = float(self.rng.uniform(5, 25))
                rev = float(self.rng.uniform(10, 200))
                beta = float(self.rng.uniform(0.5, 2.0))
                mkt_ret = float(self.rng.uniform(6, 12))
                rf = float(self.rng.uniform(2, 5))
                vol = float(self.rng.uniform(15, 45))

                # Compute answers
                pe = price / eps
                projected = rev * (1 + growth / 100) ** 3
                capm = rf + beta * (mkt_ret - rf)
                max_alloc = 50000 / (1.645 * vol / 100 * 0.199)

                question = template["question"].format(
                    ticker=ticker, price=f"{price:.2f}", eps=f"{eps:.2f}",
                    growth=f"{growth:.1f}", rev=f"{rev:.0f}",
                    projected=f"{projected:.1f}", beta=f"{beta:.2f}",
                    mkt_ret=f"{mkt_ret:.1f}", rf=f"{rf:.1f}",
                    capm=f"{capm:.1f}", vol=f"{vol:.1f}",
                    max_alloc=f"{max_alloc:,.0f}",
                )
                reasoning = template["reasoning"].format(
                    price=f"{price:.2f}", eps=f"{eps:.2f}",
                    result=f"{pe:.1f}", growth=f"{growth:.1f}",
                    rev=f"{rev:.0f}", projected=f"{projected:.1f}",
                    beta=f"{beta:.2f}", mkt_ret=f"{mkt_ret:.1f}",
                    rf=f"{rf:.1f}", capm=f"{capm:.1f}",
                    vol=f"{vol:.1f}", max_alloc=f"{max_alloc:,.0f}",
                )

                qa_list.append(FinancialQA(
                    qa_id=f"qa_{qa_counter:05d}",
                    question=question,
                    reasoning=reasoning,
                    answer=str(round(pe if template["difficulty"] == "easy" else projected if template["difficulty"] == "medium" and "revenue" in template["question"] else capm if "CAPM" in template["question"] else max_alloc, 2)),
                    difficulty=template["difficulty"],
                    asset_class=asset_class,
                ))
                qa_counter += 1

        return qa_list

    def format_debate_for_sft(self, debate: InvestmentDebate) -> Dict:
        """Format a debate as an SFT training example.

        Args:
            debate: InvestmentDebate object.

        Returns:
            Dictionary with 'prompt' and 'completion' keys.
        """
        prompt = f"Analyze the following investment thesis using multi-expert debate:\n\nThesis: {debate.thesis}\n\n"

        completion_parts = []
        for stmt in debate.statements:
            role_name = stmt.role.value.replace("_", " ").upper()
            completion_parts.append(f"## {role_name}\n{stmt.text}\n(Confidence: {stmt.confidence:.2f})")

        completion_parts.append(f"## SYNTHESIS\n{debate.final_recommendation}")
        completion_parts.append(f"Conviction: {debate.conviction.value.upper()}")

        return {
            "prompt": prompt,
            "completion": "\n\n".join(completion_parts),
        }

    def save_datasets(
        self,
        debates: List[InvestmentDebate],
        qa_list: List[FinancialQA],
        output_dir: str,
    ) -> None:
        """Save generated datasets to JSON files.

        Args:
            debates: List of debate objects.
            qa_list: List of QA objects.
            output_dir: Output directory path.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save debates
        debate_data = [d.to_dict() for d in debates]
        with open(os.path.join(output_dir, "debates.json"), "w") as f:
            json.dump(debate_data, f, indent=2)

        # Save QA
        qa_data = [q.to_dict() for q in qa_list]
        with open(os.path.join(output_dir, "financial_qa.json"), "w") as f:
            json.dump(qa_data, f, indent=2)

        # Save SFT-formatted debates
        sft_data = [self.format_debate_for_sft(d) for d in debates]
        with open(os.path.join(output_dir, "sft_debates.json"), "w") as f:
            json.dump(sft_data, f, indent=2)


if __name__ == "__main__":
    config = DatasetConfig(n_debates_per_asset=10, n_qa_per_asset=15)
    gen = InvestmentDataGenerator(config)

    debates = gen.generate_debates()
    qa_list = gen.generate_financial_qa()

    print(f"Generated {len(debates)} debates")
    print(f"Generated {len(qa_list)} QA pairs")

    from collections import Counter
    asset_dist = Counter(d.asset_class.value for d in debates)
    print(f"Debates by asset: {dict(asset_dist)}")

    conv_dist = Counter(d.conviction.value for d in debates)
    print(f"Conviction distribution: {dict(conv_dist)}")

    diff_dist = Counter(q.difficulty for q in qa_list)
    print(f"QA difficulty: {dict(diff_dist)}")

    # Show a formatted example
    example = gen.format_debate_for_sft(debates[0])
    print(f"\nExample SFT prompt:\n{example['prompt'][:200]}...")
    print(f"Example SFT completion:\n{example['completion'][:300]}...")
