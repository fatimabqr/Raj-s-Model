"""All model assumptions in one place (mirrors the Model_Config sheet)."""
from dataclasses import dataclass, field, asdict
from datetime import date

GRADE_BINS = [  # (min sector percentile, letter, numeric equivalent)
    (0.00, "F", 1.00), (0.06, "D-", 1.33), (0.12, "D", 1.67), (0.20, "D+", 2.00),
    (0.28, "C-", 2.33), (0.36, "C", 2.67), (0.44, "C+", 3.00), (0.52, "B-", 3.33),
    (0.60, "B", 3.67), (0.68, "B+", 4.00), (0.76, "A-", 4.33), (0.84, "A", 4.67),
    (0.92, "A+", 5.00),
]

FACTOR_METRICS = {
    "value":         ["pe", "ev_ebitda", "ps", "pb", "fcf_yield"],
    "growth":        ["rev_g", "ebitda_g", "eps_g"],
    "profitability": ["gross_m", "oper_m", "net_m", "roe", "fcf_m"],
    "momentum":      ["ret_3m", "ret_6m", "ret_12m"],
    "sentiment":     ["bullish"],
}
LOWER_IS_BETTER = {"pe", "ev_ebitda", "ps", "pb"}   # valuation multiples


@dataclass
class Config:
    # factor weights
    w_value: float = 2.0
    w_growth: float = 1.0
    w_profitability: float = 1.0
    w_momentum: float = 2.0
    w_sentiment: float = 1.0
    # quality circuit-breaker
    min_pct_gms: float = 0.28        # growth / momentum / sentiment
    min_pct_vp: float = 0.12         # value / profitability
    disq_ceiling: float = 3.49
    # rating tiers (minimum Quant Score)
    t_strong_buy: float = 4.5
    t_buy: float = 3.5
    t_hold: float = 2.5
    t_sell: float = 1.5
    # TradeQuest rules
    capital: float = 4_000_000
    max_position: float = 0.20
    max_cash_pct: float = 0.10
    max_positions: int = 30
    max_daily_trades: int = 5
    min_monthly_trades: int = 10
    fee_per_trade: float = 5.0
    week1_start: date = date(2026, 11, 11)
    week1_end: date = date(2026, 11, 13)
    # BIBF risk overlay
    base_risk: float = 0.01
    atr_mult: float = 2.0
    practical_cap: float = 0.085
    beta_risk_on: float = 1.50
    beta_def_ceiling: float = 0.85
    beta_def_floor: float = 0.60
    vix_stress: float = 25.0
    regime_mult_defensive: float = 0.5
    # eligibility
    min_mkt_cap_b: float = 10.0
    min_price: float = 10.0
    # window
    inception: date = date(2026, 11, 11)
    valuation: date = date(2026, 11, 20)

    def weights(self):
        return {"value": self.w_value, "growth": self.w_growth, "profitability": self.w_profitability,
                "momentum": self.w_momentum, "sentiment": self.w_sentiment}

    def as_dict(self):
        return asdict(self)
