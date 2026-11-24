"""5-factor sector-relative scoring engine (port of Quant_Scoring_Engine)."""
import numpy as np
import pandas as pd
from .config import Config, GRADE_BINS, FACTOR_METRICS, LOWER_IS_BETTER

EXCLUDE_NAME_HINTS = ("depositary", "depository", "preferred", "pfd", "units", "perpetual", "warrant", " rights")


def is_non_common(row) -> bool:
    name = str(row.get("name", "")).lower()
    return any(h in name for h in EXCLUDE_NAME_HINTS)


def _sector_pct(s: pd.Series, lower_better: bool) -> pd.Series:
    """Sector percentile with half-credit for ties (identical to the Excel COUNTIFS construction)."""
    out = pd.Series(np.nan, index=s.index)
    ok = s.notna()
    if not ok.any():
        return out
    x = s[ok]
    n = len(x)
    eq = x.groupby(x).transform("count")
    if lower_better:
        better = n - x.rank(method="max")            # peers with a HIGHER multiple (more expensive)
    else:
        better = x.rank(method="min") - 1            # peers with a LOWER value
    pct = (better + 0.5 * eq) / n
    if lower_better:
        pct[x <= 0] = 0.05                           # negative / zero multiples are penalised
    out[ok] = pct
    return out


def letter(p: float) -> str:
    g = "F"
    for lo, lt, _ in GRADE_BINS:
        if p >= lo:
            g = lt
    return g


def score_universe(df: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    d = df.copy()
    d["excluded"] = d.apply(is_non_common, axis=1)
    d = d[~d["excluded"]].reset_index(drop=True)

    # 1) metric percentiles within GICS sector
    for factor, metrics in FACTOR_METRICS.items():
        for m in metrics:
            d["p_" + m] = d.groupby("sector", group_keys=False)[m].apply(lambda s: _sector_pct(s, m in LOWER_IS_BETTER))
        d[factor + "_pct"] = d[["p_" + m for m in metrics]].mean(axis=1, skipna=True)
    d["sentiment_pct"] = d["sentiment_pct"].fillna(0.5)            # missing analyst data -> neutral
    for f in ("value", "growth", "profitability", "momentum"):
        d[f + "_pct"] = d[f + "_pct"].fillna(0.5)

    # 2) weighted composite and universe-wide rank -> raw score 1..5
    w = cfg.weights()
    d["composite"] = sum(d[f + "_pct"] * w[f] for f in w) / sum(w.values())
    n = len(d)
    less = d["composite"].rank(method="min") - 1
    eq = d.groupby("composite")["composite"].transform("count")
    d["raw_score"] = 1 + 4 * (less + 0.5 * eq) / n

    # 3) quality circuit-breaker (hold-cap)
    d["disqualified"] = ((d.growth_pct < cfg.min_pct_gms) | (d.momentum_pct < cfg.min_pct_gms) |
                         (d.sentiment_pct < cfg.min_pct_gms) | (d.value_pct < cfg.min_pct_vp) |
                         (d.profitability_pct < cfg.min_pct_vp))
    d["quant_score"] = np.where(d.disqualified & (d.raw_score > cfg.disq_ceiling), cfg.disq_ceiling, d.raw_score)

    def rating(s):
        return ("Strong Buy" if s >= cfg.t_strong_buy else "Buy" if s >= cfg.t_buy else
                "Hold" if s >= cfg.t_hold else "Sell" if s >= cfg.t_sell else "Strong Sell")
    d["rating"] = d.quant_score.map(rating)
    for f in ("value", "growth", "profitability", "momentum", "sentiment"):
        d["g_" + f] = d[f + "_pct"].map(letter)
    d["eligible"] = ((d.rating == "Strong Buy") & (d.sector != "Real Estate") &
                     (d.mkt_cap_b >= cfg.min_mkt_cap_b) & (d.price > cfg.min_price))
    return d
