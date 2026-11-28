"""One call that runs the whole framework and returns every table the UI/export needs."""
import numpy as np
import pandas as pd
from .config import Config
from .scoring import score_universe
from . import portfolio as pf


def resolve_ma(inputs: pd.DataFrame, market: dict, cfg: Config):
    """Pick EMA or SMA versions of ATR / 50-DMA when the data source provides both (live data).
    The frozen workbook snapshot carries single hard-coded values, so it is passed through unchanged."""
    inputs = inputs.copy(); market = dict(market)
    t = cfg.ma_type.lower()
    if f"dma50_{t}" in inputs.columns:
        inputs["dma50"] = inputs[f"dma50_{t}"]; inputs["atr14"] = inputs[f"atr14_{t}"]
    if f"spx_50dma_{t}" in market:
        market["spx_50dma"] = market[f"spx_50dma_{t}"]
    return inputs, market


def run_model(universe, inputs, market, ledger, cfg: Config = Config(), asof=None):
    inputs, market = resolve_ma(inputs, market, cfg)
    led = pf.load_ledger(ledger, asof=asof)
    if led.empty:
        raise ValueError("No trades on or before the as-of date — the portfolio is empty then.")
    scored = score_universe(universe, cfg)
    prices = scored.set_index("ticker")["price"].to_dict()
    held = sorted(led.ticker.unique())
    missing = [t for t in held if t not in prices]
    if missing:
        raise ValueError("Ledger tickers missing from the universe/price data: %s" % missing)

    op, cl, cash, fees = pf.build_book(led, prices, cfg)
    if op.empty:
        raise ValueError("No open positions as of this date (all trades are closed round trips).")
    nav = cash + op.market_value.sum()
    reg = pf.regime(market, cfg)
    pos = pf.enrich_positions(op, scored, inputs, nav, reg, cfg)

    closed = cl.merge(scored[["ticker", "name", "sector", "quant_score", "rating"]], on="ticker", how="left")

    cands = scored[scored.eligible].merge(inputs[["ticker", "beta", "atr14", "dma50"]], on="ticker", how="left")
    cands["trend_gate"] = np.where(cands.dma50.isna(), "Data needed",
                                   np.where(cands.price > cands.dma50, "Above 50-DMA", "Below 50-DMA"))
    cands["stop_2atr"] = cands.price - cfg.atr_mult * cands.atr14
    cands["regime_target"] = np.minimum(nav * cfg.practical_cap,
                                        (nav * reg["risk_budget"]) / (cfg.atr_mult * cands.atr14) * cands.price)
    cands["held"] = cands.ticker.isin(pos.ticker)
    cands["signal"] = np.where(cands.held, "Already Held",
                       np.where(cands.trend_gate == "Data needed", "Input beta/ATR/50-DMA",
                       np.where(cands.trend_gate == "Above 50-DMA", "Gear 1: Buy Now", "Gear 2 Watchlist")))
    cands = cands.sort_values("quant_score", ascending=False).reset_index(drop=True)

    plan, post = pf.rotation_plan(pos, cands[cands.trend_gate != "Data needed"], nav, reg, cfg)
    gov = pf.governance(led, None, nav, cash, fees, pos, reg, cfg)
    tc = pf.trade_compliance(led, cfg)
    dz = pf.dietz(led, nav, cfg)
    bench = market["spx_end"] / market["spx_start"] - 1
    summary = dict(
        nav=nav, cash=cash, cash_pct=cash / cfg.capital, invested=pos.market_value.sum(), fees=fees,
        ret=nav / cfg.capital - 1, bench=bench, rel_bps=(nav / cfg.capital - 1 - bench) * 1e4,
        dietz_ret=dz["ret"], dietz_rel_bps=(dz["ret"] - bench) * 1e4, avg_deployed=dz["avg_deployed"],
        unreal=pos.unreal_pl.sum(), realized=closed.realized_pl.sum() if len(closed) else 0.0,
        beta=float((pos.market_value * pos.beta).sum() / nav),
        port_score=float((pos.market_value * pos.quant_score).sum() / pos.market_value.sum()),
        regime=reg)
    return dict(ledger_used=led, scored=scored, positions=pos, closed=closed, candidates=cands, rotation=plan, post_rotation=post,
                governance=gov, ledger=tc, cash_path=pf.cash_path(led, cfg), summary=summary, market=market)
