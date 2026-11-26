"""Portfolio audit, TradeQuest compliance, regime/risk overlay, Modified Dietz and rotation logic."""
import numpy as np
import pandas as pd
from .config import Config


# ---------------------------------------------------------------- ledger
def load_ledger(src) -> pd.DataFrame:
    """src: path / file-like / DataFrame with columns ticker, order_type, quantity, price, net_cash, timestamp.
    quantity is signed (sells negative). If net_cash is missing it is derived as -quantity*price."""
    d = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    d.columns = [c.strip().lower() for c in d.columns]
    d["ticker"] = d["ticker"].str.strip().str.upper()
    d["ts"] = pd.to_datetime(d["timestamp"].astype(str).str.replace(" - ", " ", regex=False), errors="coerce")
    if d["ts"].isna().any():
        raise ValueError("Unparseable timestamps in ledger rows: %s" % list(d.index[d["ts"].isna()]))
    if "net_cash" not in d.columns:
        d["net_cash"] = -d["quantity"] * d["price"]
    d["date"] = d["ts"].dt.date
    return d.sort_values("ts").reset_index(drop=True)


def trade_compliance(led: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = led.copy()
    d["same_day_trades"] = d.groupby("date")["ticker"].transform("count")
    wk1 = (d["date"] >= cfg.week1_start) & (d["date"] <= cfg.week1_end)
    d["daily_limit_check"] = np.where(wk1, "Week 1 Exempt",
                             np.where(d.same_day_trades <= cfg.max_daily_trades, "Compliant", "BREACH"))
    return d


def cash_path(led: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """End-of-day cash and cash % of initial capital (fees deducted)."""
    d = led.copy()
    d["cash_flow"] = d["net_cash"] - cfg.fee_per_trade
    eod = d.groupby("date")["cash_flow"].sum().cumsum() + cfg.capital
    out = eod.rename("cash").to_frame()
    out["cash_pct_initial"] = out["cash"] / cfg.capital
    out["over_ceiling"] = (out.index > cfg.week1_end) & (out["cash_pct_initial"] > cfg.max_cash_pct)
    return out


# ---------------------------------------------------------------- book
def build_book(led: pd.DataFrame, prices: dict, cfg: Config):
    """Returns (open_positions, closed_positions, cash, fees)."""
    fees = cfg.fee_per_trade * len(led)
    cash = cfg.capital + led["net_cash"].sum() - fees
    rows_o, rows_c = [], []
    for t, g in led.groupby("ticker"):
        net = g["quantity"].sum()
        buys = g[g.quantity > 0]; sells = g[g.quantity < 0]
        buy_cost = -buys["net_cash"].sum()
        proceeds = sells["net_cash"].sum()
        if abs(net) < 1e-9:
            rows_c.append(dict(ticker=t, shares=buys.quantity.sum(), buy_cost=buy_cost, sell_proceeds=proceeds,
                               realized_pl=proceeds - buy_cost, realized_ret=(proceeds - buy_cost) / buy_cost))
        else:
            rows_o.append(dict(ticker=t, shares=net, cost_basis=buy_cost, avg_cost=buy_cost / net))
    op = pd.DataFrame(rows_o); cl = pd.DataFrame(rows_c)
    if len(op):
        op["price"] = op.ticker.map(prices)
        op["market_value"] = op.shares * op.price
        op["unreal_pl"] = op.market_value - op.cost_basis
        op["unreal_ret"] = op.unreal_pl / op.cost_basis
    return op, cl, cash, fees


def regime(mkt: dict, cfg: Config) -> dict:
    stressed = (mkt["spx_end"] < mkt["spx_50dma"]) or (mkt["vix_end"] >= cfg.vix_stress)
    mult = cfg.regime_mult_defensive if stressed else 1.0
    return dict(multiplier=mult, defensive=stressed, risk_budget=cfg.base_risk * mult,
                spx_ret=mkt["spx_end"] / mkt["spx_start"] - 1)


def enrich_positions(op: pd.DataFrame, scored: pd.DataFrame, inputs: pd.DataFrame, nav: float,
                     reg: dict, cfg: Config) -> pd.DataFrame:
    p = op.merge(scored[["ticker", "name", "sector", "country", "quant_score", "rating",
                         "g_value", "g_growth", "g_profitability", "g_momentum", "g_sentiment"]], on="ticker", how="left")
    p = p.merge(inputs[["ticker", "beta", "atr14", "dma50"]], on="ticker", how="left")
    p["weight"] = p.market_value / nav
    p["trend_gate"] = np.where(p.price > p.dma50, "Above 50-DMA", "Below 50-DMA")
    p["stop_2atr"] = p.price - cfg.atr_mult * p.atr14
    p["risk_cap"] = np.minimum(nav * cfg.practical_cap,
                               (nav * reg["risk_budget"]) / (cfg.atr_mult * p.atr14) * p.price)
    p["over_cap"] = p.market_value > p.risk_cap + 1
    def action(r):
        if r.rating == "Strong Buy": return "Core Hold / Add"
        if r.rating == "Buy": return "Hold / Accumulate"
        if r.rating == "Hold": return "Hold (Defensive)" if r.trend_gate == "Above 50-DMA" else "Monitor / Trim"
        if r.rating == "Sell": return "Trim / Rotate"
        return "Priority Exit"
    p["action"] = p.apply(action, axis=1)
    return p.sort_values("market_value", ascending=False).reset_index(drop=True)


def dietz(led: pd.DataFrame, nav: float, cfg: Config) -> dict:
    """Modified Dietz return on time-weighted deployed capital (start capital = 0, flows = trades)."""
    span = (cfg.valuation - cfg.inception).days
    w = led["date"].map(lambda d: (cfg.valuation - d).days / span)
    avg_cap = float((w * -led["net_cash"]).sum())
    gain = nav - cfg.capital
    return dict(avg_deployed=avg_cap, gain=gain, ret=gain / avg_cap if avg_cap else np.nan)


# ---------------------------------------------------------------- governance
def governance(led, book, nav, cash, fees, pos, reg, cfg: Config) -> pd.DataFrame:
    tc = trade_compliance(led, cfg)
    post = tc[tc.daily_limit_check != "Week 1 Exempt"]
    per_day = post.groupby("date").size()
    over_days = per_day[per_day > cfg.max_daily_trades]
    cash_pct = cash / cfg.capital
    maxw = pos.weight.max()
    beta = float((pos.market_value * pos.beta).sum() / nav)
    if reg["defensive"]:
        beta_status = "DEFENSIVE GEAR" if beta <= cfg.beta_def_ceiling else "TRIM BETA"
    else:
        beta_status = "RISK-ON GEAR" if beta <= cfg.beta_risk_on else "TRIM BETA"
    rows = [
        ("Max single-stock concentration", f"{maxw:.2%} ({pos.loc[pos.weight.idxmax(), 'ticker']})",
         f"<= {cfg.max_position:.0%}", "PASS" if maxw <= cfg.max_position else "BREACH"),
        ("Internal practical cap", f"{maxw:.2%}", f"<= {cfg.practical_cap:.1%}",
         "PASS" if maxw <= cfg.practical_cap else "BREACH"),
        ("Uninvested cash ceiling (% of initial)", f"{cash_pct:.2%}", f"<= {cfg.max_cash_pct:.0%}",
         "PASS" if cash_pct <= cfg.max_cash_pct else "BREACH"),
        ("Open equity count", str(len(pos)), f"<= {cfg.max_positions}", "PASS" if len(pos) <= cfg.max_positions else "BREACH"),
        ("Monthly minimum activity", str(len(led)), f">= {cfg.min_monthly_trades}", "PASS" if len(led) >= cfg.min_monthly_trades else "BREACH"),
        ("Post-Week-1 daily trade cap", f"max {int(per_day.max()) if len(per_day) else 0}/day", f"<= {cfg.max_daily_trades}/day",
         "PASS" if over_days.empty else "BREACH (%s)" % ", ".join(f"{d}: {n}" for d, n in over_days.items())),
        ("Two-gear beta governance", f"{beta:.2f}", f"{cfg.beta_risk_on} risk-on / <= {cfg.beta_def_ceiling} def.", beta_status),
        ("Positions within regime risk cap", f"{int(pos.over_cap.sum())} over", "0", "PASS" if not pos.over_cap.any() else "REVIEW"),
    ]
    return pd.DataFrame(rows, columns=["Check", "Observed", "Standard", "Status"])


# ---------------------------------------------------------------- rotation
def rotation_plan(pos, cands, nav, reg, cfg: Config, max_swaps=4):
    """Pair held Strong Sells with the best Above-50-DMA eligible Strong Buys not already held."""
    sells = pos[pos.rating == "Strong Sell"].sort_values("quant_score").head(max_swaps)
    pool = cands[(cands.trend_gate == "Above 50-DMA") & (~cands.ticker.isin(pos.ticker))].sort_values("quant_score", ascending=False)
    rows = []
    for (_, s), (_, c) in zip(sells.iterrows(), pool.iterrows()):
        cap = min(nav * cfg.practical_cap,
                  (nav * reg["risk_budget"]) / (cfg.atr_mult * c.atr14) * c.price)
        rows.append(dict(exit=s.ticker, exit_mv=s.market_value, exit_score=s.quant_score, exit_beta=s.beta,
                         buy=c.ticker, buy_sector=c.sector, buy_score=c.quant_score, buy_beta=c.beta,
                         alloc=min(s.market_value, cap)))
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan, {}
    mv = pos.market_value.sum()
    new_score = ((pos.market_value * pos.quant_score).sum() - (plan.exit_mv * plan.exit_score).sum()
                 + (plan.alloc * plan.buy_score).sum()) / (mv - plan.exit_mv.sum() + plan.alloc.sum())
    new_beta = ((pos.market_value * pos.beta).sum() - (plan.exit_mv * plan.exit_beta).sum()
                + (plan.alloc * plan.buy_beta).sum()) / nav
    fin = (pos.loc[pos.sector == "Financial Services", "market_value"].sum()
           + plan.loc[plan.buy_sector == "Financial Services", "alloc"].sum()) / nav
    band = ("BELOW BAND" if new_beta < cfg.beta_def_floor else "IN BAND" if new_beta <= cfg.beta_def_ceiling else "ABOVE CEILING")
    return plan, dict(score=new_score, beta=new_beta, band=band, financials=fin)
