"""Portfolio audit, TradeQuest compliance, regime/risk overlay, Modified Dietz and rotation logic."""
import numpy as np
import pandas as pd
from .config import Config


# ---------------------------------------------------------------- ledger
LEDGER_COLS = ["ticker", "order_type", "label", "quantity", "price", "net_cash", "timestamp"]
TS_FMT = "%m/%d/%Y - %H:%M"
_ALIASES = {
    "ticker": ["ticker", "symbol", "stock", "instrument", "security", "code"],
    "order_type": ["order_type", "order type", "type", "side", "action", "buy/sell", "transaction type"],
    "label": ["label", "csv log label", "name", "company", "description", "security name"],
    "quantity": ["quantity", "qty", "shares", "units", "filled quantity", "amount of shares"],
    "price": ["price", "exec. price", "exec price", "execution price", "executed price", "fill price", "avg price"],
    "net_cash": ["net_cash", "net cash impact", "net cash", "cash impact", "net amount", "total", "value", "amount"],
    "timestamp": ["timestamp", "time", "datetime", "date/time", "date time", "executed at", "execution time"],
    "date": ["trade date", "date"],
}


def _to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True)
                         .str.replace(r"^\((.*)\)$", r"-\1", regex=True), errors="coerce")


def normalize_ledger(src) -> pd.DataFrame:
    """Accept a DataFrame / path / file-like in the model's format OR a typical broker export and return the standard
    ledger: ticker, order_type, label, quantity (sells negative), price, net_cash, timestamp ('MM/DD/YYYY - HH:MM').

    Handles: alias column names; buy/sell given by an order-type/side column (quantity may be unsigned);
    missing net_cash (derived as -quantity*price); separate date and time columns; $ and thousands separators."""
    d = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    d.columns = [str(c).strip().lower() for c in d.columns]
    ren, used = {}, set()
    for std, names in _ALIASES.items():
        for n in names:
            if n in d.columns and n not in used and std not in ren.values():
                ren[n] = std; used.add(n); break
    d = d.rename(columns=ren)
    for need in ("ticker", "quantity", "price"):
        if need not in d.columns:
            raise ValueError(f"Ledger is missing a '{need}' column (found: {list(d.columns)})")
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    d["quantity"] = _to_num(d["quantity"]); d["price"] = _to_num(d["price"])
    if "order_type" not in d.columns: d["order_type"] = ""
    d["order_type"] = d["order_type"].fillna("").astype(str)
    sell = d.order_type.str.contains("sell|short|close|sld", case=False, regex=True)
    buy = d.order_type.str.contains("buy|bought|open|long", case=False, regex=True)
    d.loc[sell, "quantity"] = -d.loc[sell, "quantity"].abs()
    d.loc[buy & ~sell, "quantity"] = d.loc[buy & ~sell, "quantity"].abs()
    if "timestamp" in d.columns and "date" in d.columns and not d["timestamp"].astype(str).str.contains(r"\d{1,2}/\d{1,2}|-").all():
        d["timestamp"] = d["date"].astype(str) + " " + d["timestamp"].astype(str)
    if "timestamp" not in d.columns:
        if "date" not in d.columns: raise ValueError("Ledger needs a timestamp (or date) column")
        d["timestamp"] = d["date"]
    ts = pd.to_datetime(d["timestamp"].astype(str).str.replace(" - ", " ", regex=False), errors="coerce")
    if ts.isna().any():
        raise ValueError("Unparseable timestamps in rows: %s" % list(d.index[ts.isna()] + 1))
    d["timestamp"] = ts.dt.strftime(TS_FMT)
    nc = _to_num(d["net_cash"]) if "net_cash" in d.columns else pd.Series(np.nan, index=d.index)
    d["net_cash"] = nc.fillna(-d["quantity"] * d["price"])
    if "label" not in d.columns: d["label"] = d["ticker"]
    d["label"] = d["label"].fillna(d["ticker"])
    bad = d[["ticker", "quantity", "price"]].isna().any(axis=1) | (d["quantity"] == 0)
    if bad.any():
        raise ValueError("Rows with missing/zero ticker, quantity or price: %s" % list(d.index[bad] + 1))
    d["order_type"] = np.where(d.order_type.str.strip() == "", np.where(d.quantity > 0, "Market - Buy", "Market - Sell"), d.order_type)
    return d[LEDGER_COLS].reset_index(drop=True)


def validate_ledger(led: pd.DataFrame) -> list:
    """Return a list of problems (e.g. selling more shares than are held at that time)."""
    d = led.copy()
    d["ts"] = pd.to_datetime(d["timestamp"].astype(str).str.replace(" - ", " ", regex=False))
    d = d.sort_values(["ts"], kind="stable")
    issues, held = [], {}
    for _, r in d.iterrows():
        held[r.ticker] = held.get(r.ticker, 0) + r.quantity
        if held[r.ticker] < -1e-9:
            issues.append(f"{r.ticker}: sells {abs(r.quantity):g} shares at {r.timestamp} but only {held[r.ticker] + abs(r.quantity):g} held")
    return issues


# ---------------------------------------------------------------- open-positions import
def is_positions_csv(df: pd.DataFrame) -> bool:
    cols = {str(c).strip().lower() for c in df.columns}
    return {"symbol", "quantity", "pricepaid"} <= cols and not ({"timestamp", "datetime", "date", "time"} & cols)


def read_positions(src) -> pd.DataFrame:
    """Broker 'Open Positions' export (Symbol, Description, Quantity, LastPrice, PricePaid, MarketValue, ...)."""
    d = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    d.columns = [str(c).strip().lower() for c in d.columns]
    d = d[d["symbol"].notna()].copy()
    out = pd.DataFrame({
        "ticker": d["symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False),
        "name": d.get("description", d["symbol"]),
        "quantity": _to_num(d["quantity"]), "price_paid": _to_num(d["pricepaid"]),
        "last_price": _to_num(d["lastprice"]) if "lastprice" in d else np.nan})
    if out[["quantity", "price_paid"]].isna().any().any() or (out.quantity <= 0).any():
        raise ValueError("Open-positions file has missing/non-positive quantity or PricePaid values")
    return out.reset_index(drop=True)


def positions_to_ledger(pos: pd.DataFrame, ts: str) -> pd.DataFrame:
    """Rebuild a ledger from positions only: one buy per position at PricePaid (history/dates/closed trades are lost)."""
    return pd.DataFrame({"ticker": pos.ticker, "order_type": "Imported - Buy", "label": pos.name,
                         "quantity": pos.quantity, "price": pos.price_paid,
                         "net_cash": -pos.quantity * pos.price_paid, "timestamp": ts})[LEDGER_COLS]


def sync_trades(ledger: pd.DataFrame, pos: pd.DataFrame, ts: str, default_prices: dict = None) -> pd.DataFrame:
    """Trades needed to turn the current ledger into the uploaded open positions.
    New ticker -> buy at PricePaid. Closed ticker / reduced quantity -> sell at `default_prices` (editable estimate;
    the positions file contains no exit price). Added shares on an existing ticker are priced so the new average
    cost equals PricePaid. Returns columns: ticker, side, quantity, price, note."""
    default_prices = default_prices or {}
    led = ledger.copy()
    held = led.groupby("ticker")["quantity"].sum()
    cost = led[led.quantity > 0].groupby("ticker").apply(lambda g: -g["net_cash"].sum(), include_groups=False)
    buys = led[led.quantity > 0].groupby("ticker")["quantity"].sum()
    rows = []
    target = pos.set_index("ticker")
    for t in sorted(set(held[held.abs() > 1e-9].index) | set(target.index)):
        cur = float(held.get(t, 0.0)); tgt = float(target.quantity.get(t, 0.0)); delta = tgt - cur
        if abs(delta) < 1e-9:
            continue
        if delta > 0:
            paid = float(target.price_paid[t])
            if cur > 0:   # price the extra shares so the blended average equals PricePaid
                avg_cur = float(cost[t] / buys[t])
                px = (tgt * paid - cur * avg_cur) / delta
                px = px if px > 0 else paid
                note = "Added shares (price solved so average cost = PricePaid)"
            else:
                px, note = paid, "New position (price = PricePaid)"
            rows.append(dict(ticker=t, side="Buy", quantity=delta, price=round(px, 4), note=note))
        else:
            px = float(default_prices.get(t, 0.0))
            rows.append(dict(ticker=t, side="Sell", quantity=-delta, price=round(px, 4),
                             note=("Position closed" if tgt == 0 else "Position reduced") +
                                  " — price is an ESTIMATE, edit to the actual fill" if px else
                                  ("Position closed" if tgt == 0 else "Position reduced") + " — ENTER the actual sell price"))
    return pd.DataFrame(rows, columns=["ticker", "side", "quantity", "price", "note"])


def cost_mismatches(ledger: pd.DataFrame, pos: pd.DataFrame, tol: float = 0.005) -> list:
    """Tickers whose quantity is unchanged but whose average cost differs from the file's PricePaid."""
    buys = ledger[ledger.quantity > 0].groupby("ticker").agg(q=("quantity", "sum"), c=("net_cash", lambda x: -x.sum()))
    held = ledger.groupby("ticker")["quantity"].sum()
    out = []
    for _, r in pos.iterrows():
        if r.ticker in buys.index and abs(held.get(r.ticker, 0) - r.quantity) < 1e-9:
            avg = buys.loc[r.ticker, "c"] / buys.loc[r.ticker, "q"]
            if abs(avg - r.price_paid) / r.price_paid > tol:
                out.append(f"{r.ticker}: ledger avg cost {avg:,.2f} vs file PricePaid {r.price_paid:,.2f}")
    return out


def trades_to_rows(trades: pd.DataFrame, ts: str, names: dict = None) -> pd.DataFrame:
    names = names or {}
    q = np.where(trades.side == "Buy", trades.quantity, -trades.quantity).astype(float)
    return pd.DataFrame({"ticker": trades.ticker, "order_type": "Synced - " + trades.side, "label": trades.ticker.map(lambda t: names.get(t, t)),
                         "quantity": q, "price": trades.price, "net_cash": -q * trades.price, "timestamp": ts})[LEDGER_COLS]


def load_ledger(src, asof=None) -> pd.DataFrame:
    """Normalise, validate and parse the ledger; optionally keep only trades on/before `asof` (date)."""
    d = normalize_ledger(src)
    issues = validate_ledger(d)
    if issues: raise ValueError("Invalid ledger: " + "; ".join(issues))
    d["ts"] = pd.to_datetime(d["timestamp"].str.replace(" - ", " ", regex=False))
    d["date"] = d["ts"].dt.date
    d = d.sort_values("ts", kind="stable").reset_index(drop=True)
    if asof is not None:
        d = d[d["date"] <= pd.Timestamp(asof).date()].reset_index(drop=True)
    return d


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
    op = pd.DataFrame(rows_o, columns=["ticker", "shares", "cost_basis", "avg_cost"])
    cl = pd.DataFrame(rows_c, columns=["ticker", "shares", "buy_cost", "sell_proceeds", "realized_pl", "realized_ret"])
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
    w = led["date"].map(lambda d: max((cfg.valuation - d).days, 0) / span)   # trades after valuation get 0 weight
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
