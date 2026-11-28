"""Data layer: frozen Nov-20-2026 snapshot (offline) and live yfinance extraction."""
import hashlib, io, json, pathlib, pickle, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CACHE = pathlib.Path(__file__).resolve().parent.parent / ".cache"
METRIC_COLS = ["pe", "ev_ebitda", "ps", "pb", "fcf_yield", "rev_g", "ebitda_g", "eps_g", "gross_m", "oper_m",
               "net_m", "roe", "fcf_m", "ret_3m", "ret_6m", "ret_12m", "bullish"]


# ------------------------------------------------------------------ snapshot
def load_snapshot():
    uni = pd.read_csv(DATA / "snapshot_universe.csv")
    inputs = pd.read_csv(DATA / "snapshot_market_inputs.csv")
    mkt = json.load(open(DATA / "snapshot_market.json"))
    return uni, inputs, mkt


def load_universe_list() -> pd.DataFrame:
    return pd.read_csv(DATA / "universe.csv")


# ------------------------------------------------------------------ S&P 500 list
SECTOR_MAP = {  # Wikipedia GICS names -> sector names used by the model (Yahoo-style, as in the workbook)
    "Information Technology": "Technology", "Health Care": "Healthcare", "Financials": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical", "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials", "Communication Services": "Communication Services",
    "Industrials": "Industrials", "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate",
}


def parse_sp500_table(t: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"ticker": t["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
                        "name": t["Security"], "sector": t["GICS Sector"].map(SECTOR_MAP), "country": "US"})
    return out.dropna(subset=["sector"]).drop_duplicates("ticker").reset_index(drop=True)


def get_sp500() -> pd.DataFrame:
    """Current S&P 500 constituents from Wikipedia (needs internet + lxml). Cached for 24h on disk."""
    import requests
    f = CACHE / "sp500.csv"
    if f.exists() and time.time() - f.stat().st_mtime < 24 * 3600:
        return pd.read_csv(f)
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0 (BIBF-model)"}, timeout=30)
    r.raise_for_status()
    out = parse_sp500_table(pd.read_html(io.StringIO(r.text))[0])
    if len(out) < 450:
        raise ValueError(f"S&P 500 table looks incomplete ({len(out)} rows)")
    CACHE.mkdir(exist_ok=True); out.to_csv(f, index=False)
    return out


# ------------------------------------------------------------------ helpers
def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _asof_close(s: pd.Series, date, months=0):
    d = pd.Timestamp(date) - pd.DateOffset(months=months)
    s = s[s.index <= d].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _fetch_fundamentals(t: str, with_ebitda_growth: bool):
    import yfinance as yf
    tk = yf.Ticker(t)
    try:
        info = tk.get_info()
    except Exception:
        info = {}
    mc = _num(info.get("marketCap")); rev = _num(info.get("totalRevenue")); fcf = _num(info.get("freeCashflow"))
    out = dict(
        ticker=t, name=info.get("shortName") or info.get("longName") or t, sector_live=info.get("sector"),
        mkt_cap_b=mc / 1e9 if mc == mc else np.nan,
        pe=_num(info.get("trailingPE")), ev_ebitda=_num(info.get("enterpriseToEbitda")),
        ps=_num(info.get("priceToSalesTrailing12Months")), pb=_num(info.get("priceToBook")),
        fcf_yield=fcf / mc if (fcf == fcf and mc == mc and mc) else np.nan,
        rev_g=_num(info.get("revenueGrowth")), eps_g=_num(info.get("earningsGrowth")),
        gross_m=_num(info.get("grossMargins")), oper_m=_num(info.get("operatingMargins")),
        net_m=_num(info.get("profitMargins")), roe=_num(info.get("returnOnEquity")),
        fcf_m=fcf / rev if (fcf == fcf and rev == rev and rev) else np.nan, ebitda_g=np.nan, bullish=np.nan)
    try:  # analyst sentiment: share of Strong Buy + Buy ratings
        rs = tk.recommendations_summary
        if rs is not None and len(rs):
            r = rs.iloc[0]
            tot = float(r[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum())
            if tot > 0:
                out["bullish"] = float(r["strongBuy"] + r["buy"]) / tot
    except Exception:
        pass
    if with_ebitda_growth:
        try:  # YoY change in latest quarterly EBITDA
            q = tk.quarterly_income_stmt
            row = q.loc["EBITDA"] if "EBITDA" in q.index else q.loc["Normalized EBITDA"]
            row = row.dropna()
            if len(row) >= 5 and row.iloc[4] > 0:
                out["ebitda_g"] = float(row.iloc[0] / row.iloc[4] - 1)
        except Exception:
            pass
    return out


# ------------------------------------------------------------------ live
def _download_chunked(symbols, start, end, progress=None, size=100):
    import yfinance as yf
    frames = []
    for i in range(0, len(symbols), size):
        chunk = symbols[i:i + size]
        f = yf.download(chunk, start=start, end=end, auto_adjust=True, group_by="ticker",
                        threads=True, progress=False)
        if not isinstance(f.columns, pd.MultiIndex):          # single-symbol chunk comes back flat
            f.columns = pd.MultiIndex.from_product([[chunk[0]], f.columns])
        frames.append(f)
        if progress: progress(0.0, f"prices {min(i + size, len(symbols))}/{len(symbols)}")
    return pd.concat(frames, axis=1)


def fetch_live(tickers, asof=None, inception=None, with_ebitda_growth=True, max_workers=8,
               use_cache=True, cache_hours=6, progress=None, meta=None, reprice=True):
    """Pull prices/fundamentals from Yahoo Finance for `tickers`, evaluated as of `asof`.

    Price-based items (price, 3/6/12M returns, beta, ATR, moving averages, S&P/VIX) are truly historical.
    Yahoo only serves CURRENT fundamentals/analyst ratings, so for a past `asof` those are today's values;
    with reprice=True the valuation multiples and market cap are rescaled by price(asof)/price(today).
    Both SMA and EMA versions of the 50-day average and ATR are returned so the UI can switch without refetching.
    Returns (universe_df, market_inputs_df, market_dict, issues)."""
    import yfinance as yf
    tickers = sorted(set(tickers))
    today = pd.Timestamp.today().normalize()
    asof = pd.Timestamp(asof or today)
    digest = hashlib.md5((",".join(tickers) + f"|{int(reprice)}").encode()).hexdigest()[:10]   # stable across restarts
    key = CACHE / f"live_{asof.date()}_{len(tickers)}_{digest}_{int(with_ebitda_growth)}.pkl"
    if use_cache and key.exists() and time.time() - key.stat().st_mtime < cache_hours * 3600:
        return pickle.load(open(key, "rb"))

    issues = []
    start = (asof - pd.DateOffset(months=15)).date()
    px = _download_chunked(["^GSPC", "^VIX"] + tickers, start, (max(today, asof) + pd.Timedelta(days=1)).date(), progress)
    spx = px["^GSPC"]["Close"].dropna(); spx = spx[spx.index <= asof]
    vix = px["^VIX"]["Close"].dropna(); vix = vix[vix.index <= asof]
    if spx.empty:
        raise ValueError(f"No S&P 500 data on or before {asof.date()}")
    spx_ret = spx.pct_change()

    rows, mrows = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_fundamentals, t, with_ebitda_growth): t for t in tickers}
        for f in as_completed(futs):
            t = futs[f]
            try:
                r = f.result()
            except Exception as e:
                issues.append(f"{t}: fundamentals failed ({e})"); r = dict(ticker=t)
            try:
                h_full = px[t].dropna(subset=["Close"])
                h = h_full[h_full.index <= asof]
                c = h["Close"]
                price = float(c.iloc[-1])
                r["price"] = price
                for m, k in ((3, "ret_3m"), (6, "ret_6m"), (12, "ret_12m")):
                    r[k] = price / _asof_close(c, c.index[-1], m) - 1
                ratio = price / float(h_full["Close"].iloc[-1])          # price(asof) / price(latest)
                if reprice and abs(ratio - 1) > 1e-9:
                    for k in ("pe", "ev_ebitda", "ps", "pb"):
                        if k in r and r[k] == r[k]: r[k] = r[k] * ratio
                    if r.get("fcf_yield") == r.get("fcf_yield"): r["fcf_yield"] = r["fcf_yield"] / ratio
                    if r.get("mkt_cap_b") == r.get("mkt_cap_b"): r["mkt_cap_b"] = r["mkt_cap_b"] * ratio
                j = pd.concat([c.pct_change(), spx_ret], axis=1, join="inner").dropna().tail(252)
                beta = j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var()
                tr = _true_range(h["High"], h["Low"], c)
                mrows.append(dict(
                    ticker=t, beta=float(beta),
                    atr14_sma=float(tr.tail(14).mean()), atr14_ema=float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]),
                    dma50_sma=float(c.tail(50).mean()), dma50_ema=float(c.ewm(span=50, adjust=False).mean().iloc[-1])))
            except Exception as e:
                issues.append(f"{t}: price history failed ({e})")
            rows.append(r)
            done += 1
            if progress: progress(done / len(tickers), t)

    uni = pd.DataFrame(rows)
    base = (meta if meta is not None else load_universe_list()).drop_duplicates("ticker").set_index("ticker")
    live_sector = uni["sector_live"].map(lambda x: SECTOR_MAP.get(x, x)) if "sector_live" in uni else None
    uni["sector"] = uni.ticker.map(base["sector"])
    if live_sector is not None: uni["sector"] = uni["sector"].fillna(live_sector)
    uni["country"] = uni.ticker.map(base["country"]).fillna("US")
    uni["role"] = "Live"
    for col in METRIC_COLS + ["mkt_cap_b", "price"]:
        if col not in uni: uni[col] = np.nan
    missing = uni[uni.sector.isna() | uni.price.isna()].ticker.tolist()
    if missing: issues.append("No sector/price (dropped): " + ", ".join(missing))
    uni = uni.dropna(subset=["sector", "price"]).reset_index(drop=True)
    if asof < today:
        issues.append(f"As-of {asof.date()}: prices, returns, beta, ATR and moving averages are historical; "
                      "fundamentals and analyst ratings are the latest Yahoo values "
                      + ("(multiples re-priced to the date)." if reprice else "(not re-priced)."))

    inception = pd.Timestamp(inception) if inception is not None else asof
    mkt = dict(spx_start=_asof_close(spx, inception), spx_end=float(spx.iloc[-1]),
               spx_50dma_sma=float(spx.tail(50).mean()), spx_50dma_ema=float(spx.ewm(span=50, adjust=False).mean().iloc[-1]),
               vix_start=_asof_close(vix, inception), vix_end=float(vix.iloc[-1]),
               last_date=str(spx.index[-1].date()))
    res = (uni, pd.DataFrame(mrows), mkt, issues)
    CACHE.mkdir(exist_ok=True)
    pickle.dump(res, open(key, "wb"))
    return res
