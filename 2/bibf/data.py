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


def _atr14(h, l, c):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.tail(14).mean())


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
def fetch_live(tickers, asof=None, inception=None, with_ebitda_growth=True, max_workers=8,
               use_cache=True, cache_hours=6, progress=None, meta=None):
    """Pull prices/fundamentals from Yahoo Finance for `tickers`.
    Returns (universe_df, market_inputs_df, market_dict, issues)."""
    import yfinance as yf
    tickers = sorted(set(tickers))
    asof = pd.Timestamp(asof or pd.Timestamp.today().normalize())
    digest = hashlib.md5(",".join(tickers).encode()).hexdigest()[:10]   # stable across restarts
    key = CACHE / f"live_{asof.date()}_{len(tickers)}_{digest}_{int(with_ebitda_growth)}.pkl"
    if use_cache and key.exists() and time.time() - key.stat().st_mtime < cache_hours * 3600:
        return pickle.load(open(key, "rb"))

    issues = []
    start = asof - pd.DateOffset(months=15)
    frames = []   # download in chunks so large universes (S&P 500) don't time out
    allt = tickers + ["^GSPC", "^VIX"]
    for i in range(0, len(allt), 100):
        frames.append(yf.download(allt[i:i + 100], start=start.date(), end=(asof + pd.Timedelta(days=1)).date(),
                                  auto_adjust=True, group_by="ticker", threads=True, progress=False))
        if progress: progress(0.0, f"prices {min(i + 100, len(allt))}/{len(allt)}")
    px = pd.concat(frames, axis=1)
    spx = px["^GSPC"]["Close"].dropna(); vix = px["^VIX"]["Close"].dropna()
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
                h = px[t].dropna(subset=["Close"])
                c = h["Close"]
                r["price"] = float(c.iloc[-1])
                r["ret_3m"] = r["price"] / _asof_close(c, c.index[-1], 3) - 1
                r["ret_6m"] = r["price"] / _asof_close(c, c.index[-1], 6) - 1
                r["ret_12m"] = r["price"] / _asof_close(c, c.index[-1], 12) - 1
                j = pd.concat([c.pct_change(), spx_ret], axis=1, join="inner").dropna().tail(252)
                beta = j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var()
                mrows.append(dict(ticker=t, beta=float(beta), atr14=_atr14(h["High"], h["Low"], c),
                                  dma50=float(c.tail(50).mean())))
            except Exception as e:
                issues.append(f"{t}: price history failed ({e})")
            rows.append(r)
            done += 1
            if progress: progress(done / len(tickers), t)

    uni = pd.DataFrame(rows)
    base = (meta if meta is not None else load_universe_list()).drop_duplicates("ticker").set_index("ticker")
    uni["sector"] = uni.ticker.map(base["sector"]).fillna(uni.get("sector_live").map(lambda x: SECTOR_MAP.get(x, x)))
    uni["country"] = uni.ticker.map(base["country"]).fillna("US")
    uni["role"] = "Live"
    for col in METRIC_COLS + ["mkt_cap_b", "price"]:
        if col not in uni: uni[col] = np.nan
    missing = uni[uni.sector.isna() | uni.price.isna()].ticker.tolist()
    if missing: issues.append("No sector/price (dropped): " + ", ".join(missing))
    uni = uni.dropna(subset=["sector", "price"]).reset_index(drop=True)

    inception = pd.Timestamp(inception) if inception is not None else asof
    mkt = dict(spx_start=_asof_close(spx, inception), spx_end=float(spx.iloc[-1]),
               spx_50dma=float(spx.tail(50).mean()), vix_start=_asof_close(vix, inception), vix_end=float(vix.iloc[-1]),
               last_date=str(spx.index[-1].date()))
    res = (uni, pd.DataFrame(mrows), mkt, issues)
    CACHE.mkdir(exist_ok=True)
    pickle.dump(res, open(key, "wb"))
    return res
