"""BIBF TradeQuest US Quantamental & Risk Model — Streamlit front end.

Run:  streamlit run app.py
"""
import io
from dataclasses import replace
from datetime import date

import pandas as pd
import streamlit as st

from bibf.config import Config
from bibf.data import DATA, fetch_live, get_sp500, load_snapshot, load_universe_list
from bibf.pipeline import run_model
from bibf import portfolio as pf

st.set_page_config(page_title="BIBF Quantamental Model", layout="wide")
st.title("BIBF TradeQuest — US Quantamental & Risk Model")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Live (yfinance)", "Snapshot (Nov 20, 2026 close)"],
                    help="Snapshot reproduces the Excel model offline. Live pulls current Yahoo Finance data.")
    ledger_file = st.file_uploader("TradeQuest ledger CSV (optional)", type="csv",
                                   help="Columns: ticker, order_type, quantity (sells negative), price, net_cash, timestamp")
    universe_choice = st.radio("Universe", ["S&P 500 (+ your holdings)", "Workbook list (173)", "Both"], index=0,
                               help="S&P 500 is fetched live from Wikipedia. Holdings in your ledger are always added.") if mode.startswith("Live") else ""
    with_ebitda = st.checkbox("Fetch EBITDA growth (slower)", value=True) if mode.startswith("Live") else False
    use_cache = st.checkbox("Use 6-hour disk cache", value=True) if mode.startswith("Live") else True
    extra = st.text_input("Extra tickers to score (comma-separated)", "") if mode.startswith("Live") else ""
    refresh = st.button("Run / refresh", type="primary")

    st.header("Assumptions")
    base = Config()
    with st.expander("Factor weights & circuit-breaker"):
        wv = st.number_input("Value", 0.0, 10.0, base.w_value, 0.5)
        wg = st.number_input("Growth", 0.0, 10.0, base.w_growth, 0.5)
        wp = st.number_input("Profitability", 0.0, 10.0, base.w_profitability, 0.5)
        wm = st.number_input("Momentum", 0.0, 10.0, base.w_momentum, 0.5)
        ws = st.number_input("Sentiment", 0.0, 10.0, base.w_sentiment, 0.5)
        gms = st.slider("Min pct: growth/momentum/sentiment", 0.0, 0.5, base.min_pct_gms, 0.01)
        vp = st.slider("Min pct: value/profitability", 0.0, 0.5, base.min_pct_vp, 0.01)
    with st.expander("TradeQuest rules"):
        capital = st.number_input("Starting capital ($)", 100_000.0, 50_000_000.0, float(base.capital), 100_000.0)
        max_daily = st.number_input("Max trades/day after Week 1", 1, 50, base.max_daily_trades)
        fee = st.number_input("Fee per trade ($)", 0.0, 50.0, base.fee_per_trade, 0.5)
        w1s = st.date_input("Week 1 start", base.week1_start)
        w1e = st.date_input("Week 1 end", base.week1_end)
        incep = st.date_input("Inception date", base.inception)
    with st.expander("Risk overlay & eligibility"):
        prac = st.number_input("Practical single-stock cap", 0.01, 0.5, base.practical_cap, 0.005, format="%.3f")
        risk = st.number_input("Base risk per trade", 0.001, 0.05, base.base_risk, 0.001, format="%.3f")
        mcap = st.number_input("Min market cap for eligible pick ($B)", 0.0, 500.0, base.min_mkt_cap_b, 1.0)

cfg = replace(base, w_value=wv, w_growth=wg, w_profitability=wp, w_momentum=wm, w_sentiment=ws,
              min_pct_gms=gms, min_pct_vp=vp, capital=capital, max_daily_trades=int(max_daily),
              fee_per_trade=fee, week1_start=w1s, week1_end=w1e, inception=incep, practical_cap=prac,
              base_risk=risk, min_mkt_cap_b=mcap)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def _snapshot():
    return load_snapshot()


def get_ledger():
    return ledger_file if ledger_file is not None else DATA / "ledger.csv"


def get_data():
    if mode.startswith("Snapshot"):
        uni, inputs, mkt = _snapshot()
        return uni, inputs, mkt, [], cfg
    led = pf.load_ledger(get_ledger())
    wb_list = load_universe_list()
    sp = None
    if not universe_choice.startswith("Workbook"):
        try:
            sp = get_sp500()
        except Exception as e:
            st.warning(f"Could not load the S&P 500 list ({e}); falling back to the workbook universe.")
    meta = pd.concat([sp, wb_list]) if sp is not None else wb_list          # S&P sector labels take priority
    held = set(led.ticker)
    if sp is None or universe_choice.startswith("Workbook"):
        tickers = set(wb_list.ticker) | held
    elif universe_choice.startswith("S&P"):
        tickers = set(sp.ticker) | held
    else:
        tickers = set(sp.ticker) | set(wb_list.ticker) | held
    tickers |= {t.strip().upper() for t in extra.split(",") if t.strip()}
    bar = st.progress(0.0, text="Fetching from Yahoo Finance…")
    uni, inputs, mkt, issues = fetch_live(sorted(tickers), asof=pd.Timestamp.today().normalize(),
                                          inception=pd.Timestamp(cfg.inception), with_ebitda_growth=with_ebitda,
                                          use_cache=use_cache, meta=meta,
                                          progress=lambda p, t: bar.progress(min(max(p, 0.0), 1.0), text=f"{t}"))
    bar.empty()
    live_cfg = replace(cfg, valuation=pd.Timestamp(mkt["last_date"]).date())
    return uni, inputs, mkt, issues, live_cfg


if "result" not in st.session_state or refresh:
    try:
        uni, inputs, mkt, issues, run_cfg = get_data()
        st.session_state["result"] = run_model(uni, inputs, mkt, get_ledger(), run_cfg)
        st.session_state["issues"] = issues
        st.session_state["cfg"] = run_cfg
    except Exception as e:  # surface data problems instead of a blank page
        st.error(f"Model run failed: {e}")
        st.stop()

R = st.session_state["result"]; S = R["summary"]; issues = st.session_state.get("issues", [])
if issues:
    with st.expander(f"{len(issues)} data warnings"):
        st.write("\n".join(f"- {i}" for i in issues))
st.caption(f"Source: {mode} · S&P 500 {R['market']['spx_end']:,.0f} · VIX {R['market']['vix_end']:.2f} · "
           f"regime multiplier {S['regime']['multiplier']}x ({'Defensive' if S['regime']['defensive'] else 'Risk-On'} gear)")

tab_sum, tab_pos, tab_score, tab_cand, tab_led, tab_exp = st.tabs(
    ["Executive summary", "Portfolio", "Scoring engine", "Candidates & rotation", "Ledger & compliance", "Export"])

# ------------------------------------------------------------------ summary
with tab_sum:
    c = st.columns(4)
    c[0].metric("NAV (net of fees)", f"${S['nav']:,.0f}", f"{S['ret']:.2%}")
    c[1].metric("Cash", f"${S['cash']:,.0f}", f"{S['cash_pct']:.2%} of capital")
    c[2].metric("Portfolio beta", f"{S['beta']:.2f}")
    c[3].metric("Value-wtd Quant Score", f"{S['port_score']:.2f}")
    c = st.columns(4)
    c[0].metric("S&P 500 return", f"{S['bench']:.2%}")
    c[1].metric("vs S&P (simple, full capital)", f"{S['rel_bps']:+.0f} bps")
    c[2].metric("Return on deployed capital", f"{S['dietz_ret']:.2%}", f"{S['dietz_rel_bps']:+.0f} bps vs S&P")
    c[3].metric("Avg capital deployed", f"${S['avg_deployed']:,.0f}")
    st.subheader("Governance & compliance")
    st.dataframe(R["governance"], hide_index=True, use_container_width=True)
    sec = R["positions"].groupby("sector").agg(stocks=("ticker", "count"), cost=("cost_basis", "sum"),
                                                mv=("market_value", "sum"), score=("quant_score", "mean"))
    sec["unreal_%"] = sec.mv / sec.cost - 1; sec["% NAV"] = sec.mv / S["nav"]
    st.subheader("Sector allocation")
    st.dataframe(sec.sort_values("mv", ascending=False).style.format(
        {"cost": "${:,.0f}", "mv": "${:,.0f}", "score": "{:.2f}", "unreal_%": "{:.1%}", "% NAV": "{:.1%}"}),
        use_container_width=True)
    st.bar_chart(sec["% NAV"])

# ------------------------------------------------------------------ portfolio
with tab_pos:
    st.subheader("Open positions")
    cols = ["ticker", "name", "sector", "shares", "avg_cost", "price", "market_value", "unreal_pl", "unreal_ret",
            "weight", "beta", "atr14", "dma50", "trend_gate", "stop_2atr", "risk_cap", "over_cap",
            "g_value", "g_growth", "g_profitability", "g_momentum", "g_sentiment", "quant_score", "rating", "action"]
    st.dataframe(R["positions"][cols].style.format(
        {"avg_cost": "{:,.2f}", "price": "{:,.2f}", "market_value": "{:,.0f}", "unreal_pl": "{:,.0f}",
         "unreal_ret": "{:.1%}", "weight": "{:.2%}", "beta": "{:.2f}", "atr14": "{:.2f}", "dma50": "{:,.2f}",
         "stop_2atr": "{:,.2f}", "risk_cap": "{:,.0f}", "quant_score": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.subheader("Closed round trips")
    st.dataframe(R["closed"], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ scoring
with tab_score:
    d = R["scored"]
    f = st.columns(3)
    sectors = f[0].multiselect("Sector", sorted(d.sector.unique()))
    ratings = f[1].multiselect("Rating", ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"])
    q = f[2].text_input("Ticker contains")
    v = d.copy()
    if sectors: v = v[v.sector.isin(sectors)]
    if ratings: v = v[v.rating.isin(ratings)]
    if q: v = v[v.ticker.str.contains(q.upper())]
    show = ["ticker", "name", "sector", "price", "mkt_cap_b", "g_value", "g_growth", "g_profitability", "g_momentum",
            "g_sentiment", "disqualified", "raw_score", "quant_score", "rating", "eligible"]
    st.dataframe(v.sort_values("quant_score", ascending=False)[show], hide_index=True, use_container_width=True)
    st.caption(f"{len(v)} of {len(d)} stocks. Non-common instruments (preferreds, depositary shares, units) are excluded automatically.")
    st.bar_chart(d.rating.value_counts().reindex(["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]))

# ------------------------------------------------------------------ candidates
with tab_cand:
    st.subheader("Eligible Strong Buy candidates")
    ccols = ["ticker", "name", "sector", "price", "quant_score", "beta", "atr14", "dma50", "trend_gate",
             "stop_2atr", "regime_target", "signal"]
    st.dataframe(R["candidates"][ccols], hide_index=True, use_container_width=True)
    st.subheader("Gear 1 rotation: exit held Strong Sells → best Above-50-DMA candidates")
    if R["rotation"].empty:
        st.info("No Strong Sell positions or no trend-confirmed candidates.")
    else:
        st.dataframe(R["rotation"], hide_index=True, use_container_width=True)
        p = R["post_rotation"]; c = st.columns(4)
        c[0].metric("Score after rotation", f"{p['score']:.2f}")
        c[1].metric("Beta after rotation", f"{p['beta']:.2f}", p["band"], delta_color="off")
        c[2].metric("Financial Services % NAV", f"{p['financials']:.1%}")
        c[3].metric("Cash reinvested", f"${R['rotation'].alloc.sum():,.0f}")

# ------------------------------------------------------------------ ledger
with tab_led:
    st.subheader("Ledger with daily-limit check")
    st.dataframe(R["ledger"].drop(columns=["ts"]), hide_index=True, use_container_width=True)
    st.subheader("End-of-day cash vs ceiling")
    cp = R["cash_path"]
    st.dataframe(cp.style.format({"cash": "${:,.0f}", "cash_pct_initial": "{:.2%}"}), use_container_width=True)
    st.line_chart(cp["cash_pct_initial"])

# ------------------------------------------------------------------ export
with tab_exp:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([{k: v for k, v in S.items() if k != "regime"}]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="Summary")
        R["governance"].to_excel(xw, sheet_name="Governance", index=False)
        R["positions"].to_excel(xw, sheet_name="Positions", index=False)
        R["closed"].to_excel(xw, sheet_name="Closed", index=False)
        R["scored"].to_excel(xw, sheet_name="Scoring", index=False)
        R["candidates"].to_excel(xw, sheet_name="Candidates", index=False)
        R["rotation"].to_excel(xw, sheet_name="Rotation", index=False)
        R["ledger"].drop(columns=["ts"]).to_excel(xw, sheet_name="Ledger", index=False)
    st.download_button("Download results (.xlsx)", buf.getvalue(), file_name=f"bibf_model_{date.today()}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download scoring table (.csv)", R["scored"].to_csv(index=False), file_name="bibf_scoring.csv")
    st.json(st.session_state["cfg"].as_dict(), expanded=False)
