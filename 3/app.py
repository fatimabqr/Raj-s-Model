"""BIBF TradeQuest US Quantamental & Risk Model — Streamlit front end.

Run:  streamlit run app.py
"""
import io
import re
from dataclasses import replace
from datetime import date, datetime, time as dtime

import pandas as pd
import streamlit as st

from bibf.config import Config
from bibf.data import DATA, fetch_live, get_sp500, load_snapshot, load_universe_list
from bibf.pipeline import run_model
from bibf import portfolio as pf

st.set_page_config(page_title="BIBF Quantamental Model", layout="wide")
st.title("BIBF TradeQuest — US Quantamental & Risk Model")
ss = st.session_state
TODAY = date.today()

# ------------------------------------------------------------------ state
if "ledger" not in ss:
    ss.ledger = pf.normalize_ledger(DATA / "ledger.csv")
    ss.ledger_version = 0
    ss.flash = None


def set_ledger(new: pd.DataFrame, msg: str):
    issues = pf.validate_ledger(new)
    if issues:
        st.error("Not applied — " + "; ".join(issues)); return False
    ss.ledger = new.reset_index(drop=True); ss.ledger_version += 1; ss.flash = msg
    st.rerun()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Live (yfinance)", "Snapshot (Nov 20, 2026 close)"],
                    help="Snapshot reproduces the Excel model offline. Live pulls Yahoo Finance data.")
    live = mode.startswith("Live")
    if live:
        asof = st.date_input("Scoring as-of date", TODAY, max_value=TODAY,
                             help="Scores, positions and risk are shown as of this trading day. Prices, returns, beta, "
                                  "ATR and moving averages are historical; Yahoo only provides CURRENT fundamentals "
                                  "and analyst ratings, so those are today's values (multiples can be re-priced).")
        reprice = st.checkbox("Re-price valuation multiples to that date", True)
        universe_choice = st.radio("Universe", ["S&P 500 (+ your holdings)", "Workbook list (173)", "Both"],
                                   help="S&P 500 is fetched live from Wikipedia. Ledger holdings are always added.")
        with_ebitda = st.checkbox("Fetch EBITDA growth (slower)", True)
        use_cache = st.checkbox("Use 6-hour disk cache", True)
        extra = st.text_input("Extra tickers to score (comma-separated)", "")
    else:
        asof = None; reprice = True; universe_choice = ""; with_ebitda = False; use_cache = True; extra = ""
        st.caption("Snapshot is fixed at Nov 20, 2026; the as-of date and moving-average switch need Live mode.")
    refresh = st.button("Run / refresh data", type="primary")

    st.header("Assumptions")
    base = Config()
    ma_type = st.radio("Moving average for 50-day trend & ATR", ["EMA", "SMA"], index=0, horizontal=True,
                       help="EMA: exponential 50-day average and Wilder-smoothed 14-day ATR. SMA: simple averages. "
                            "Applies to live data; the snapshot carries fixed workbook values.")
    max_daily = st.slider("Max trades/day after Week 1", 0, 5, base.max_daily_trades)
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
              base_risk=risk, min_mkt_cap_b=mcap, ma_type=ma_type)


# ------------------------------------------------------------------ data load
@st.cache_data(show_spinner=False)
def _snapshot():
    return load_snapshot()


def load_data():
    if not live:
        uni, inputs, mkt = _snapshot()
        return uni, inputs, mkt, []
    held = set(ss.ledger.ticker)
    wb_list = load_universe_list()
    sp = None
    if not universe_choice.startswith("Workbook"):
        try:
            sp = get_sp500()
        except Exception as e:
            st.warning(f"Could not load the S&P 500 list ({e}); using the workbook universe instead.")
    meta = pd.concat([sp, wb_list]) if sp is not None else wb_list
    if sp is None or universe_choice.startswith("Workbook"):
        tickers = set(wb_list.ticker) | held
    elif universe_choice.startswith("S&P"):
        tickers = set(sp.ticker) | held
    else:
        tickers = set(sp.ticker) | set(wb_list.ticker) | held
    tickers |= {t.strip().upper() for t in extra.split(",") if t.strip()}
    bar = st.progress(0.0, text="Fetching from Yahoo Finance…")
    out = fetch_live(sorted(tickers), asof=pd.Timestamp(asof), inception=pd.Timestamp(cfg.inception),
                     with_ebitda_growth=with_ebitda, use_cache=use_cache, meta=meta, reprice=reprice,
                     progress=lambda p, t: bar.progress(min(max(p, 0.0), 1.0), text=str(t)))
    bar.empty()
    return out


sel_key = (mode, str(asof), reprice, universe_choice, with_ebitda, extra)
if "raw" not in ss or refresh:
    try:
        ss.raw = load_data(); ss.raw_key = sel_key
    except Exception as e:
        st.error(f"Data load failed: {e}")
        if "raw" not in ss: st.stop()
elif ss.raw_key != sel_key:
    st.info("Data settings changed — click **Run / refresh data** to reload.")

uni, inputs, mkt, issues = ss.raw
loaded_asof = pd.Timestamp(ss.raw_key[1]).date() if live and ss.raw_key[0].startswith("Live") else None
run_cfg = replace(cfg, valuation=pd.Timestamp(mkt["last_date"]).date()) if "last_date" in mkt else cfg
try:
    R = run_model(uni, inputs, mkt, ss.ledger, run_cfg, asof=loaded_asof)
    err = None
except Exception as e:
    R, err = None, str(e)

if R is not None and not live and pd.to_datetime(ss.ledger.timestamp.str.replace(" - ", " ", regex=False)).dt.date.max() > run_cfg.valuation:
    st.warning("Some trades are dated after the Nov 20, 2026 snapshot, so they are valued at snapshot prices. "
               "Switch to Live mode for current prices.")
if ss.get("flash"):
    st.success(ss.flash); ss.flash = None
if issues:
    with st.expander(f"{len(issues)} data notes / warnings"):
        st.write("\n".join(f"- {i}" for i in issues))
if err:
    st.error(err)
    if "missing from the universe" in err:
        st.info("Add the new ticker(s) under *Extra tickers* in the sidebar (or choose a larger universe) and click Run / refresh data.")

tabs = st.tabs(["Executive summary", "Portfolio", "Scoring engine", "Candidates & rotation",
                "Ledger & compliance", "Manage portfolio", "Export"])
tab_sum, tab_pos, tab_score, tab_cand, tab_led, tab_manage, tab_exp = tabs

if R is not None:
    S = R["summary"]
    st.caption(f"{'As of ' + str(run_cfg.valuation) if live else 'Snapshot'} · {mode} · S&P 500 {mkt['spx_end']:,.0f} · "
               f"VIX {mkt['vix_end']:.2f} · {ma_type if live else 'workbook'} averages · regime multiplier "
               f"{S['regime']['multiplier']}x ({'Defensive' if S['regime']['defensive'] else 'Risk-On'} gear)")

# ------------------------------------------------------------------ summary
with tab_sum:
    if R is not None:
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
    if R is not None:
        st.subheader("Open positions")
        cols = ["ticker", "name", "sector", "shares", "avg_cost", "price", "market_value", "unreal_pl", "unreal_ret",
                "weight", "beta", "atr14", "dma50", "trend_gate", "stop_2atr", "risk_cap", "over_cap",
                "g_value", "g_growth", "g_profitability", "g_momentum", "g_sentiment", "quant_score", "rating", "action"]
        st.dataframe(R["positions"][cols].style.format(
            {"avg_cost": "{:,.2f}", "price": "{:,.2f}", "market_value": "{:,.0f}", "unreal_pl": "{:,.0f}",
             "unreal_ret": "{:.1%}", "weight": "{:.2%}", "beta": "{:.2f}", "atr14": "{:.2f}", "dma50": "{:,.2f}",
             "stop_2atr": "{:,.2f}", "risk_cap": "{:,.0f}", "quant_score": "{:.2f}"}),
            hide_index=True, use_container_width=True)
        st.subheader("Closed round trips")
        st.dataframe(R["closed"], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ scoring
with tab_score:
    if R is not None:
        d = R["scored"]
        st.subheader(f"Scoring engine — {'as of ' + str(run_cfg.valuation) if live else 'Nov 20, 2026 snapshot'}")
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
        st.caption(f"{len(v)} of {len(d)} stocks. Preferreds, depositary shares and units are excluded automatically.")
        st.bar_chart(d.rating.value_counts().reindex(["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]))

# ------------------------------------------------------------------ candidates
with tab_cand:
    if R is not None:
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
    if R is not None:
        st.subheader("Ledger with daily-limit check")
        st.dataframe(R["ledger"].drop(columns=["ts"]), hide_index=True, use_container_width=True)
        st.subheader("End-of-day cash vs ceiling")
        cp = R["cash_path"]
        st.dataframe(cp.style.format({"cash": "${:,.0f}", "cash_pct_initial": "{:.2%}"}), use_container_width=True)
        st.line_chart(cp["cash_pct_initial"])

# ------------------------------------------------------------------ manage portfolio
with tab_manage:
    st.subheader("Update the portfolio")
    st.caption("The portfolio is the trade ledger: opening a position = a buy, closing = a sell. Changes apply to this "
               "session; download the ledger CSV to keep them (or commit it as data/ledger.csv).")
    price_map = R["scored"].set_index("ticker")["price"].to_dict() if R is not None else {}

    up, add = st.columns(2)
    with up:
        st.markdown("**1 · Upload a CSV**")
        f = st.file_uploader("Trade ledger CSV, or an open-positions export", type="csv", key="ledger_upload")
        raw_df = None
        if f is not None:
            try:
                raw_df = pd.read_csv(f); f.seek(0)
            except Exception as e:
                st.error(f"Could not read that file: {e}")
        if raw_df is not None and pf.is_positions_csv(raw_df):
            st.info("Detected an **open-positions snapshot** (no trade dates). Choose the snapshot date and how to apply it.")
            try:
                pos_in = pf.read_positions(raw_df)
                m = re.search(r"(\d{1,2})_(\d{1,2})_(\d{4})", f.name)
                snap_default = date(int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else (loaded_asof or TODAY)
                c = st.columns(2)
                snap_d = c[0].date_input("Positions as of", snap_default, key="snap_date")
                snap_t = c[1].time_input("Time", dtime(16, 0), key="snap_time")
                snap_ts = datetime.combine(snap_d, snap_t).strftime(pf.TS_FMT)
                names = dict(zip(pos_in.ticker, pos_in.name))
                how_pos = st.radio("How to apply", ["Sync current ledger to these positions (keeps history)",
                                                    "Rebuild ledger from these positions only"], key="how_pos")
                if how_pos.startswith("Sync"):
                    for w in pf.cost_mismatches(ss.ledger, pos_in):
                        st.warning("Average cost differs — " + w)
                    tr = pf.sync_trades(ss.ledger, pos_in, snap_ts, price_map)
                    if tr.empty:
                        st.success("The ledger already matches this file — no trades needed.")
                    else:
                        st.caption("Proposed trades. The file has no exit prices, so confirm or edit the sell prices.")
                        edited_tr = st.data_editor(tr, hide_index=True, disabled=["ticker", "side", "quantity", "note"],
                                                   key=f"sync_{f.name}_{ss.ledger_version}", use_container_width=True)
                        if st.button("Apply sync"):
                            if (edited_tr.price <= 0).any():
                                st.error("Enter a price for every trade (sell prices are not in the file).")
                            else:
                                set_ledger(pd.concat([ss.ledger, pf.trades_to_rows(edited_tr, snap_ts, names)], ignore_index=True),
                                           f"Synced ledger to {len(pos_in)} open positions ({len(edited_tr)} trade(s) added).")
                else:
                    st.warning("This replaces the ledger with one buy per position at PricePaid. Trade history, closed "
                               "positions, dates and fees are lost, so cash, Dietz return and compliance will not be meaningful.")
                    if st.button("Rebuild ledger from positions"):
                        set_ledger(pf.positions_to_ledger(pos_in, snap_ts), f"Ledger rebuilt from {len(pos_in)} positions.")
            except Exception as e:
                st.error(f"Could not process the positions file: {e}")
        elif raw_df is not None:
            how = st.radio("When importing", ["Replace the ledger", "Append to the ledger"], horizontal=True)
            if st.button("Import CSV"):
                try:
                    new = pf.normalize_ledger(raw_df)
                    if how.startswith("Append"):
                        new = pd.concat([ss.ledger, new], ignore_index=True)
                    set_ledger(new, f"Imported {len(new)} ledger rows.")
                except Exception as e:
                    st.error(f"Could not read that CSV: {e}")
        if st.button("Reset to bundled ledger"):
            set_ledger(pf.normalize_ledger(DATA / "ledger.csv"), "Ledger reset to the bundled TradeQuest history.")

    with add:
        st.markdown("**2 · Open / add to a position**")
        with st.form("add_trade", clear_on_submit=True):
            c = st.columns(3)
            tk = c[0].text_input("Ticker").strip().upper()
            side = c[1].selectbox("Side", ["Buy", "Sell"])
            qty = c[2].number_input("Shares", 0, 10_000_000, 0, 1)
            c = st.columns(3)
            px = c[0].number_input("Price ($, 0 = latest price)", 0.0, 100000.0, 0.0, 0.01)
            td = c[1].date_input("Trade date", loaded_asof or TODAY)
            tt = c[2].time_input("Time", datetime.now().replace(second=0, microsecond=0).time())
            if st.form_submit_button("Add trade"):
                tk = tk.replace(".", "-")
                price = px or price_map.get(tk, 0.0)
                if not tk or qty <= 0 or price <= 0:
                    st.error("Enter a ticker, a share quantity and a price (no latest price is available for this ticker).")
                else:
                    q_signed = qty if side == "Buy" else -qty
                    row = pd.DataFrame([dict(ticker=tk, order_type=f"Manual - {side}", label=tk, quantity=q_signed,
                                             price=price, net_cash=-q_signed * price,
                                             timestamp=datetime.combine(td, tt).strftime(pf.TS_FMT))])
                    set_ledger(pd.concat([ss.ledger, row], ignore_index=True), f"{side} {qty:,} {tk} @ {price:,.2f} added.")

        st.markdown("**3 · Close a position**")
        if R is not None:
            with st.form("close_pos"):
                c = st.columns(3)
                ct = c[0].selectbox("Open position", list(R["positions"].ticker))
                cur = float(price_map.get(ct, 0.0))
                cpx = c[1].number_input("Sell price ($, 0 = latest)", 0.0, 100000.0, 0.0, 0.01)
                cd = c[2].date_input("Date", loaded_asof or TODAY, key="close_date")
                if st.form_submit_button("Close entire position"):
                    sh = float(R["positions"].set_index("ticker").loc[ct, "shares"])
                    price = cpx or cur
                    row = pd.DataFrame([dict(ticker=ct, order_type="Manual - Sell", label=ct, quantity=-sh, price=price,
                                             net_cash=sh * price,
                                             timestamp=datetime.combine(cd, datetime.now().time().replace(second=0, microsecond=0)).strftime(pf.TS_FMT))])
                    set_ledger(pd.concat([ss.ledger, row], ignore_index=True), f"Closed {ct}: sold {sh:,.0f} @ {price:,.2f}.")
        else:
            st.info("Fix the ledger error above to enable closing positions.")

    st.markdown("**4 · Edit or remove individual trades**")
    st.caption("Edit cells, add rows at the bottom, or select rows and press Delete; then click Apply. "
               "Leave net_cash blank to derive it. Sells use negative quantities.")
    edit_cols = ["ticker", "order_type", "quantity", "price", "net_cash", "timestamp"]
    edited = st.data_editor(ss.ledger[edit_cols], num_rows="dynamic", use_container_width=True,
                            key=f"ledger_editor_{ss.ledger_version}")
    if st.button("Apply edits"):
        try:
            new = pf.normalize_ledger(edited.dropna(how="all"))
            set_ledger(new, f"Ledger updated ({len(new)} rows).")
        except Exception as e:
            st.error(f"Could not apply edits: {e}")
    st.download_button("Download current ledger (.csv)", ss.ledger.to_csv(index=False), file_name="ledger.csv")

# ------------------------------------------------------------------ export
with tab_exp:
    if R is not None:
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
        st.download_button("Download results (.xlsx)", buf.getvalue(), file_name=f"bibf_model_{run_cfg.valuation}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download scoring table (.csv)", R["scored"].to_csv(index=False), file_name="bibf_scoring.csv")
        st.json(run_cfg.as_dict(), expanded=False)
