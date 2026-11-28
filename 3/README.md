# BIBF TradeQuest Quantamental & Risk Model (Python / Streamlit)

Python port of the Excel framework (revised version), with live Yahoo Finance data.

    pip install -r requirements.txt
    streamlit run app.py
    python tests/test_parity.py     # offline check against the Excel model

Layout
- `bibf/config.py`    all assumptions (weights, caps, Week-1 window, fees, regime rules)
- `bibf/scoring.py`   5-factor sector-relative engine, hold-cap, ratings, eligibility
- `bibf/portfolio.py` ledger -> positions, compliance, regime, risk caps, Modified Dietz, rotation
- `bibf/data.py`      snapshot loader + yfinance extraction
- `bibf/pipeline.py`  `run_model(...)` returns every table
- `data/`             universe, Nov-20-2026 snapshot, ledger

Live-data notes
- Fundamentals come from Yahoo `info` (TTM). EPS growth uses `earningsGrowth`; EBITDA growth is the YoY change in
  the latest quarterly EBITDA; analyst sentiment is (Strong Buy + Buy) / all ratings. Missing values are skipped
  within a factor (sentiment defaults to 0.5), exactly as in the workbook.
- Beta is computed from 252 daily returns vs ^GSPC; ATR is the 14-day simple mean of true range; 50-DMA is the
  mean of the last 50 closes. The Excel model used hard-coded figures for these.
- Live scores will differ from the Nov-20 workbook because inputs and date differ, and Yahoo fields are not
  identical to the workbook's original data vendor.
- The 173-stock universe is `data/universe.csv`; add tickers in the sidebar. Preferreds/units are auto-excluded.

New in this version
- **EMA / SMA switch** (sidebar). Live data returns both versions of the 50-day average and the 14-day ATR
  (EMA = exponential span-50 / Wilder smoothing; SMA = simple mean), so switching needs no re-download. The frozen
  Nov-20 snapshot carries the workbook's fixed values and is not switchable.
- **As-of date** (Live mode). Re-scores the universe and rebuilds the portfolio as of any past trading day.
  Prices, returns, beta, ATR and moving averages are truly historical; Yahoo exposes only CURRENT fundamentals and
  analyst ratings, so those are today's values (multiples are re-priced by price(asof)/price(today) by default).
- **Max trades/day after Week 1** slider is 0-5.
- **Manage portfolio tab**: import a CSV (replace or append), add/open a position, close a position, edit or delete
  individual trades, download the ledger. Sells larger than the held quantity are rejected. The CSV reader accepts
  the model format or a broker-style export (aliases for column names, buy/sell column with unsigned quantities,
  separate date/time columns, $ and thousands separators; net cash is derived when absent).

Tests (no network needed): `python tests/test_parity.py`, `python tests/test_edit.py`, `python tests/smoke_app.py`

Open-positions CSV (e.g. `OpenPosition_11_27_2026.csv`: Symbol, Description, Quantity, LastPrice, PricePaid, ...)
- Detected automatically in Manage portfolio. "Sync" compares it with the ledger and proposes only the missing trades
  (closed positions -> sell, new tickers -> buy at PricePaid, changed quantities -> buy/sell the difference). The file
  has no exit prices or trade dates, so you confirm sell prices and the snapshot date (read from the filename).
- "Rebuild" instead creates a fresh ledger of one buy per position (history, closed trades, fees and dates are lost).
- Sample file in `tests/`; `tests/test_edit.py` checks it produces exactly: sell INTC, buy NEE.
