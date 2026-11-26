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
