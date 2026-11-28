import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from dataclasses import replace
from bibf.config import Config
from bibf.data import load_snapshot, DATA
from bibf import portfolio as pf
from bibf.pipeline import run_model

uni, inputs, mkt = load_snapshot()
led = pf.normalize_ledger(DATA / "ledger.csv")
base = run_model(uni, inputs, mkt, led, Config())
# 1) open a new position (manual buy) and close another (manual full sell)
px = uni.set_index("ticker").price
new = pd.DataFrame([
    dict(ticker="V", order_type="Manual - Buy", label="V", quantity=100, price=px["V"], net_cash=-100 * px["V"], timestamp="11/20/2026 - 16:00"),
    dict(ticker="WAT", order_type="Manual - Sell", label="WAT", quantity=-468, price=px["WAT"], net_cash=468 * px["WAT"], timestamp="11/20/2026 - 16:01"),
])
r = run_model(uni, inputs, mkt, pd.concat([led, new], ignore_index=True), Config())
assert "V" in set(r["positions"].ticker) and "WAT" not in set(r["positions"].ticker)
assert "WAT" in set(r["closed"].ticker) and len(r["positions"]) == len(base["positions"])
# 2) selling more than held is rejected
bad = pd.concat([led, new.assign(quantity=[100, -1000]).iloc[[1]]], ignore_index=True)
try: run_model(uni, inputs, mkt, bad, Config()); raise SystemExit("should have failed")
except ValueError as e: assert "Invalid ledger" in str(e)
# 3) as-of date trims the ledger
r13 = run_model(uni, inputs, mkt, led, Config(), asof="2026-11-13")
assert len(r13["ledger_used"]) == 18
# 4) 0 trades/day allowed
r0 = run_model(uni, inputs, mkt, led, replace(Config(), max_daily_trades=0))
assert r0["governance"].set_index("Check").loc["Post-Week-1 daily trade cap", "Status"].startswith("BREACH")
print("edit/as-of/limit tests OK")

# 5) broker "open positions" export -> sync (sample: OpenPosition_11_27_2026.csv)
sample = pathlib.Path(__file__).resolve().parent / "sample_OpenPosition_11_27_2026.csv"
pos = pf.read_positions(sample)
assert pf.is_positions_csv(pd.read_csv(sample)) and len(pos) == 16
tr = pf.sync_trades(led, pos, "11/27/2026 - 16:00", px.to_dict())
assert list(tr.ticker) == ["INTC", "NEE"] and list(tr.side) == ["Sell", "Buy"], tr
assert pf.cost_mismatches(led, pos) == []
synced = pd.concat([led, pf.trades_to_rows(tr, "11/27/2026 - 16:00")], ignore_index=True)
rs = run_model(uni, inputs, mkt, synced, Config())
assert set(rs["positions"].ticker) == set(pos.ticker) and len(rs["positions"]) == 16
rb = run_model(uni, inputs, mkt, pf.positions_to_ledger(pos, "11/27/2026 - 16:00"), Config())
assert set(rb["positions"].ticker) == set(pos.ticker)
print("open-positions import tests OK")
