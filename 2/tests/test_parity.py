"""Offline check: the Python port must reproduce the revised Excel workbook (Nov-20-2026 snapshot)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from bibf.config import Config
from bibf.data import load_snapshot, DATA
from bibf.pipeline import run_model


def test_parity():
    uni, inputs, mkt = load_snapshot()
    r = run_model(uni, inputs, mkt, DATA / "ledger.csv", Config())
    s = r["scored"].merge(uni[["ticker", "wb_score", "wb_rating", "wb_eligible"]], on="ticker", suffixes=("", "_wb"))
    assert (s.quant_score - s.wb_score).abs().max() < 1e-9
    assert (s.rating == s.wb_rating).all()
    assert (s.eligible == (s.wb_eligible == "Yes")).all()
    sm = r["summary"]
    assert abs(sm["nav"] - 3677307.47) < 0.01          # Excel NAV (fees deducted from cash)
    assert abs(sm["cash"] - 363955.24) < 0.01
    assert abs(sm["port_score"] - 2.4454023168697) < 1e-9
    assert abs(sm["dietz_ret"] - (-0.10768546863)) < 1e-6
    assert abs(sm["realized"] - (-130424.21)) < 0.01
    assert abs(sm["beta"] - 0.655663794167312) < 1e-9
    assert (r["governance"].set_index("Check").loc["Post-Week-1 daily trade cap", "Status"]).startswith("BREACH")
    print("parity OK")


if __name__ == "__main__":
    test_parity()
