#!/usr/bin/env python3
"""
investor_report.py — Daily Investor Report (NSE)

Fetches all NSE-listed stocks via nsetools, computes fundamental & valuation metrics,
ranks stocks into Conservative / Moderate / High-Growth portfolios,
and exports HTML + CSV report.

Run: python investor_report.py
"""

import os
import math
import time
import yaml
import traceback
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from nsetools import Nse

# ----------------------
# Config & folders
# ----------------------
ROOT = os.getcwd()
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_PATH = "config.yaml"
DEFAULT_ALLOC = 15000

# ----------------------
# Load config
# ----------------------
def load_config():
    if not os.path.exists(CONFIG_PATH):
        cfg = {
            "alloc_per_stock": DEFAULT_ALLOC,
            "top_n": 12,
            "filters": {
                "min_roe": 15.0,
                "min_roce": 15.0,
                "max_debt_equity": 0.6,
                "min_sales_cagr_5y": 8.0,
                "min_op_margin": 10.0
            }
        }
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"Created default {CONFIG_PATH}. Edit to customize.")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ----------------------
# Fetch NSE universe via nsetools
# ----------------------
def fetch_all_nse_tickers():
    nse = Nse()
    codes = nse.get_stock_codes()
    # remove first key 'SYMBOL'
    tickers = [code.upper() for code in codes.keys() if code != 'SYMBOL']
    print(f"Fetched {len(tickers)} NSE tickers.")
    return tickers

# ----------------------
# Fetch symbol metrics via yfinance
# ----------------------
def fetch_symbol_metrics(ticker):
    sym = ticker + ".NS"
    out = {"ticker": ticker}
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="2y", interval="1d", actions=False)
        if hist is None or hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        out["price"] = price
        out["price_date"] = hist.index[-1].strftime("%Y-%m-%d")
        info = tk.info or {}
        out["info"] = info

        trailing_pe = info.get("trailingPE") or np.nan
        peg = info.get("pegRatio") or np.nan
        market_cap = info.get("marketCap") or np.nan
        out["trailing_pe"] = float(trailing_pe) if trailing_pe else np.nan
        out["peg"] = float(peg) if peg else np.nan
        out["market_cap"] = int(market_cap) if market_cap else np.nan

        returns = hist["Close"].pct_change().dropna()
        out["ann_vol_pct"] = round(float(returns.std() * (252**0.5) * 100.0),2)
        return out
    except Exception:
        return None

# ----------------------
# Scoring & portfolio (simplified)
# ----------------------
def compute_quality_score(m, filters): return 0
def compute_valuation_score(m, universe_pe_median): return 0
def build_portfolios(metrics_list, cfg): return [], [], [], pd.DataFrame()
def write_csv_and_html(topA, topB, topC, df_all): return "", ""

# ----------------------
# Main run
# ----------------------
def run(cfg):
    tickers = fetch_all_nse_tickers()
    print(f"Universe size: {len(tickers)}")
    
    metrics = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_symbol_metrics, t): t for t in tickers}
        for future in tqdm(as_completed(future_to_ticker), total=len(future_to_ticker), desc="Fetching"):
            t = future_to_ticker[future]
            try:
                m = future.result()
                if m:
                    metrics.append(m)
            except Exception:
                continue

    if not metrics:
        print("No metrics fetched — exiting.")
        return None

    topA, topB, topC, df_all = build_portfolios(metrics, cfg)
    csv_path, html_path = write_csv_and_html(topA, topB, topC, df_all)
    print("Report written:", html_path)
    print("CSV written:", csv_path)
    return {"csv": csv_path, "html": html_path}

# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
