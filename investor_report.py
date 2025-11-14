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
    tickers = []

    # handle dict or list
    if isinstance(codes, dict):
        tickers = [code.upper() for code in codes.keys() if code != 'SYMBOL']
    elif isinstance(codes, list):
        tickers = [str(code).upper() for code in codes if code != 'SYMBOL']
    else:
        print("Unexpected format from nsetools.get_stock_codes():", type(codes))
    
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
# Scoring & filtering
# ----------------------
def compute_quality_score(m, filters):
    if m is None:
        return 0.0
    score = 0.0; total = 0.0
    # ROE
    total += 30
    roe = m.get("roe_pct", np.nan)
    if not np.isnan(roe) and roe >= filters.get("min_roe",15):
        score += 30
    elif not np.isnan(roe):
        score += max(0, 30 * (roe / filters.get("min_roe",15)))
    # ROCE
    total += 25
    roce = m.get("roce_pct", np.nan)
    if not np.isnan(roce) and roce >= filters.get("min_roce",15):
        score += 25
    elif not np.isnan(roce):
        score += max(0, 25 * (roce / filters.get("min_roce",15)))
    # Debt/equity
    total += 15
    de = m.get("debt_to_equity", np.nan)
    if not np.isnan(de):
        quota = max(0, 1 - (de / filters.get("max_debt_equity",0.6)))
        score += 15 * quota
    # sales cagr
    total += 15
    sg = m.get("sales_cagr_5y_pct", np.nan)
    if not np.isnan(sg):
        quota = min(1.0, sg / max(1.0, filters.get("min_sales_cagr_5y",8.0)))
        score += 15 * quota
    # op margin
    total += 15
    om = m.get("op_margin_pct", np.nan)
    if not np.isnan(om):
        quota = min(1.0, om / max(1.0, filters.get("min_op_margin",10.0)))
        score += 15 * quota

    if total == 0:
        return 0.0
    return round((score/total)*100.0,2)

def compute_valuation_score(m, universe_pe_median):
    score = 0.0
    pe = m.get("trailing_pe", np.nan)
    peg = m.get("peg", np.nan)
    if not np.isnan(pe) and not np.isnan(universe_pe_median):
        ratio = pe / universe_pe_median if universe_pe_median>0 else 1.0
        if ratio <= 0.7:
            score += 60
        elif ratio >= 1.5:
            score += 0
        else:
            score += 60 * (1 - (ratio - 0.7) / (1.5 - 0.7))
    elif not np.isnan(pe):
        if pe <= 15:
            score += 60
        elif pe >= 30:
            score += 0
        else:
            score += 60 * (1 - (pe - 15)/15)
    if not np.isnan(peg):
        if peg <= 1.0:
            score += 25
        elif peg >= 2.0:
            score += 0
        else:
            score += 25 * (1 - (peg - 1.0)/1.0)
    return round(score,2)

# ----------------------
# Build portfolios
# ----------------------
def build_portfolios(metrics_list, cfg):
    df = pd.DataFrame(metrics_list)
    if df.empty:
        return [], [], [], df
    pe_vals = df["trailing_pe"].dropna().astype(float)
    universe_pe_median = float(pe_vals.median()) if not pe_vals.empty else np.nan
    df["quality_score"] = df.apply(lambda r: compute_quality_score(r, cfg.get("filters", {})), axis=1)
    df["valuation_score"] = df.apply(lambda r: compute_valuation_score(r, universe_pe_median), axis=1)

    df["score_A"] = df["quality_score"]*0.8 + df["valuation_score"]*0.2
    df["score_B"] = df["quality_score"]*0.6 + df["valuation_score"]*0.4
    df["score_C"] = df["quality_score"]*0.3 + df["valuation_score"]*0.1  # simplified growth proxy

    top_n = int(cfg.get("top_n", 12))
    topA = df.sort_values("score_A", ascending=False).head(top_n).reset_index(drop=True)
    topB = df.sort_values("score_B", ascending=False).head(top_n).reset_index(drop=True)
    topC = df.sort_values("score_C", ascending=False).head(top_n).reset_index(drop=True)

    alloc = cfg.get("alloc_per_stock", DEFAULT_ALLOC)
    def allocate(row):
        price = row.get("price", np.nan)
        if np.isnan(price) or price<=0:
            return pd.Series({"alloc_amount": alloc, "shares": 0, "cash_left": alloc})
        shares = int(math.floor(alloc/price))
        cash_left = round(alloc - shares*price,2)
        return pd.Series({"alloc_amount": alloc, "shares": shares, "cash_left": cash_left})

    for df_sel in (topA, topB, topC):
        df_sel[["alloc_amount","shares","cash_left"]] = df_sel.apply(allocate, axis=1)

    return topA, topB, topC, df

# ----------------------
# Write CSV & HTML
# ----------------------
def write_csv_and_html(topA, topB, topC, df_all):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"top_picks_{ts}.csv")
    df_all.to_csv(csv_path, index=False)

    html_path = os.path.join(OUT_DIR, f"report_{ts}.html")
    def df_to_html_table(df, title):
        cols = ["ticker","price","price_date","quality_score","valuation_score","score_A","score_B","score_C","shares","alloc_amount","cash_left","trailing_pe","peg","market_cap"]
        present = [c for c in cols if c in df.columns]
        html = f"<h2>{title}</h2>"
        html += df[present].to_html(index=False, classes='table', justify='left', border=0)
        return html

    html = "<html><head><meta charset='utf-8'><title>Daily Investor Report</title>"
    html += "<style>body{font-family:Arial,Helvetica,sans-serif;padding:20px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:6px;text-align:left} th{background:#f2f2f2}</style></head><body>"
    html += f"<h1>Daily Investor Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h1>"
    html += "<p>This report is generated automatically. Allocation per stock: ₹{:,}</p>".format(int(DEFAULT_ALLOC))
    html += df_to_html_table(topA, "Top Conservative Picks (A)")
    html += df_to_html_table(topB, "Top Moderate Picks (B)")
    html += df_to_html_table(topC, "Top High-Growth Picks (C)")
    html += "<h3>Full Universe Metrics (Top 50)</h3>"
    html += df_all.sort_values("score_A", ascending=False).head(50).to_html(index=False)
    html += "</body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return csv_path, html_path

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
