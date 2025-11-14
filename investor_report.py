#!/usr/bin/env python3
"""
investor_report.py

Daily Investor Report (NSE) — produces:
 - outputs/report.html
 - outputs/top_picks.csv

Enhanced: uses yfinance primarily, but falls back to nsetools for live NSE prices.
Designed for long-term investors with max ₹15,000 per stock.
"""

import os
import math
import time
import yaml
import traceback
from datetime import datetime
from tqdm import tqdm

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

FALLBACK_UNIVERSE = [
    "TCS", "RELIANCE", "INFY", "HDFCBANK", "HDFC", "ICICIBANK",
    "KOTAKBANK", "HINDUNILVR", "ITC", "BHARTIARTL", "LT", "AXISBANK",
    "SBIN", "BAJAJ-AUTO", "MARUTI", "ULTRACEMCO", "TITAN", "SUNPHARMA",
    "NTPC", "ONGC", "POWERGRID", "BPCL", "IOC", "ADANIPORTS"
]

nse = Nse()

# ----------------------
# Load or create config
# ----------------------
def load_config():
    if not os.path.exists(CONFIG_PATH):
        cfg = {
            "alloc_per_stock": DEFAULT_ALLOC,
            "watchlist_url": "https://raw.githubusercontent.com/datasets/nifty50/master/data/nifty_50.csv",
            "watchlist_fallback": FALLBACK_UNIVERSE,
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
# Fetch universe
# ----------------------
def fetch_universe(cfg):
    url = cfg.get("watchlist_url")
    try:
        if url:
            df = pd.read_csv(url)
            for c in ['Symbol','Ticker','Code','symbol','ticker','code','Company']:
                if c in df.columns:
                    syms = df[c].astype(str).str.strip().str.replace(r'\.NS$','', regex=True).str.upper().tolist()
                    syms = [s for s in syms if len(s)>0]
                    if len(syms) >= 10:
                        return list(dict.fromkeys(syms))
    except Exception:
        pass
    return cfg.get("watchlist_fallback", FALLBACK_UNIVERSE)

# ----------------------
# Fetch single ticker metrics
# ----------------------
def fetch_symbol_metrics(ticker):
    """Return dict with price and basic metrics. Yfinance primary, nsetools fallback."""
    sym_ns = ticker + ".NS"
    out = {"ticker": ticker}

    # --- Try yfinance first ---
    try:
        tk = yf.Ticker(sym_ns)
        hist = tk.history(period="5y", interval="1d", actions=False)
        if hist is None or hist.empty:
            hist = tk.history(period="2y", interval="1d", actions=False)
        if hist is not None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
            out["price"] = price
            out["price_date"] = hist.index[-1].strftime("%Y-%m-%d")
            info = tk.info or {}
            out["info"] = info
            out["trailing_pe"] = float(info.get("trailingPE") or np.nan)
            out["peg"] = float(info.get("pegRatio") or np.nan)
            out["market_cap"] = int(info.get("marketCap") or np.nan)
            return out
    except Exception:
        pass

    # --- Fallback to nsetools ---
    try:
        data = nse.get_quote(ticker.upper())
        if data and "lastPrice" in data:
            out["price"] = float(data["lastPrice"])
            out["price_date"] = datetime.now().strftime("%Y-%m-%d")
            out["trailing_pe"] = float(data.get("p/e") or np.nan)
            out["peg"] = np.nan
            out["market_cap"] = float(data.get("marketCap") or np.nan)
            out["info"] = {}
            return out
    except Exception:
        pass

    return None

# ----------------------
# Scoring & filtering (same as your script)
# ----------------------
def compute_quality_score(m, filters):
    if m is None: return 0.0
    score = 0.0; total = 0.0
    # ROE
    total += 30
    roe = m.get("roe_pct", np.nan)
    if not np.isnan(roe) and roe >= filters.get("min_roe",15): score += 30
    elif not np.isnan(roe): score += max(0, 30*(roe/filters.get("min_roe",15)))
    # ROCE
    total += 25
    roce = m.get("roce_pct", np.nan)
    if not np.isnan(roce) and roce >= filters.get("min_roce",15): score += 25
    elif not np.isnan(roce): score += max(0, 25*(roce/filters.get("min_roce",15)))
    # Debt/equity
    total += 15
    de = m.get("debt_to_equity", np.nan)
    if not np.isnan(de): score += 15 * max(0, 1-(de/filters.get("max_debt_equity",0.6)))
    # Sales CAGR
    total += 15
    sg = m.get("sales_cagr_5y_pct", np.nan)
    if not np.isnan(sg): score += 15 * min(1.0, sg/max(1.0, filters.get("min_sales_cagr_5y",8.0)))
    # Op Margin
    total += 15
    om = m.get("op_margin_pct", np.nan)
    if not np.isnan(om): score += 15 * min(1.0, om/max(1.0, filters.get("min_op_margin",10.0)))
    if total==0: return 0.0
    return round((score/total)*100.0,2)

def compute_valuation_score(m, universe_pe_median):
    score = 0.0
    pe = m.get("trailing_pe", np.nan)
    peg = m.get("peg", np.nan)
    if not np.isnan(pe) and not np.isnan(universe_pe_median):
        ratio = pe/universe_pe_median if universe_pe_median>0 else 1.0
        if ratio <=0.7: score+=60
        elif ratio>=1.5: score+=0
        else: score+=60*(1-(ratio-0.7)/(1.5-0.7))
    elif not np.isnan(pe):
        if pe<=15: score+=60
        elif pe>=30: score+=0
        else: score+=60*(1-(pe-15)/15)
    if not np.isnan(peg):
        if peg<=1.0: score+=25
        elif peg>=2.0: score+=0
        else: score+=25*(1-(peg-1.0)/1.0)
    return round(score,2)

# ----------------------
# Build portfolios (same logic as your script)
# ----------------------
def build_portfolios(metrics_list, cfg):
    df = pd.DataFrame(metrics_list)
    pe_vals = df["trailing_pe"].dropna().astype(float)
    universe_pe_median = float(pe_vals.median()) if not pe_vals.empty else np.nan
    df["quality_score"] = df.apply(lambda r: compute_quality_score(r, cfg.get("filters",{})), axis=1)
    df["valuation_score"] = df.apply(lambda r: compute_valuation_score(r, universe_pe_median), axis=1)
    df["score_A"] = df["quality_score"]*0.8 + df["valuation_score"]*0.2
    df["score_B"] = df["quality_score"]*0.6 + df["valuation_score"]*0.4
    df["score_C"] = df["quality_score"]*0.3 + df["valuation_score"]*0.1
    top_n = int(cfg.get("top_n",12))
    topA = df.sort_values("score_A",ascending=False).head(top_n).reset_index(drop=True)
    topB = df.sort_values("score_B",ascending=False).head(top_n).reset_index(drop=True)
    topC = df.sort_values("score_C",ascending=False).head(top_n).reset_index(drop=True)
    alloc = cfg.get("alloc_per_stock",DEFAULT_ALLOC)
    def allocate(row):
        price = row.get("price",np.nan)
        if np.isnan(price) or price<=0: return pd.Series({"alloc_amount":alloc,"shares":0,"cash_left":alloc})
        shares = int(math.floor(alloc/price))
        cash_left = round(alloc-shares*price,2)
        return pd.Series({"alloc_amount":alloc,"shares":shares,"cash_left":cash_left})
    for df_sel in (topA,topB,topC):
        cols = df_sel.apply(allocate,axis=1)
        df_sel[["alloc_amount","shares","cash_left"]] = cols
    return topA, topB, topC, df

# ----------------------
# Report writing
# ----------------------
def write_csv_and_html(topA,topB,topC,df_all):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR,f"top_picks_{ts}.csv")
    df_all.to_csv(csv_path,index=False)
    html_path = os.path.join(OUT_DIR,f"report_{ts}.html")
    html = "<html><head><meta charset='utf-8'><title>Daily Investor Report</title></head><body>"
    html += f"<h1>Daily Investor Report — {datetime.now().strftime('%Y-%m-%d')}</h1>"
    html += df_all.head(50).to_html(index=False)
    html += "</body></html>"
    with open(html_path,"w",encoding="utf-8") as f:
        f.write(html)
    return csv_path, html_path

# ----------------------
# Main run
# ----------------------
def run(cfg):
    tickers = fetch_universe(cfg)
    print(f"Universe size: {len(tickers)}")
    metrics = []
    for t in tqdm(tickers, desc="Fetching"):
        try:
            m = fetch_symbol_metrics(t)
            if m:
                metrics.append(m)
            time.sleep(0.1)
        except Exception:
            traceback.print_exc()
            continue
    if not metrics:
        print("No metrics fetched — exiting.")
        return None
    topA,topB,topC,df_all = build_portfolios(metrics,cfg)
    csv_path, html_path = write_csv_and_html(topA,topB,topC,df_all)
    print("Report written:", html_path)
    print("CSV written:", csv_path)
    return {"csv":csv_path,"html":html_path}

# ----------------------
# CLI
# ----------------------
if __name__=="__main__":
    cfg = load_config()
    res = run(cfg)
    if res:
        print("DONE.")
