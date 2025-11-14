#!/usr/bin/env python3
"""
investor_report.py

Daily Investor Report (NSE) — produces:
 - outputs/report.html
 - outputs/top_picks.csv

Designed for long-term investors with max ₹15,000 per stock.
This script fetches data from yfinance, computes simple fundamental/valuation metrics,
ranks stocks into Conservative / Moderate / High-Growth portfolios, and exports HTML+CSV.

Run: python investor_report.py
"""

import os
import math
import time
import yaml
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------
# Config & folders
# ----------------------
ROOT = os.getcwd()
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_PATH = "config.yaml"
DEFAULT_ALLOC = 15000

# Default small universe fallback (safe)
FALLBACK_UNIVERSE = [
    "TCS", "RELIANCE", "INFY", "HDFCBANK", "HDFC", "ICICIBANK",
    "KOTAKBANK", "HINDUNILVR", "ITC", "BHARTIARTL", "LT", "AXISBANK",
    "SBIN", "BAJAJ-AUTO", "MARUTI", "ULTRACEMCO", "TITAN", "SUNPHARMA",
    "NTPC", "ONGC", "POWERGRID", "BPCL", "IOC", "ADANIPORTS"
]

# Try to load config or create default
def load_config():
    if not os.path.exists(CONFIG_PATH):
        cfg = {
            "alloc_per_stock": DEFAULT_ALLOC,
            "watchlist_url": "https://raw.githubusercontent.com/datasets/nifty50/master/data/nifty_50.csv",
            "watchlist_fallback": FALLBACK_UNIVERSE,
            "top_n": 12,
            # thresholds (used by screening)
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
# Universe fetch
# ----------------------
def fetch_universe(cfg):
    url = cfg.get("watchlist_url")
    try:
        if url:
            df = pd.read_csv(url)
            # try to find column with symbols
            for c in ['Symbol','Ticker','Code','symbol','ticker','code','Company']:
                if c in df.columns:
                    syms = df[c].astype(str).str.strip().str.replace(r'\.NS$','', regex=True).str.upper().tolist()
                    syms = [s for s in syms if len(s)>0]
                    if len(syms) >= 10:
                        return list(dict.fromkeys(syms))
    except Exception:
        pass
    # fallback
    return cfg.get("watchlist_fallback", FALLBACK_UNIVERSE)

# ----------------------
# Fetch single ticker fundamentals (best-effort)
# ----------------------
def fetch_symbol_metrics(ticker):
    """Return dict with price and a set of metrics (best-effort via yfinance)."""
    sym = ticker + ".NS"
    out = {"ticker": ticker}
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="5y", interval="1d", actions=False)
        if hist is None or hist.empty:
            hist = tk.history(period="2y", interval="1d", actions=False)
        if hist is None or hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        out["price"] = price
        out["price_date"] = hist.index[-1].strftime("%Y-%m-%d")
        info = tk.info or {}
        out["info"] = info
        fin = {}
        fin["financials"] = getattr(tk, "financials", None)
        fin["balance_sheet"] = getattr(tk, "balance_sheet", None)
        out["financials"] = fin

        # trailing PE / PEG / market cap if available
        trailing_pe = info.get("trailingPE") or np.nan
        peg = info.get("pegRatio") or np.nan
        market_cap = info.get("marketCap") or np.nan
        out["trailing_pe"] = float(trailing_pe) if trailing_pe and not np.isnan(trailing_pe) else np.nan
        out["peg"] = float(peg) if peg and not np.isnan(peg) else np.nan
        out["market_cap"] = int(market_cap) if market_cap and not np.isnan(market_cap) else np.nan

        # compute simple metrics from financials if available
        roe = np.nan; roce = np.nan; debt_equity = np.nan; op_margin = np.nan; sales_cagr_5y = np.nan
        try:
            fin_df = fin["financials"]
            bs_df = fin["balance_sheet"]
            if fin_df is not None and not fin_df.empty and bs_df is not None and not bs_df.empty:
                # try to extract net income
                net_income = None
                for cand in ["Net Income", "Net Income Applicable To Common Shares", "Net Income Common Stockholders"]:
                    if cand in fin_df.index:
                        net_income = float(fin_df.loc[cand].dropna().iloc[0])
                        break
                if net_income is None:
                    # fallback first numeric row
                    try:
                        net_income = float(fin_df.iloc[0].dropna().iloc[0])
                    except Exception:
                        net_income = None

                # shareholder equity
                se = None
                for cand in ['Total Stockholder Equity', 'Total Stockholders Equity', 'Total Equity', 'Total stockholder equity', 'Total shareholders equity']:
                    if cand in bs_df.index:
                        se = float(bs_df.loc[cand].dropna().iloc[0])
                        break
                if se is None:
                    try:
                        se = float(bs_df.iloc[0].dropna().iloc[0])
                    except Exception:
                        se = None
                if net_income is not None and se is not None and se!=0:
                    roe = (net_income / se) * 100.0

                # operating income / revenue for op margin
                ebit = None; revenue = None
                for cand in ["Operating Income", "Ebit", "Operating Income or Loss"]:
                    if fin_df is not None and cand in fin_df.index:
                        ebit = float(fin_df.loc[cand].dropna().iloc[0])
                        break
                for cand in ["Total Revenue","Revenue","Sales"]:
                    if fin_df is not None and cand in fin_df.index:
                        revenue = float(fin_df.loc[cand].dropna().iloc[0])
                        break
                if ebit is not None and revenue is not None and revenue!=0:
                    op_margin = (ebit / revenue) * 100.0

                # debt/equity
                debt = None
                for cand in ["Long Term Debt","Long term debt","Total Debt","Total Liab","Total liabilities"]:
                    if bs_df is not None and cand in bs_df.index:
                        try:
                            debt = float(bs_df.loc[cand].dropna().iloc[0])
                            break
                        except Exception:
                            debt = None
                if debt is not None and se is not None and se!=0:
                    debt_equity = debt / se

                # sales CAGR approx (use revenue series)
                if revenue is not None and fin_df is not None:
                    try:
                        rev_series = None
                        for cand in ["Total Revenue","Revenue","Sales"]:
                            if fin_df is not None and cand in fin_df.index:
                                vec = fin_df.loc[cand].dropna().astype(float)
                                if len(vec) >= 3:
                                    rev_series = vec
                                    break
                        if rev_series is not None and len(rev_series) >= 3:
                            recent = float(rev_series.iloc[0])
                            older = float(rev_series.iloc[min(4, len(rev_series)-1)])
                            years = min(5, len(rev_series)-1)
                            if older>0 and years>0:
                                sales_cagr_5y = ((recent / older) ** (1.0/years) - 1.0) * 100.0
                    except Exception:
                        sales_cagr_5y = np.nan
        except Exception:
            pass

        # approx ROCE: use EBIT / (Total Assets - Current Liabilities) if available
        roce_val = np.nan
        try:
            bs_df = fin.get("balance_sheet")
            if bs_df is not None and not bs_df.empty and ebit is not None:
                # find total assets and current liabilities
                ta = None; cl = None
                for cand in ["Total Assets", "totalAssets","Total assets"]:
                    if cand in bs_df.index:
                        ta = float(bs_df.loc[cand].dropna().iloc[0])
                        break
                for cand in ["Total Current Liabilities", "Current Liabilities"]:
                    if cand in bs_df.index:
                        cl = float(bs_df.loc[cand].dropna().iloc[0])
                        break
                if ta is not None and cl is not None and (ta - cl) != 0:
                    roce_val = (ebit / (ta - cl)) * 100.0
        except Exception:
            roce_val = np.nan

        out["roe_pct"] = round(float(roe),2) if not np.isnan(roe) else np.nan
        out["roce_pct"] = round(float(roce_val),2) if not np.isnan(roce_val) else np.nan
        out["debt_to_equity"] = round(float(debt_equity),3) if not np.isnan(debt_equity) else np.nan
        out["op_margin_pct"] = round(float(op_margin),2) if not np.isnan(op_margin) else np.nan
        out["sales_cagr_5y_pct"] = round(float(sales_cagr_5y),2) if not np.isnan(sales_cagr_5y) else np.nan

        # volatility / annualized
        try:
            returns = hist["Close"].pct_change().dropna()
            ann_vol = returns.std() * (252**0.5) * 100.0
            out["ann_vol_pct"] = round(float(ann_vol),2)
        except Exception:
            out["ann_vol_pct"] = np.nan

        # promoter / insider holding not reliably available via yfinance for NSE; leave NaN
        out["promoter_pct"] = np.nan

        return out
    except Exception as e:
        # noisy but safe
        # print("fetch error", ticker, e)
        return None

# ----------------------
# Scoring & filtering
# ----------------------
def compute_quality_score(m, filters):
    # weighted score: ROE(30), ROCE(25), DebtToEquity(15), SalesCAGR(15), OpMargin(15)
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
    # Debt/equity (lower better)
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
    # 0..100 (higher better). uses trailing_pe and peg
    score = 0.0; total = 100.0
    pe = m.get("trailing_pe", np.nan)
    peg = m.get("peg", np.nan)
    # PE: if below median -> good
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
    # PEG
    if not np.isnan(peg):
        if peg <= 1.0:
            score += 25
        elif peg >= 2.0:
            score += 0
        else:
            score += 25 * (1 - (peg - 1.0)/(1.0))
    # small bonus for lower market cap? not necessary
    return round(score,2)

# ----------------------
# Build portfolios
# ----------------------
def build_portfolios(metrics_list, cfg):
    df = pd.DataFrame(metrics_list)
    # compute universe PE median
    pe_vals = df["trailing_pe"].dropna().astype(float)
    universe_pe_median = float(pe_vals.median()) if not pe_vals.empty else np.nan
    df["quality_score"] = df.apply(lambda r: compute_quality_score(r, cfg.get("filters", {})), axis=1)
    df["valuation_score"] = df.apply(lambda r: compute_valuation_score(r, universe_pe_median), axis=1)

    # composite
    df["score_A"] = df["quality_score"] * 0.8 + df["valuation_score"] * 0.2  # conservative
    df["score_B"] = df["quality_score"] * 0.6 + df["valuation_score"] * 0.4  # moderate
    # growth proxy (use sales growth + roce + quality) for C
    df["growth_proxy"] = df["sales_cagr_5y_pct"].fillna(0) * 0.6 + df["roce_pct"].fillna(0) * 0.4
    # normalize growth to 0..100
    if df["growth_proxy"].max() > 0:
        df["growth_score"] = (df["growth_proxy"] / df["growth_proxy"].max()) * 100.0
    else:
        df["growth_score"] = 0.0
    df["score_C"] = df["growth_score"] * 0.6 + df["quality_score"] * 0.3 + df["valuation_score"] * 0.1

    # top selections
    top_n = int(cfg.get("top_n", 12))
    topA = df.sort_values("score_A", ascending=False).head(top_n).reset_index(drop=True)
    topB = df.sort_values("score_B", ascending=False).head(top_n).reset_index(drop=True)
    topC = df.sort_values("score_C", ascending=False).head(top_n).reset_index(drop=True)

    # allocate shares per stock with alloc constraint
    alloc = cfg.get("alloc_per_stock", DEFAULT_ALLOC)
    def allocate(row):
        price = row.get("price", np.nan)
        if np.isnan(price) or price <= 0:
            return pd.Series({"alloc_amount": alloc, "shares": 0, "cash_left": alloc})
        shares = int(math.floor(alloc / price))
        cash_left = round(alloc - shares * price, 2)
        return pd.Series({"alloc_amount": alloc, "shares": shares, "cash_left": cash_left})
    for df_sel in (topA, topB, topC):
        cols = df_sel.apply(allocate, axis=1)
        df_sel[["alloc_amount","shares","cash_left"]] = cols

    return topA, topB, topC, df

# ----------------------
# Report writing (HTML + CSV)
# ----------------------
def write_csv_and_html(topA, topB, topC, df_all):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"top_picks_{ts}.csv")
    df_all.to_csv(csv_path, index=False)

    # create a presentable HTML
    html_path = os.path.join(OUT_DIR, f"report_{ts}.html")
    def df_to_html_table(df, title):
        cols = ["ticker","price","price_date","quality_score","valuation_score","score_A","score_B","score_C","shares","alloc_amount","cash_left","roe_pct","roce_pct","debt_to_equity","sales_cagr_5y_pct","op_margin_pct","trailing_pe","peg","market_cap"]
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
    html += "<h3>Full Universe Metrics</h3>"
    # include small subset of full universe for brevity
    html += df_all.sort_values("score_A", ascending=False).head(50).to_html(index=False)
    html += "</body></html>"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return csv_path, html_path

# ----------------------
# Main run flow
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
    res = run(cfg)
    if res:
        print("DONE.")
