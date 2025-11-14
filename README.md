# Daily Investor Report (NSE) — GitHub Actions + Gmail

This repository runs a daily long-term investor screen for NSE stocks and emails a report.

## What it does
- Fetches NSE stock fundamentals via yfinance
- Scores & ranks stocks under 3 investor profiles:
  - Conservative (A)
  - Moderate (B)
  - High-Growth (C)
- Produces HTML report + CSV and emails them daily.

## Setup

1. **Create repo** and push these files (`investor_report.py`, `requirements.txt`, `.github/workflows/daily-report.yml`, `config.yaml`).

2. **Create Gmail App Password**:
   - For Gmail, enable 2-Step Verification on your Google account.
   - Create an App Password (select Mail → Other) and copy the 16-character app password.

3. **Add GitHub Secrets**:
   - Go to your repository → Settings → Secrets → Actions → New repository secret.
   - Add:
     - `GMAIL_USER` = your Gmail address (e.g. `you@gmail.com`)
     - `GMAIL_APP_PASSWORD` = the 16-char app password from step 2
     - `RECIPIENT_EMAIL` = the address you want reports sent to (can be same as `GMAIL_USER`)

4. **Customize** (optional)
   - Edit `config.yaml` to change `alloc_per_stock`, `top_n`, `watchlist_url`, or filter thresholds.

5. **Manual run**
   - You can test locally:
     ```
     python -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     python investor_report.py
     ```
   - Outputs will be in `outputs/` (report HTML + CSV).

6. **GitHub Actions**
   - The workflow is scheduled daily by cron (03:30 IST). You can run it manually via the Actions UI (`workflow_dispatch`).

## Security
- Keep secrets in GitHub repository secrets — do not store credentials in the repo.
- The workflow uses Gmail SMTP and the app password you create (no OAuth flow required here).

## Notes & Limitations
- This is an **investor research tool**, **not financial advice**.
- yfinance provides best-effort financials; some tickers may have missing fields. Always verify before investing.
- If a share price > ₹15,000 the script will show `shares=0`. Consider increasing allocation or choose alternate picks.
