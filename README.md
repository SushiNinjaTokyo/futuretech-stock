
# FutureTech Markets — Starter v2 (MOCK + Hard Budget Cap)

Run fully remote with **no local dev**. Works even **without API keys** (MOCK_MODE=true).
Enforces a **hard ¥10,000 monthly cap** — if exceeded, the job skips execution.

## Quick Start
1. Create a GitHub repo and upload these files.
2. GitHub → Settings → Secrets and variables → Actions → Variables:
   - `MOCK_MODE` = `true`
   - `BUDGET_JPY_MAX` = `10000`
   - `MANUAL_DAILY_COST_JPY` = `0`
3. Deploy on Vercel (Framework: Other, Output dir: `site`).
4. Actions → daily-report → Run workflow.

## Switch to real data (later)
- Add `TIINGO_TOKEN` (Secrets), set `MOCK_MODE=false`, run again.

## Budget
- Monthly spend tracked in `site/data/spend.json` and stops before exceeding the cap.
