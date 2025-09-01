#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US/Eastern market-date resolver.
Rules:
- If REPORT_DATE env is set (YYYY-MM-DD), return it.
- Otherwise, use the last available trading day from SPY daily data.
  This makes the site independent from the action runtime:
    * If executed after US market close on a trading day → returns today.
    * If executed during market hours, weekends, or holidays → returns the
      previous trading day.
- Fallbacks: yfinance → Stooq CSV → simple weekday rollback.
"""
from __future__ import annotations

import os
import sys
import datetime as _dt
from zoneinfo import ZoneInfo

def _today_et() -> _dt.date:
    return _dt.datetime.now(ZoneInfo("America/New_York")).date()

def _prev_us_weekday(d: _dt.date) -> _dt.date:
    while d.weekday() >= 5:
        d -= _dt.timedelta(days=1)
    return d

def _spy_last_trading_date_via_yf() -> _dt.date | None:
    try:
        import yfinance as yf
        # 7 days is enough to cover weekends/holidays
        df = yf.Ticker("SPY").history(period="7d", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
        idx = df.index
        if hasattr(idx, "tz_localize"):
            last = idx[-1].tz_convert("America/New_York").date()
        else:
            last = idx[-1].date()
        return last
    except Exception:
        return None

def _spy_last_trading_date_via_stooq() -> _dt.date | None:
    try:
        import requests, io, csv
        url = "https://stooq.com/q/d/l/?s=spy.us&i=d"
        r = requests.get(url, timeout=20)
        if not r.ok or "Date,Open,High,Low,Close,Volume" not in r.text:
            return None
        # Read last non-empty data row
        lines = [line for line in r.text.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        last_date = lines[-1].split(",")[0]
        return _dt.date.fromisoformat(last_date)
    except Exception:
        return None

def get_effective_market_date() -> _dt.date:
    # 1) Manual override
    forced = os.getenv("REPORT_DATE")
    if forced:
        try:
            return _dt.date.fromisoformat(forced)
        except Exception:
            pass

    # 2) Try SPY via yfinance
    d = _spy_last_trading_date_via_yf()
    if d:
        return d

    # 3) Fallback to Stooq
    d = _spy_last_trading_date_via_stooq()
    if d:
        return d

    # 4) Ultimate fallback: previous US weekday (approx)
    return _prev_us_weekday(_today_et())

def main():
    print(get_effective_market_date().isoformat())

if __name__ == "__main__":
    main()
