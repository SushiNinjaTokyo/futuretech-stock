#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def env_i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def is_market_day(d: date) -> bool:
    try:
        import pandas_market_calendars as mcal  # type: ignore
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=d.isoformat(), end_date=d.isoformat())
        return not sched.empty
    except Exception:
        return d.weekday() < 5  # fallback: Mon-Fri


def previous_market_day(d: date) -> date:
    cur = d
    while not is_market_day(cur):
        cur -= timedelta(days=1)
    return cur


def resolve_report_date(now_et: datetime, cutoff_hour_et: int) -> date:
    base = now_et.date()
    if now_et.hour < cutoff_hour_et:
        base = base - timedelta(days=1)
    return previous_market_day(base)


def main() -> None:
    cutoff = env_i("MARKET_CUTOFF_HOUR_ET", 20)
    now_et = datetime.now(ET)
    report_date = resolve_report_date(now_et, cutoff)
    print(report_date.isoformat())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # CIを止めない最終フォールバック
        print(datetime.now(ET).date().isoformat())
        sys.exit(0)
