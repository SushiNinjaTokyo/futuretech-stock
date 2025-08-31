#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

def prev_us_business_day(d: date) -> date:
    while d.weekday() >= 5:  # 5,6 = Sat, Sun
        d -= timedelta(days=1)
    return d

def et_market_date_now() -> date:
    now = datetime.now(ZoneInfo("America/New_York"))
    d = now.date()
    # 報告は引け後前提。18:00ETまでは前営業日に寄せる
    if now.hour < 18:
        d -= timedelta(days=1)
    return prev_us_business_day(d)

if __name__ == "__main__":
    print(et_market_date_now().isoformat())
