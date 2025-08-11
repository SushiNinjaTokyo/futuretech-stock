#!/usr/bin/env python3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def usa_market_date_now():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 引け後18:00(ET)より前なら前営業日に倒す
    if now_et.hour < 18:
        d = d - timedelta(days=1)
    # 週末は直近の金曜へ
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d = d - timedelta(days=1)
    return d

if __name__ == "__main__":
    print(usa_market_date_now().isoformat())
