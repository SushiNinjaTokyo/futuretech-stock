#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, datetime
from zoneinfo import ZoneInfo

def et_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 引け後18:00未満は前営業日に倒す
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    # 週末調整（土=5, 日=6 → 直近金曜）
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

def main():
    # もし外から REPORT_DATE が与えられたらそれを採用（手動再実行の柔軟性）
    env = os.getenv("REPORT_DATE")
    if env:
        try:
            d = datetime.date.fromisoformat(env)
            # 週末なら直近の金曜へ
            while d.weekday() >= 5:
                d = d - datetime.timedelta(days=1)
            print(d.isoformat())
            return
        except Exception:
            pass
    print(et_market_date_now().isoformat())

if __name__ == "__main__":
    main()
