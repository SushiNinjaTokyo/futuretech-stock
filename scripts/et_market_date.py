#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ET market date resolver
- New York時間(ET)でのカットオフを基準に "レポート日" を決める。
- 既定カットオフ: 20:00 ET（それ以前は前営業日、それ以降は当日）
- WEEKEND(⼟⽇) は直近の平⽇(通常は⾦曜)にロールバック。
- FORCE_REPORT_DATE があればそれを最優先（YYYY-MM-DD）。
"""

import os
import sys
import datetime
from zoneinfo import ZoneInfo

def prev_business_day(d: datetime.date) -> datetime.date:
    # ⼟⽇なら直近の平⽇へ。平⽇ならその⽇のまま返す。
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= datetime.timedelta(days=1)
    return d

def main():
    # 1) 明示指定があれば最優先
    force = os.getenv("FORCE_REPORT_DATE", "").strip()
    if force:
        try:
            dt = datetime.date.fromisoformat(force)
            print(dt.isoformat())
            return
        except Exception:
            print(force)  # 形式不正でもとりあえず出す（ワークフローで気づける）
            return

    # 2) ET現在時刻
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    cutoff_h = int(os.getenv("MARKET_CUTOFF_HOUR_ET", "20"))  # 既定20:00 ET
    base = now_et.date()

    # 3) カットオフ前は「前営業日」、カットオフ以後は「当日」
    if now_et.hour < cutoff_h:
        base = base - datetime.timedelta(days=1)

    # 4) 週末補正（⼟⽇は直近平⽇へ）
    base = prev_business_day(base)

    print(base.isoformat())

if __name__ == "__main__":
    main()
