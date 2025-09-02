#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
米国市場（NYSE/Nasdaq）向けのレポート日付決定。
- 規定: "前営業日"。米東部時刻の16:15より前は前営業日、以降は当日を採用。
- 休日は簡易処理（週末のみ）。必要ならUSマーケット休日リストを追加可能。
- 環境変数 REPORT_DATE があればそれをそのまま出力（上書き用途）。
"""

import os
from datetime import datetime, timedelta, timezone

def is_weekend(dt):
    # Monday=0 ... Sunday=6
    return dt.weekday() >= 5

def prev_weekday(dt):
    d = dt
    while is_weekend(d):
        d -= timedelta(days=1)
    # 平日でも土日跨ぎ直後に対応
    if d.weekday() == 0 and (dt - timedelta(days=3)).weekday() == 4:
        return d
    return d

def main():
    if os.environ.get("REPORT_DATE"):
        print(os.environ["REPORT_DATE"])
        return

    # 現在UTC -> US/Eastern (固定オフセット近似: DSTはざっくり)
    # GitHub Actionsでは厳密TZ不要。厳密が必要なら pytz / zoneinfo を利用。
    now_utc = datetime.now(timezone.utc)
    # US/Eastern: 標準時-5, 夏時間-4。簡易に-4固定（大半の稼働期間が夏時間）
    eastern = now_utc.astimezone(timezone(timedelta(hours=-4)))
    today = eastern.date()

    # 16:15より前なら実質前日クローズを採用
    market_close_buffer = eastern.replace(hour=16, minute=15, second=0, microsecond=0)
    if eastern < market_close_buffer:
        target = eastern.date() - timedelta(days=1)
    else:
        target = eastern.date()

    # 週末回避
    dt = datetime(target.year, target.month, target.day)
    while is_weekend(dt):
        dt -= timedelta(days=1)

    print(dt.strftime("%Y-%m-%d"))

if __name__ == "__main__":
    main()
