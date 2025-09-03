#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US 市場の「前営業日」をレポート日として返す。
環境変数 REPORT_DATE があればそれを優先（YYYY-MM-DD）。
"""
from __future__ import annotations
import os, sys, datetime as dt, logging

TZ = dt.timezone.utc

def us_prev_business_day(ref: dt.date|None=None) -> dt.date:
    if ref is None:
        ref = dt.datetime.now(tz=TZ).date()
    d = ref
    # 米国の簡易休日: 土日。必要なら NYSE 休場表を入れる。
    if d.weekday() >= 5:
        # 土: -1, 日: -2
        d = d - dt.timedelta(days=d.weekday() - 4)
    # 当日が平日でも「市場クローズ後に回る」前提で前日を使う
    d = d - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d

def main():
    rd_env = os.getenv("REPORT_DATE", "").strip()
    if rd_env:
        print(rd_env)
        return
    print(us_prev_business_day().isoformat())

if __name__ == "__main__":
    main()
