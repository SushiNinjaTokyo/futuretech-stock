#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta
import pytz

# 米東部時間で「営業日基準のレポート日」を決める
# 22:10 UTC(=ET 18:10)起動を想定
def resolve_et_market_date(now_utc=None):
    utc = pytz.utc
    et = pytz.timezone("US/Eastern")
    now = now_utc or datetime.now(tz=utc)
    now_et = now.astimezone(et)

    d = now_et.date()
    # 土日は直近の金曜日へ
    if d.weekday() == 5:  # Sat
        d = d - timedelta(days=1)
    elif d.weekday() == 6:  # Sun
        d = d - timedelta(days=2)
    # 祝日判定など必要ならここに追加（簡略化）

    return d.isoformat()

if __name__ == "__main__":
    print(resolve_et_market_date())
