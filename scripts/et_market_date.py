#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from datetime import datetime, timezone, timedelta

def main() -> None:
    # 1) 昼夜・サマータイム等に左右されにくいようUTC基準
    now_utc = datetime.now(timezone.utc)

    # 2) 既に REPORT_DATE が来ていればそれを尊重
    env_date = os.getenv("REPORT_DATE", "").strip()
    if env_date:
        try:
            # YYYY-MM-DD ならそのまま
            _ = datetime.strptime(env_date, "%Y-%m-%d")
            print(env_date)
            return
        except Exception:
            # 無効な値なら無視して続行
            pass

    # 3) 米株想定: 取引終了後に「当日」を締める。UTCで翌日 AM5:00 相当で日繰り上げ。
    # （NY - UTC は概ね 4〜5 時差なので AM5:00 でほぼ確実にクローズ後）
    cutoff_hour = 5
    report_date = (now_utc - timedelta(hours=cutoff_hour)).date()
    print(report_date.isoformat())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 何があっても今日の日付を返す（CIを止めない）
        print(datetime.utcnow().date().isoformat())
        sys.exit(0)
