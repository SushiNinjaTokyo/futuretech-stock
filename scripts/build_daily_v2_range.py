#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
START_DATE = os.getenv("DAILY_V2_START_DATE", os.getenv("START_DATE", "")).strip()
END_DATE = os.getenv("DAILY_V2_END_DATE", os.getenv("END_DATE", "")).strip()
MODE = os.getenv("DAILY_V2_RANGE_MODE", "missing_only").strip().lower()
MAX_DAYS = int(os.getenv("DAILY_V2_MAX_DAYS_PER_RUN", "10") or "10")
SLEEP = float(os.getenv("DAILY_V2_RANGE_SLEEP_SECONDS", "5") or "5")


def log(msg: str) -> None:
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} [INFO] {msg}", flush=True)


def trading_dates(start: str, end: str) -> List[str]:
    try:
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize() if end else pd.Timestamp.utcnow().normalize()
    except Exception as exc:
        raise SystemExit(f"Invalid date input: {exc}")
    if e < s:
        raise SystemExit(f"end_date must be >= start_date. start={s.date()}, end={e.date()}")
    # Weekday filter only. US holidays are harmless; yfinance will use prior/available rows and output still records the requested date.
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(s, e, freq="B")]


def main() -> None:
    if not START_DATE:
        raise SystemExit("DAILY_V2_START_DATE is required")
    dates = trading_dates(START_DATE, END_DATE)
    if MAX_DAYS > 0:
        dates = dates[:MAX_DAYS]
    log(f"Daily v2 range build dates={dates}, mode={MODE}, max_days={MAX_DAYS}")
    for i, d in enumerate(dates, 1):
        out = OUT_DIR / "data" / "daily-v2" / d / "top10.json"
        if MODE == "missing_only" and out.exists():
            log(f"skip existing {d}")
            continue
        env = os.environ.copy()
        env["REPORT_DATE"] = d
        log(f"build {i}/{len(dates)} {d}")
        subprocess.run([sys.executable, "scripts/build_daily_v2.py"], cwd=str(ROOT), env=env, check=True)
        if SLEEP > 0 and i < len(dates):
            time.sleep(SLEEP)


if __name__ == "__main__":
    main()
