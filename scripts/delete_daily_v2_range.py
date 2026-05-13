#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
START_DATE = os.getenv("DAILY_V2_DELETE_START_DATE", os.getenv("START_DATE", "")).strip()
END_DATE = os.getenv("DAILY_V2_DELETE_END_DATE", os.getenv("END_DATE", "")).strip()
CONFIRM = os.getenv("CONFIRM_DELETE", "").strip()
DELETE_DAILY = os.getenv("DELETE_DAILY_V2", "true").strip().lower() == "true"
DELETE_BACKTEST = os.getenv("DELETE_BACKTEST_V2", "false").strip().lower() == "true"


def log(msg: str) -> None:
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} [INFO] {msg}", flush=True)


def dates(start: str, end: str):
    if not start:
        raise SystemExit("start_date is required")
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize() if end else s
    if e < s:
        raise SystemExit("end_date must be >= start_date")
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(s, e, freq="D")]


def update_manifest(remove_dates):
    path = OUT_DIR / "data" / "daily-v2" / "manifest.json"
    if not path.exists():
        return
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    for d in remove_dates:
        data.get("dates", {}).pop(d, None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if CONFIRM != "DELETE":
        raise SystemExit("Refusing to delete. Set confirm to DELETE.")
    remove_dates = dates(START_DATE, END_DATE)
    if DELETE_DAILY:
        for d in remove_dates:
            p = OUT_DIR / "data" / "daily-v2" / d
            if p.exists():
                shutil.rmtree(p)
                log(f"deleted {p}")
        update_manifest(remove_dates)
    if DELETE_BACKTEST:
        for p in [OUT_DIR / "data" / "signals-v2", OUT_DIR / "data" / "backtest-v2"]:
            if p.exists():
                shutil.rmtree(p)
                log(f"deleted {p}")
        html = OUT_DIR / "backtest" / "index.html"
        if html.exists():
            html.unlink()
            log(f"deleted {html}")


if __name__ == "__main__":
    main()
