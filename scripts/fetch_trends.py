#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE")
TRENDS_GEO = os.getenv("TRENDS_GEO", "US")
TRENDS_TIMEFRAME = os.getenv("TRENDS_TIMEFRAME", "today 3-m")
TRENDS_SLEEP = float(os.getenv("TRENDS_SLEEP", "4.0"))


def log(level: str, msg: str) -> None:
    from datetime import datetime, timezone
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_universe() -> List[Dict[str, str]]:
    if not UNIVERSE_CSV.exists():
        return []
    df = pd.read_csv(UNIVERSE_CSV)
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol", list(df.columns)[0])
    name_col = cols.get("name")
    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip().upper()
        if not sym:
            continue
        nm = str(row.get(name_col, "")).strip() if name_col else ""
        out.append({"symbol": sym, "name": nm})
    return out


def calc_breakout_score(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) < 8:
        return 0.0
    last = float(s.iloc[-1])
    ref = s.iloc[:-1]
    if len(ref) == 0:
        return 0.0
    pct_rank = float((ref <= last).mean())
    mean = float(ref.mean())
    std = float(ref.std(ddof=0) or 0.0)
    z = 0.0 if std == 0 else (last - mean) / std
    z_sig = 1.0 / (1.0 + math.exp(-z))
    score = 0.7 * pct_rank + 0.3 * z_sig
    return max(0.0, min(1.0, score))


def fetch_trends_series(keyword: str) -> pd.Series:
    try:
        from pytrends.request import TrendReq  # type: ignore
    except Exception:
        return pd.Series(dtype=float)

    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload([keyword], timeframe=TRENDS_TIMEFRAME, geo=TRENDS_GEO)
        df = pytrends.interest_over_time()
        if keyword not in df.columns:
            return pd.Series(dtype=float)
        return df[keyword]
    except Exception:
        return pd.Series(dtype=float)


def main() -> None:
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is required")

    universe = load_universe()
    items: List[Dict[str, Any]] = []

    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")
        series = fetch_trends_series(sym)
        score = calc_breakout_score(series) if not series.empty else 0.0

        items.append({
            "symbol": sym,
            "name": nm,
            "score_0_1": round(score, 6),
            "points_observed": int(len(series)),
            "last_value": float(series.iloc[-1]) if not series.empty else None,
        })

        time.sleep(TRENDS_SLEEP)

    payload = {"date": REPORT_DATE, "items": items}
    day_path = OUT_DIR / "data" / REPORT_DATE / "trends.json"
    latest_path = OUT_DIR / "data" / "trends" / "latest.json"
    write_json(day_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote Trends: {day_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
