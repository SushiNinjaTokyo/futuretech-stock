#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DII（ダークプール/内部者モメンタム）プロキシ。
外部 API 依存を避け、ここでは「出来高急増（最近4週 vs 直近12週）」を 0..1 に正規化して代用。
結果: site/data/dii/latest.json, site/data/{DATE}/dii.json
"""
from __future__ import annotations
import os, json, logging
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO"),
    format="%(asctime)sZ [%(levelname)s] [DII] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dii")

OUT_DIR   = Path(os.getenv("OUT_DIR","site"))
UNIVERSE  = Path(os.getenv("UNIVERSE_CSV","data/universe.csv"))
REPORT    = os.getenv("REPORT_DATE") or pd.Timestamp.utcnow().date().isoformat()
LOOKBACK  = int(os.getenv("DII_LOOKBACK_WEEKS","12"))
RECENT    = int(os.getenv("DII_RECENT_WEEKS","4"))

def load_universe() -> list[str]:
    df = pd.read_csv(UNIVERSE)
    return df["symbol"].dropna().astype(str).tolist()

def dii_proxy(symbols: list[str]) -> list[dict]:
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(weeks=LOOKBACK+RECENT+2)
    rows = []
    for sym in symbols:
        try:
            hist = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False, group_by="ticker")
            vol = hist["Volume"].dropna()
            if vol.empty:
                score = 0.0
            else:
                recent_w = RECENT*5
                look_w   = LOOKBACK*5
                r = vol.tail(recent_w).mean()
                l = vol.tail(look_w).mean() if len(vol)>=look_w else vol.mean()
                ratio = (r / l) if l>0 else 0.0
                score = max(0.0, min(1.0, (ratio - 0.8)/1.6))
            rows.append({"symbol": sym, "score_0_1": float(round(score,6)), "components": {"volume": float(round(score,6))}})
        except Exception as e:
            log.warning("failed %s: %s", sym, e)
            rows.append({"symbol": sym, "score_0_1": 0.0, "components": {"volume": 0.0}})
    return rows

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_dir = OUT_DIR / "data" / REPORT
    (OUT_DIR/"data"/"dii").mkdir(parents=True, exist_ok=True)
    date_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_universe()
    items = dii_proxy(symbols)
    payload = {"date": REPORT, "items": items, "meta": {"source_used": "fast_volume", "lookback_weeks": LOOKBACK, "recent_weeks": RECENT}}
    with open(OUT_DIR/"data"/"dii"/"latest.json","w") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(date_dir/"dii.json","w") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("saved: %s and %s (symbols=%d) source=%s", OUT_DIR/"data/dii/latest.json", date_dir/"dii.json", len(items), payload["meta"]["source_used"])

if __name__=="__main__":
    main()
