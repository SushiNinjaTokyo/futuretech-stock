#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Trends breakout scorer (0..1)
- name を優先、無ければ symbol で検索
- 3件ずつの小バッチ、間隔スリープ＆クールダウン
- 指数化: 最終値 / 直近30日の中央値 を 0..1 へクリップ
- 出力: site/data/trends/latest.json と site/data/<DATE>/trends.json
"""

import os, sys, json, time, random, datetime, pathlib
import pandas as pd
from pytrends.request import TrendReq

OUT_DIR      = os.getenv("OUT_DIR","site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV","data/universe.csv")
DATE         = os.getenv("REPORT_DATE") or datetime.date.today().isoformat()

GEO       = os.getenv("TRENDS_GEO","US")
TIMEFRAME = os.getenv("TRENDS_TIMEFRAME","today 3-m")
BATCH     = int(os.getenv("TRENDS_BATCH","3"))
SLEEP     = float(os.getenv("TRENDS_SLEEP","4.0"))
JITTER    = float(os.getenv("TRENDS_JITTER","1.5"))
CD_BASE   = int(os.getenv("TRENDS_COOLDOWN_BASE","45"))
CD_MAX    = int(os.getenv("TRENDS_COOLDOWN_MAX","180"))
RETRIES   = int(os.getenv("TRENDS_BATCH_RETRIES","4"))

def load_universe(path):
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["name"] = (df["name"].astype(str) if "name" in df.columns else df["symbol"])
    return df[["symbol","name"]]

def fetch_batch(pytrends, terms):
    for attempt in range(1, RETRIES+1):
        try:
            pytrends.build_payload(terms, timeframe=TIMEFRAME, geo=GEO)
            df = pytrends.interest_over_time()
            if df is not None and not df.empty:
                return df
            print(f"[WARN] Trends returned empty frame for {terms} (attempt {attempt})")
        except Exception as e:
            print(f"[WARN] Trends error for {terms} (attempt {attempt}): {e}")
        time.sleep(min(CD_MAX, CD_BASE + int(random.random()*CD_BASE)))
    print(f"[ERROR] Trends batch failed after retries: {terms} :: None")
    return pd.DataFrame()

def score_series(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 10: return 0.0
    last = float(s.iloc[-1])
    med30 = float(s.tail(30).median()) if len(s) >= 30 else float(s.median())
    if med30 <= 0: return 0.0
    r = last / med30
    # 1.0 を超える分をやや控えめに（3.0以上はサチる）
    if r <= 0.2: sc = 0.0
    elif r >= 3.0: sc = 1.0
    else: sc = (r - 0.2) / (3.0 - 0.2)
    return max(0.0, min(1.0, sc))

def main():
    uni = load_universe(UNIVERSE_CSV)
    terms = uni["name"].tolist()
    syms  = uni["symbol"].tolist()

    pytrends = TrendReq(hl="en-US", tz=360)
    items = {}
    for i in range(0, len(terms), BATCH):
        chunk = terms[i:i+BATCH]
        print(f"[TRENDS] fetching: {chunk}")
        df = fetch_batch(pytrends, chunk)
        # sleep between batches
        time.sleep(SLEEP + random.random()*JITTER)
        if df is None or df.empty:
            continue
        for t in chunk:
            if t in df.columns:
                sc = score_series(df[t])
                # t -> symbol の対応を解決
                for j in range(i, min(i+BATCH, len(terms))):
                    if terms[j] == t:
                        items[syms[j]] = {"score_0_1": sc}

    payload = {"items": items, "date": DATE, "geo": GEO, "timeframe": TIMEFRAME}
    out_latest = pathlib.Path(OUT_DIR)/"data"/"trends"/"latest.json"
    out_daily  = pathlib.Path(OUT_DIR)/"data"/DATE/"trends.json"
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, indent=2))
    out_daily.write_text(json.dumps(payload, indent=2))
    print(f"[TRENDS] saved: {out_latest} and {out_daily} ({len(items)} symbols)")

if __name__ == "__main__":
    main()
