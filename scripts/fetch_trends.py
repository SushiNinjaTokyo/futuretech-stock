#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Trends 収集（pytrends）
- 環境:
  OUT_DIR=site, UNIVERSE_CSV=data/universe.csv, REPORT_DATE=YYYY-MM-DD
  TRENDS_GEO=US, TRENDS_TIMEFRAME="today 3-m"
  TRENDS_BATCH=5（1リクエストでのキーワード数: 2〜5が安全）
  TRENDS_SLEEP=4.0, TRENDS_JITTER=1.5
  TRENDS_COOLDOWN_BASE=45, TRENDS_COOLDOWN_MAX=180
  TRENDS_BATCH_RETRIES=4
- 出力: site/data/trends/latest.json と site/data/<date>/trends.json
- 失敗時でも空配列を書き出し、後段が必ず動く
"""

import os, csv, json, time, random, math
from typing import List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
from pytrends.request import TrendReq

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "data/universe.csv").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()

GEO = os.environ.get("TRENDS_GEO", "US").strip()
TIMEFRAME = os.environ.get("TRENDS_TIMEFRAME", "today 3-m")
BATCH = max(2, int(float(os.environ.get("TRENDS_BATCH", "5"))))
SLEEP_BASE = float(os.environ.get("TRENDS_SLEEP", "4.0"))
JITTER = float(os.environ.get("TRENDS_JITTER", "1.5"))
COOLDOWN_BASE = int(float(os.environ.get("TRENDS_COOLDOWN_BASE", "45")))
COOLDOWN_MAX  = int(float(os.environ.get("TRENDS_COOLDOWN_MAX", "180")))
RETRIES = int(float(os.environ.get("TRENDS_BATCH_RETRIES", "4")))

DATA_ROOT = os.path.join(OUT_DIR, "data")
DATE_DIR  = os.path.join(DATA_ROOT, REPORT_DATE or "today")
OUT_LATEST = os.path.join(DATA_ROOT, "trends", "latest.json")
OUT_DATE   = os.path.join(DATE_DIR, "trends.json")

def ensure_dirs():
    os.makedirs(os.path.dirname(OUT_LATEST), exist_ok=True)
    os.makedirs(DATE_DIR, exist_ok=True)

def sleep_brief(t=SLEEP_BASE, j=JITTER):
    time.sleep(t + random.random() * j)

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def load_universe(path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "symbol" not in cols:
        if "ticker" in cols:
            df = df.rename(columns={"ticker":"symbol"})
        else:
            raise RuntimeError("universe.csv requires 'symbol'")
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    recs = df[["symbol","name"]].copy()
    recs["symbol"] = recs["symbol"].astype(str).str.upper().str.strip()
    recs["name"]   = recs["name"].astype(str).str.strip()
    return recs.to_dict(orient="records")

def score_from_series(s: pd.Series) -> float:
    """系列(0..100)から"ブレイクアウト"度合いを 0..1 に正規化。
       近30日末値 / 近90%分位 を使う。データが薄い場合は末値/100。
    """
    if s is None or s.empty:
        return 0.0
    tail = s.dropna().iloc[-30:] if len(s) >= 30 else s.dropna()
    if tail.empty:
        return 0.0
    pctl = tail.quantile(0.9) if len(tail) > 5 else 100.0
    pctl = max(1.0, float(pctl))
    last = float(tail.iloc[-1])
    return clamp01(last / pctl)

def fetch_batch(py: TrendReq, kw: List[str]) -> pd.DataFrame:
    for i in range(RETRIES):
        try:
            py.build_payload(kw_list=kw, timeframe=TIMEFRAME, geo=GEO)
            df = py.interest_over_time()
            if not df.empty:
                return df
            # 空でも少し待つ
            sleep_brief()
        except Exception:
            # 429等は指数バックオフ
            cool = min(COOLDOWN_MAX, COOLDOWN_BASE * (2 ** i))
            time.sleep(cool + random.uniform(0, 3))
    return pd.DataFrame()

def main():
    ensure_dirs()
    uni = load_universe(UNIVERSE_CSV)
    # 名前優先で検索精度を上げる（重複回避のためTickerも保持）
    queries = [(x["symbol"], x["name"]) for x in uni]

    py = TrendReq(hl="en-US", tz=0)
    items: List[Dict[str, Any]] = []

    # 5語ずつ（BATCH）で投げる
    buf = []
    meta = []
    for sym, name in queries:
        q = name if len(name) >= 3 else sym
        buf.append(q)
        meta.append((sym, name, q))
        if len(buf) == BATCH:
            df = fetch_batch(py, buf)
            for (sym_, name_, q_) in meta:
                try:
                    series = df[q_] if q_ in df.columns else pd.Series(dtype=float)
                    score = score_from_series(series)
                    items.append({
                        "symbol": sym_,
                        "name": name_,
                        "query": q_,
                        "score_0_1": score,
                        "breakout_0_1": score
                    })
                except Exception:
                    items.append({
                        "symbol": sym_,
                        "name": name_,
                        "query": q_,
                        "score_0_1": 0.0,
                        "breakout_0_1": 0.0
                    })
            buf, meta = [], []
            sleep_brief()

    # 端数
    if buf:
        df = fetch_batch(py, buf)
        for (sym_, name_, q_) in meta:
            try:
                series = df[q_] if q_ in df.columns else pd.Series(dtype=float)
                score = score_from_series(series)
                items.append({
                    "symbol": sym_,
                    "name": name_,
                    "query": q_,
                    "score_0_1": score,
                    "breakout_0_1": score
                })
            except Exception:
                items.append({
                    "symbol": sym_,
                    "name": name_,
                    "query": q_,
                    "score_0_1": 0.0,
                    "breakout_0_1": 0.0
                })

    payload = {
        "date": REPORT_DATE,
        "items": items,
        "meta": {
            "geo": GEO,
            "timeframe": TIMEFRAME,
            "batch": BATCH
        }
    }

    # 常に書き出し
    with open(OUT_LATEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_DATE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[TRENDS] saved: {OUT_LATEST} and {OUT_DATE} (symbols={len(items)})")

if __name__ == "__main__":
    main()
