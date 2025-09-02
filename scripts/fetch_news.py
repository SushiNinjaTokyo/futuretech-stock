#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News/RSS（Google News 検索RSS）
- 環境:
  OUT_DIR=site, UNIVERSE_CSV=data/universe.csv, REPORT_DATE=YYYY-MM-DD
  NEWS_LOOKBACK_DAYS=7, NEWS_MAX_PER_SYMBOL=20, NEWS_SLEEP_SEC=1.0
- 出力形式は items の配列で、各要素は:
  { symbol, name, recent_count, score_0_1 }
- スコア: min(1, recent_count / NEWS_MAX_PER_SYMBOL)
"""

import os, json, time, urllib.parse, math, random
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

import pandas as pd
import feedparser

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "data/universe.csv").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()

LOOKBACK_DAYS = int(float(os.environ.get("NEWS_LOOKBACK_DAYS", "7")))
MAX_PER_SYMBOL = int(float(os.environ.get("NEWS_MAX_PER_SYMBOL", "20")))
SLEEP_SEC = float(os.environ.get("NEWS_SLEEP_SEC", "1.0"))

DATA_ROOT = os.path.join(OUT_DIR, "data")
DATE_DIR  = os.path.join(DATA_ROOT, REPORT_DATE or "today")
OUT_LATEST = os.path.join(DATA_ROOT, "news", "latest.json")
OUT_DATE   = os.path.join(DATE_DIR, "news.json")

def ensure_dirs():
    os.makedirs(os.path.dirname(OUT_LATEST), exist_ok=True)
    os.makedirs(DATE_DIR, exist_ok=True)

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

def news_url(q: str) -> str:
    base = "https://news.google.com/rss/search"
    params = {
        "q": q,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en"
    }
    return f"{base}?{urllib.parse.urlencode(params)}"

def within_days(published_parsed, days: int) -> bool:
    if not published_parsed:
        return False
    dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt) <= timedelta(days=days)

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def main():
    ensure_dirs()
    uni = load_universe(UNIVERSE_CSV)
    items: List[Dict[str, Any]] = []

    for rec in uni:
        sym = rec["symbol"]
        name = rec["name"]

        # シンプル検索クエリ（AND/ORのバランス重視）
        q = f'"{name}" OR {sym}'
        url = news_url(q)
        try:
            feed = feedparser.parse(url)
            cnt = 0
            for e in (feed.entries or []):
                if within_days(getattr(e, "published_parsed", None), LOOKBACK_DAYS):
                    cnt += 1
                    if cnt >= MAX_PER_SYMBOL:
                        break
            score = clamp01(cnt / MAX_PER_SYMBOL)
            items.append({
                "symbol": sym,
                "name": name,
                "query": q,
                "recent_count": int(cnt),
                "score_0_1": score,
            })
        except Exception:
            items.append({
                "symbol": sym,
                "name": name,
                "query": q,
                "recent_count": 0,
                "score_0_1": 0.0,
            })
        time.sleep(SLEEP_SEC)

    payload = {"date": REPORT_DATE, "items": items}

    with open(OUT_LATEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_DATE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[NEWS] saved: {OUT_LATEST} and {OUT_DATE} (symbols={len(items)})")

if __name__ == "__main__":
    main()
