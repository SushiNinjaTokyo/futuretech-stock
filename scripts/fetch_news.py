#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News score (0..1) via Google News RSS + VADER sentiment (last N days)
- 1銘柄につき最大 N 記事、過去 LOOKBACK 日
- スコア: 正/負の平均 compound を 0..1 に正規化（記事数が少ないと控えめ）
- 出力: site/data/news/latest.json と site/data/<DATE>/news.json
"""

import os, sys, json, time, datetime, pathlib, urllib.parse
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT_DIR      = os.getenv("OUT_DIR","site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV","data/universe.csv")
DATE         = os.getenv("REPORT_DATE") or datetime.date.today().isoformat()

LOOKBACK_DAYS      = int(os.getenv("NEWS_LOOKBACK_DAYS","7"))
MAX_PER_SYMBOL     = int(os.getenv("NEWS_MAX_PER_SYMBOL","20"))
SLEEP_SEC          = float(os.getenv("NEWS_SLEEP_SEC","1.0"))

def load_universe(path):
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["name"] = (df["name"].astype(str) if "name" in df.columns else df["symbol"])
    return df[["symbol","name"]]

def google_news_rss(q: str) -> str:
    # “会社名 stock” で検索
    qs = urllib.parse.quote_plus(f'{q} stock')
    return f"https://news.google.com/rss/search?q={qs}&hl=en-US&gl=US&ceid=US:en"

def main():
    uni = load_universe(UNIVERSE_CSV)
    sid = SentimentIntensityAnalyzer()
    since = datetime.datetime.utcnow() - datetime.timedelta(days=LOOKBACK_DAYS)

    items = {}
    for sym, name in uni.values:
        url = google_news_rss(name)
        feed = feedparser.parse(url)
        recs = []
        for e in feed.entries[:MAX_PER_SYMBOL]:
            # PubDate がないケースもあるのでスキップしすぎない
            published = None
            if "published_parsed" in e and e.published_parsed:
                published = datetime.datetime(*e.published_parsed[:6])
            if published and published < since:
                continue
            title = e.title if "title" in e else ""
            s = sid.polarity_scores(title or "")
            recs.append(s["compound"])
        recent_count = len(recs)
        if recent_count == 0:
            score = 0.0
        else:
            avg = sum(recs)/recent_count
            # compound(-1..1) → (0..1) へ変換、件数が少ないと控えめに
            score = max(0.0, min(1.0, (avg+1)/2)) * min(1.0, recent_count/10.0)
        items[sym] = {"score_0_1": score, "recent_count": recent_count}
        time.sleep(SLEEP_SEC)

    payload = {"items": items, "date": DATE, "lookback_days": LOOKBACK_DAYS}
    out_latest = pathlib.Path(OUT_DIR)/"data"/"news"/"latest.json"
    out_daily  = pathlib.Path(OUT_DIR)/"data"/DATE/"news.json"
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, indent=2))
    out_daily.write_text(json.dumps(payload, indent=2))
    print(f"[NEWS] saved: {out_latest} and {out_daily} ({len(items)} symbols)")

if __name__ == "__main__":
    main()
