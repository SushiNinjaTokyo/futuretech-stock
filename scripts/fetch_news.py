#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSS/Atom から各シンボルの最近記事件数（lookback 日数）を集計し 0..1 に正規化。
結果: site/data/news/latest.json, site/data/{DATE}/news.json
"""
from __future__ import annotations
import os, json, time, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import feedparser

logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO"),
    format="%(asctime)sZ [%(levelname)s] [NEWS] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("news")

OUT_DIR   = Path(os.getenv("OUT_DIR","site"))
UNIVERSE  = Path(os.getenv("UNIVERSE_CSV","data/universe.csv"))
REPORT    = os.getenv("REPORT_DATE") or pd.Timestamp.utcnow().date().isoformat()
LOOKBACK  = int(os.getenv("NEWS_LOOKBACK_DAYS","7"))
MAX_PER   = int(os.getenv("NEWS_MAX_PER_SYMBOL","20"))
SLEEP_SEC = float(os.getenv("NEWS_SLEEP_SEC","1.0"))

SOURCES = [
    "https://finance.yahoo.com/rss/topstories",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.reuters.com/finance/us/technologyNews",
]

def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE)
    if "news_keyword" not in df.columns:
        df["news_keyword"] = df.get("name", df["symbol"])
    return df[["symbol","news_keyword"]]

def fetch_all() -> list[dict]:
    results = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK)
    for url in SOURCES:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            log.warning("parse failed: %s (%s)", url, e)
            continue
        for e in feed.entries[:1000]:
            dt_pub = None
            for key in ("published_parsed","updated_parsed","created_parsed"):
                val = getattr(e, key, None) or e.get(key)
                if val:
                    dt_pub = datetime(*val[:6], tzinfo=timezone.utc)
                    break
            if dt_pub and dt_pub < cutoff: 
                continue
            title = e.get("title","")
            summary = e.get("summary","")
            results.append({"title": title, "summary": summary})
        time.sleep(SLEEP_SEC)
    return results

def count_by_symbol(news_rows: list[dict], u: pd.DataFrame) -> list[dict]:
    items = []
    for _, r in u.iterrows():
        sym = r["symbol"]
        kw  = r["news_keyword"]
        cnt = 0
        key = str(kw).lower()
        for n in news_rows:
            text = (n["title"] + " " + n["summary"]).lower()
            if key in text or str(sym).lower() in text:
                cnt += 1
                if cnt >= MAX_PER:
                    break
        items.append({"symbol": sym, "recent_count": cnt})
    return items

def to_percentile(values: list[int]) -> dict[str,float]:
    s = pd.Series(values, dtype="float")
    if len(s)==0 or s.max()==0: 
        return {}
    pct = s.rank(pct=True, method="average")
    return {i: float(round(v,6)) for i, v in enumerate(pct.tolist())}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_dir = OUT_DIR / "data" / REPORT
    (OUT_DIR/"data"/"news").mkdir(parents=True, exist_ok=True)
    date_dir.mkdir(parents=True, exist_ok=True)

    u = load_universe()
    news_rows = fetch_all()
    items = count_by_symbol(news_rows, u)

    # 0..1 正規化（百分位）
    pmap = to_percentile([x["recent_count"] for x in items])
    for i, it in enumerate(items):
        it["score_0_1"] = float(pmap.get(i, 0.0))

    payload = {"date": REPORT, "items": items, "meta": {"lookback_days": LOOKBACK}}
    with open(OUT_DIR/"data"/"news"/"latest.json","w") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(date_dir/"news.json","w") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("saved: %s and %s (symbols=%d)", OUT_DIR/"data/news/latest.json", date_dir/"news.json", len(items))

if __name__=="__main__":
    main()
