#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News fetcher (robust & idempotent)
- Google News RSS で記事取得（会社名 / "SYMBOL stock"）
- 直近 RECENT_DAYS の記事 × 簡易感情で素点
- 宇宙内パーセンタイルで 0..1 正規化（score_0_1）
- latest.json と 日付別 news.json に保存（何度実行してもOK）
"""

import os, time, json, pathlib, math, random, datetime, re
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

import requests
import feedparser
import pandas as pd

# ---------- Config ----------
OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")

DATE = os.getenv("REPORT_DATE")
if not DATE:
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 20:00 ET までは前営業日扱い（安全寄り）
    if now_et.hour < 20:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    DATE = d.isoformat()

UA = os.getenv("NEWS_USER_AGENT", "futuretech-stock/1.0 (news-fetcher)")
SLEEP = float(os.getenv("NEWS_SLEEP", "1.0"))
JITTER = float(os.getenv("NEWS_JITTER", "0.7"))

RECENT_DAYS = int(os.getenv("NEWS_RECENT_DAYS", "5"))
LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "30"))
MAX_PER_SYMBOL = int(os.getenv("NEWS_MAX_PER_SYMBOL", "80"))
RETRIES = int(os.getenv("NEWS_RETRIES", "3"))
TIMEOUT = int(os.getenv("NEWS_TIMEOUT", "25"))

# ---------- Utils ----------
def jitter_sleep(base: float = SLEEP, jitter: float = JITTER):
    time.sleep(base + random.uniform(0, jitter))

def req_get(url: str, params: dict | None = None) -> requests.Response:
    last = None
    for i in range(1, RETRIES+1):
        try:
            r = requests.get(
                url, params=params,
                headers={"User-Agent": UA},
                timeout=TIMEOUT,
            )
            if r.status_code == 429:
                wait = min(60, 5 * (2 ** (i-1))) + random.uniform(0, 1.5)
                print(f"[NEWS] 429 cooldown {wait:.1f}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            wait = 1.2 * i + random.uniform(0, 0.8)
            print(f"[NEWS] retry {i}/{RETRIES} after error: {e} (sleep {wait:.1f}s)")
            time.sleep(wait)
    raise last

def to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, bytes)):
        return x.decode() if isinstance(x, bytes) else x
    try:
        if hasattr(x, "get"):
            for k in ("$t", "name", "label", "text"):
                v = x.get(k)
                if isinstance(v, (str, bytes)):
                    return to_str(v)
        return str(x)
    except Exception:
        return ""

def clean_text(s: Any) -> str:
    s = to_str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_time(e: Any) -> datetime.datetime:
    now = datetime.datetime.now(datetime.timezone.utc)
    for k in ("published_parsed", "updated_parsed"):
        t = getattr(e, k, None) or (e.get(k) if hasattr(e, "get") else None)
        if t:
            try:
                return datetime.datetime(*t[:6], tzinfo=datetime.timezone.utc)
            except Exception:
                pass
    for k in ("published", "updated", "dc_date"):
        s = e.get(k) if hasattr(e, "get") else None
        if s:
            try:
                pt = feedparser._parse_date(s)
                if pt:
                    return datetime.datetime(*pt[:6], tzinfo=datetime.timezone.utc)
            except Exception:
                pass
    return now

def pct_rank(vals: List[float], x: float | None) -> float:
    arr = sorted([v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)])
    if not arr or x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    import bisect
    k = bisect.bisect_right(arr, x)
    return k / len(arr)

POS = set("""
beat beats tops surges jumps soars record growth bullish win wins winning upbeat upgrade upgrades
""".split())
NEG = set("""
miss misses falls plunges drops sinks cuts cut slumps bearish lawsuit probe recall downgrade downgrades
""".split())

def tiny_sentiment(headline: str) -> float:
    h = headline.lower()
    pos = sum(1 for w in POS if f" {w}" in f" {h}")
    neg = sum(1 for w in NEG if f" {w}" in f" {h}")
    if pos == 0 and neg == 0:
        return 1.0
    return max(0.2, (1.0 + 0.15 * pos - 0.15 * neg))

# ---------- Core ----------
def build_queries(symbol: str, name: str) -> List[str]:
    q1 = f'"{name}"' if name else ""
    q2 = f'"{symbol} stock"'
    return [q for q in (q1, q2) if q.strip()]

def fetch_rss_for_query(query: str, hl="en-US", gl="US") -> feedparser.FeedParserDict:
    base = "https://news.google.com/rss/search"
    params = {"q": query, "hl": hl, "gl": gl, "ceid": "US:en"}
    r = req_get(base, params=params)
    jitter_sleep()
    return feedparser.parse(r.text)

def main():
    uni = pd.read_csv(UNIVERSE_CSV)
    uni["symbol"] = uni["symbol"].astype(str).str.upper().str.strip()
    if "name" not in uni.columns:
        uni["name"] = ""
    uni["name"] = uni["name"].fillna("").astype(str)

    asof = DATE
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    recent_since = now_utc - datetime.timedelta(days=RECENT_DAYS)
    lookback_since = now_utc - datetime.timedelta(days=LOOKBACK_DAYS)

    per_symbol: Dict[str, Dict[str, Any]] = {}

    for _, row in uni.iterrows():
        sym = row["symbol"]
        name = row["name"]
        queries = build_queries(sym, name)

        seen_urls = set()
        articles: List[Dict[str, Any]] = []

        for q in queries:
            try:
                feed = fetch_rss_for_query(q)
            except Exception as e:
                print(f"[NEWS] fetch failed {sym} / {q}: {e}")
                continue

            for entry in getattr(feed, "entries", []):
                get = entry.get
                url = clean_text(get("link"))
                if not url or url in seen_urls:
                    continue

                ts = parse_time(entry)
                if ts < lookback_since:
                    continue

                title = clean_text(get("title"))
                summary = clean_text(get("summary") or get("description"))
                source = clean_text(get("source") or get("author"))

                articles.append({
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "source": source or "Google News",
                    "ts": ts.isoformat(),
                    "ts_epoch": int(ts.timestamp()),
                    "query": q,
                })
                seen_urls.add(url)
                if len(articles) >= MAX_PER_SYMBOL:
                    break
            if len(articles) >= MAX_PER_SYMBOL:
                break

        # スコア：直近だけを数えて簡易感情で重み付け
        recent = [a for a in articles if datetime.datetime.fromisoformat(a["ts"]) >= recent_since]
        raw = 0.0
        for a in recent:
            raw += 1.0 * tiny_sentiment(a["title"])

        per_symbol[sym] = {
            "as_of": asof,
            "recent_count": len(recent),
            "total_count": len(articles),
            "raw_signal": round(raw, 6),
            "examples": [{"title": a["title"], "source": a["source"], "url": a["url"]} for a in recent[:3]],
        }

    # 0–1 正規化（宇宙内パーセンタイル）
    raws = [v["raw_signal"] for v in per_symbol.values()]
    for sym, rec in per_symbol.items():
        rec["score_0_1"] = round(pct_rank(raws, rec["raw_signal"]), 6)

    # 保存
    out_latest = pathlib.Path(OUT_DIR) / "data" / "news" / "latest.json"
    out_today  = pathlib.Path(OUT_DIR) / "data" / DATE / "news.json"
    payload = {"as_of": asof, "recent_days": RECENT_DAYS, "lookback_days": LOOKBACK_DAYS, "items": per_symbol}
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_today.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, indent=2))
    out_today.write_text(json.dumps(payload, indent=2))
    print(f"[NEWS] saved: {out_latest} and {out_today} (symbols={len(per_symbol)})")

if __name__ == "__main__":
    main()
