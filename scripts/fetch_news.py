#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News fetcher (Google News RSS) + light sentiment
- No API key required (RSS)
- Safe rate limiting & robust parsing
- Outputs:
  site/data/news/latest.json
  site/data/<REPORT_DATE>/news.json
Schema (per symbol):
  {
    "query": "...",
    "articles": N (recent window),
    "recent7": <int>,
    "prior28": <int>,
    "breakout_ratio": <float or null>,
    "avg_compound": <float>,   # [-1..1] VADER compound mean (title+summary)
    "pos_share": <float>,      # [0..1]
    "neg_share": <float>,      # [0..1]
    "raw_signal": <float or null>,  # breakout_ratio * max(avg_compound, 0)
    "score_0_1": <float>,      # percentile within universe
    "rank": <int>
  }
"""
import os, time, json, math, pathlib, datetime, random, re
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus
import requests
import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- Config ----------------
OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
DATE = os.getenv("REPORT_DATE") or datetime.datetime.now(ZoneInfo("America/New_York")).date().isoformat()

# Google News params
NEWS_LANG = os.getenv("NEWS_LANG", "en")
NEWS_REGION = os.getenv("NEWS_REGION", "US")    # ceid=US:en / gl=US / hl=en
NEWS_HL = f"{NEWS_LANG}-{NEWS_REGION}"
NEWS_CEID = f"{NEWS_REGION}:{NEWS_LANG}"
NEWS_GL = NEWS_REGION

# Window config
RECENT_DAYS = int(os.getenv("NEWS_RECENT_DAYS", "7"))       # signal window
PRIOR_DAYS  = int(os.getenv("NEWS_PRIOR_DAYS", "28"))       # baseline window
LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "35"))  # fetch range
MAX_PER_QUERY = int(os.getenv("NEWS_MAX_PER_QUERY", "60"))  # just a safeguard
SLEEP_BETWEEN = float(os.getenv("NEWS_SLEEP", "1.2"))
JITTER = float(os.getenv("NEWS_JITTER", "0.6"))

# Query shaping
QUERY_MODE = os.getenv("NEWS_QUERY_MODE", "name_or_symbol") # name_or_symbol | name_only | symbol_stock
SITE_FILTER = os.getenv("NEWS_SITE_FILTER", "").strip()     # e.g. "site:bloomberg.com OR site:reuters.com"

HDRS = {
    "User-Agent": "Mozilla/5.0 (compatible; futuretech-stock/1.0; +https://example.com)"
}

analyzer = SentimentIntensityAnalyzer()

def load_universe():
    df = pd.read_csv(UNIVERSE_CSV)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["name"] = df.get("name", pd.Series([""]*len(df))).fillna("").astype(str).str.strip()
    return df[["symbol","name"]]

def build_query(sym: str, name: str) -> str:
    """
    Make a Google News query term. Keep it simple & conservative to avoid false positives.
    """
    q_base = ""
    if QUERY_MODE == "name_only" and name:
        q_base = f"\"{name}\""
    elif QUERY_MODE == "symbol_stock" or (not name):
        q_base = f"\"{sym}\" stock"
    else:
        # default
        q_base = f"\"{name}\" OR (\"{sym}\" stock)"
    if SITE_FILTER:
        q_base = f"({q_base}) {SITE_FILTER}"
    return q_base

def gnews_rss_url(query: str) -> str:
    q = quote_plus(query)
    # Example:
    # https://news.google.com/rss/search?q=QUERY&hl=en-US&gl=US&ceid=US:en
    return f"https://news.google.com/rss/search?q={q}&hl={NEWS_HL}&gl={NEWS_GL}&ceid={NEWS_CEID}"

def fetch_feed(query: str):
    url = gnews_rss_url(query)
    try:
        r = requests.get(url, headers=HDRS, timeout=30)
        r.raise_for_status()
        return feedparser.parse(r.text)
    except Exception as e:
        print(f"[WARN] feed fetch failed: {e}")
        return {"entries": []}

def normalize_date(published):
    # feedparser may already parse. Try published_parsed first.
    try:
        if hasattr(published, "tm_year"):
            dt = datetime.datetime(*published[:6], tzinfo=datetime.timezone.utc).date()
            return dt
    except Exception:
        pass
    try:
        # 'Tue, 27 Aug 2024 12:34:56 GMT' like
        dt = feedparser.parse(f"Date: {published}").feed.get("updated_parsed")
        if dt and hasattr(dt, "tm_year"):
            return datetime.datetime(*dt[:6], tzinfo=datetime.timezone.utc).date()
    except Exception:
        pass
    return None

def dedup_entries(entries):
    seen = set()
    out = []
    for e in entries:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        key = (title[:120].lower(), re.sub(r"[?#].*$","",link.lower()))
        if key in seen: 
            continue
        seen.add(key)
        out.append(e)
    return out

def series_counts_by_day(entries, start_date, end_date):
    # Initialize day index
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    s = pd.Series(0, index=dates)
    for e in entries:
        dt = normalize_date(e.get("published_parsed") or e.get("published") or "")
        if dt is None:
            continue
        if dt < start_date or dt > end_date:
            continue
        s.loc[pd.Timestamp(dt)] += 1
    return s

def clean_text(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def sentiment_on_entry(e):
    text = clean_text(e.get("title","")) + " " + clean_text(e.get("summary",""))
    sc = analyzer.polarity_scores(text)
    return sc.get("compound", 0.0)

def pct_rank(vals, x):
    arr = sorted([v for v in vals if isinstance(v,(int,float)) and not (isinstance(v,float) and math.isnan(v))])
    if not arr or x is None or (isinstance(x,float) and math.isnan(x)):
        return 0.0
    import bisect
    k = bisect.bisect_right(arr, x)
    return k/len(arr)

def main():
    uni = load_universe()
    # Define window
    end = datetime.date.fromisoformat(DATE)
    start = end - datetime.timedelta(days=LOOKBACK_DAYS)
    recent_start = end - datetime.timedelta(days=RECENT_DAYS-1)  # inclusive
    prior_start  = recent_start - datetime.timedelta(days=PRIOR_DAYS)

    results = {}
    all_raw_signals = []

    for i, row in uni.iterrows():
        sym, name = row["symbol"], row["name"]
        query = build_query(sym, name)
        print(f"[NEWS] {sym}: {query}")
        feed = fetch_feed(query)
        entries = dedup_entries(feed.get("entries", [])[:MAX_PER_QUERY])

        # sentiment per entry (only within lookback window)
        sentiments = []
        in_window = []
        for e in entries:
            dt = normalize_date(e.get("published_parsed") or e.get("published") or "")
            if dt and start <= dt <= end:
                in_window.append(e)
                sentiments.append(sentiment_on_entry(e))
        avg_compound = float(pd.Series(sentiments).mean()) if sentiments else 0.0
        pos_share = float((pd.Series(sentiments)> 0.05).mean()) if sentiments else 0.0
        neg_share = float((pd.Series(sentiments)<-0.05).mean()) if sentiments else 0.0

        # daily counts
        s = series_counts_by_day(in_window, start, end)
        recent_cnt = int(s.loc[pd.Timestamp(recent_start):pd.Timestamp(end)].sum())
        prior_cnt  = int(s.loc[pd.Timestamp(prior_start):pd.Timestamp(recent_start - datetime.timedelta(days=1))].sum())

        # Breakout ratio (recent avg per day / prior median per day)
        recent_mean = (recent_cnt / RECENT_DAYS) if RECENT_DAYS > 0 else 0.0
        prior_window = s.loc[pd.Timestamp(prior_start):pd.Timestamp(recent_start - datetime.timedelta(days=1))]
        prior_med = float(prior_window.replace(0, pd.NA).median(skipna=True)) if len(prior_window)>0 else 0.0
        if not prior_med or math.isnan(prior_med) or prior_med <= 0:
            prior_med = 1e-9
        breakout_ratio = float(recent_mean / prior_med) if prior_med>0 else None

        # Raw signal = breakout * max(avg_compound, 0)
        raw_signal = None
        if breakout_ratio is not None:
            raw_signal = breakout_ratio * max(0.0, avg_compound)

        results[sym] = {
            "query": query,
            "articles": int(len(in_window)),
            "recent7": recent_cnt,
            "prior28": prior_cnt,
            "breakout_ratio": None if breakout_ratio is None or math.isinf(breakout_ratio) or math.isnan(breakout_ratio) else breakout_ratio,
            "avg_compound": avg_compound,
            "pos_share": pos_share,
            "neg_share": neg_share,
            "raw_signal": None if raw_signal is None or math.isinf(raw_signal) or math.isnan(raw_signal) else raw_signal,
            "score_0_1": 0.0,  # set later
            "rank": None,
        }
        if results[sym]["raw_signal"] is not None:
            all_raw_signals.append(results[sym]["raw_signal"])

        # polite pacing
        time.sleep(SLEEP_BETWEEN + random.uniform(0, JITTER))

    # Rank / percentile
    for sym, rec in results.items():
        rs = rec.get("raw_signal")
        rec["score_0_1"] = pct_rank(all_raw_signals, rs) if rs is not None else 0.0

    # rank: highest first
    order = sorted(results.items(), key=lambda kv: (kv[1].get("score_0_1", 0.0)), reverse=True)
    for idx, (sym, rec) in enumerate(order, 1):
        rec["rank"] = idx

    out_latest = pathlib.Path(OUT_DIR) / "data" / "news" / "latest.json"
    out_today  = pathlib.Path(OUT_DIR) / "data" / DATE / "news.json"
    payload = {
        "as_of": DATE,
        "window": {"recent_days": RECENT_DAYS, "prior_days": PRIOR_DAYS},
        "items": results
    }
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_today.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, indent=2))
    out_today.write_text(json.dumps(payload, indent=2))
    print(f"[NEWS] saved: {out_latest} and {out_today} ({len(results)} symbols)")
    # quick debug
    top3 = sorted(results.items(), key=lambda kv: kv[1]["score_0_1"], reverse=True)[:3]
    for sym, rec in top3:
        print(f"[NEWS][TOP] {sym}: raw={rec['raw_signal']}, score={rec['score_0_1']}, rank={rec['rank']}")
    
if __name__ == "__main__":
    main()
