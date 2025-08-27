#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News fetcher (idempotent + lightweight)
- Source: Google News RSS (free)
- Per symbol: query by 'Company Name' or 'SYMBOL stock'
- Scoring:
   * Buzz (multiplier): recent 3d count / prior 28d average per 3d
   * Sentiment (very lightweight lexicon) with freshness decay (7d half-life)
- Normalization across universe: percentile ranks to 0..1
- Idempotent: de-dup by (link/title) hash; keep a rolling window

Outputs:
- site/data/news/latest.json
- site/data/{DATE}/news.json
- site/cache/news_seen.json  (dedup set)
"""

import os, re, json, time, math, pathlib, datetime, hashlib, random
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import feedparser
import requests

# ---------------- Config ----------------
OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")

DATE = os.getenv("REPORT_DATE")
if not DATE:
    # Align to ET market date (after 18:00, use the same day; earlier, use previous bus. day)
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    dt = now_et.date()
    if now_et.hour < 18:
        dt = dt - datetime.timedelta(days=1)
    while dt.weekday() >= 5:
        dt = dt - datetime.timedelta(days=1)
    DATE = dt.isoformat()

USER_AGENT = os.getenv("NEWS_USER_AGENT", "futuretech-stock/1.0 (+contact@example.com)")
NEWS_SLEEP = float(os.getenv("NEWS_SLEEP", "1.2"))     # polite pause between queries
NEWS_JITTER = float(os.getenv("NEWS_JITTER", "0.8"))   # random jitter
NEWS_DAYS_KEEP = int(os.getenv("NEWS_DAYS_KEEP", "45"))  # keep window
RECENT_DAYS = int(os.getenv("NEWS_RECENT_DAYS", "3"))     # 3d for buzz numerator
LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "31")) # 31d history for buzz denom
MAX_PER_SYMBOL = int(os.getenv("NEWS_MAX_PER_SYMBOL", "60")) # cap loaded items per symbol per run

# Fallback positive/negative lexicon (tiny, lightweight)
POS_WORDS = set(w.lower() for w in """
beat beats beating bullish surge surges soaring soar record upgrade upgrades raised raises raise
optimistic positive outperform outperforms outperformed strong stronger strongest rally rallies
wins win winning growth expand expanding expansion breakthrough partnership partnerships
""".split())
NEG_WORDS = set(w.lower() for w in """
miss misses missed bearish slump slumps plunges plunge collapse downgrade downgrades cut cuts cutting
pessimistic negative underperform underperforms underperformed weak weaker weakest tumble tumbles
lawsuit lawsuits probe probes investigation investigations recall recalls halt halts
""".split())

HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/rss+xml"}

CACHE_DIR = pathlib.Path(OUT_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEEN_PATH = CACHE_DIR / "news_seen.json"

def load_seen() -> set:
    if SEEN_PATH.exists():
        try:
            return set(json.loads(SEEN_PATH.read_text()))
        except Exception:
            return set()
    return set()

def save_seen(seen: set):
    try:
        SEEN_PATH.write_text(json.dumps(sorted(list(seen))))
    except Exception:
        pass

def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df["query"] = df.apply(lambda r: build_query(r), axis=1)
    return df[["symbol","name","query"]]

def build_query(row) -> str:
    name = str(row.get("name","") or "").strip()
    sym = str(row.get("symbol","") or "").strip().upper()
    if name and name.lower() not in ("nan","none"):
        # Google News検索は社名で十分ヒット
        return f"{name}"
    return f"{sym} stock"

def sleep_politely():
    t = NEWS_SLEEP + random.uniform(0, NEWS_JITTER)
    time.sleep(t)

def fetch_rss(query: str) -> feedparser.FeedParserDict:
    # Google News RSS 検索
    # q= はURLエンコード不要でも動くが、念のためrequestsでpre-flight
    base = "https://news.google.com/rss/search"
    params = {"q": query, "hl":"en-US", "gl":"US", "ceid":"US:en"}
    url = requests.Request("GET", base, params=params).prepare().url
    return feedparser.parse(url, request_headers=HEADERS)

def hash_id(title: str, link: str) -> str:
    h = hashlib.sha1()
    h.update((title or "").encode("utf-8"))
    h.update((link or "").encode("utf-8"))
    return h.hexdigest()

def parse_datetime(entry) -> datetime.datetime|None:
    # feedparser は published_parsed 等を持つことが多い
    for key in ("published_parsed","updated_parsed"):
        t = entry.get(key)
        if t:
            try:
                return datetime.datetime(*t[:6], tzinfo=datetime.timezone.utc)
            except Exception:
                pass
    # fallback: now
    return None

def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def simple_sentiment(text: str) -> float:
    """ very light lexicon: (#pos - #neg)/sqrt(len_words); clip [-1,1] """
    if not text:
        return 0.0
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if not toks:
        return 0.0
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = (pos - neg) / max(1.0, math.sqrt(len(toks)))
    return max(-1.0, min(1.0, score))

def et_date(dt_utc: datetime.datetime|None) -> datetime.date:
    if not dt_utc:
        # assume now UTC
        dt_utc = datetime.datetime.now(datetime.timezone.utc)
    # convert to ET
    et = dt_utc.astimezone(ZoneInfo("America/New_York"))
    return et.date()

def pct_rank(vals: List[float], x: float|None) -> float:
    arr = sorted([v for v in vals if isinstance(v, (int,float)) and not math.isnan(v)])
    if not arr or x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    import bisect
    k = bisect.bisect_right(arr, x)
    return k / len(arr)

def ensure_dirs(date_iso: str):
    (pathlib.Path(OUT_DIR)/"data"/"news").mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"data"/date_iso).mkdir(parents=True, exist_ok=True)

def decay_weight(age_days: float, half_life_days: float = 7.0) -> float:
    if age_days <= 0:
        return 1.0
    lam = math.log(2.0) / max(1e-6, half_life_days)
    return math.exp(-lam * age_days)

def main():
    ensure_dirs(DATE)
    uni = load_universe()
    seen = load_seen()

    # 既存の latest を読み込み（ロールアップ & 冪等性のため）
    latest_path = pathlib.Path(OUT_DIR)/"data"/"news"/"latest.json"
    if latest_path.exists():
        try:
            existed = json.loads(latest_path.read_text())
        except Exception:
            existed = {"items":{}, "as_of": DATE}
    else:
        existed = {"items":{}, "as_of": DATE}

    # 作業用：すべての記事を保持（rolling window）
    all_articles: Dict[str, List[Dict]] = existed.get("articles", {})

    # 今回取得
    for _, row in uni.iterrows():
        sym = str(row["symbol"]).upper()
        q = str(row["query"])
        try:
            f = fetch_rss(q)
        except Exception as e:
            print(f"[WARN] RSS fetch failed for {sym}: {e}")
            sleep_politely()
            continue

        articles = all_articles.get(sym, [])
        added = 0

        for entry in f.entries[:MAX_PER_SYMBOL]:
            title = clean_text(entry.get("title", ""))
            link = entry.get("link", "")
            summary = clean_text(entry.get("summary", "") or entry.get("description",""))
            uid = hash_id(title, link)
            if uid in seen:
                continue

            dt = parse_datetime(entry)
            if not dt:
                # fallback: treat as now
                dt = datetime.datetime.now(datetime.timezone.utc)

            rec = {
                "id": uid,
                "title": title,
                "link": link,
                "summary": summary,
                "published_utc": dt.isoformat(),
                "published_date": et_date(dt).isoformat(),
                "sentiment": simple_sentiment(f"{title}. {summary}"),
                "source": clean_text(entry.get("source", "") or entry.get("author","") or f"GoogleNews:{q}")
            }
            articles.append(rec)
            seen.add(uid)
            added += 1

        # 古い記事を落とす（KEEP）
        cutoff = datetime.date.fromisoformat(DATE) - datetime.timedelta(days=NEWS_DAYS_KEEP)
        articles = [a for a in articles if datetime.date.fromisoformat(a["published_date"]) >= cutoff]
        all_articles[sym] = articles
        print(f"[INFO] {sym}: +{added} (total {len(articles)})")
        sleep_politely()

    # ---- 集計（buzz & sentiment）----
    ref_day = datetime.date.fromisoformat(DATE)
    def count_in(sym, days: int) -> int:
        from_day = ref_day - datetime.timedelta(days=days-1)
        return sum(1 for a in all_articles.get(sym, []) if from_day <= datetime.date.fromisoformat(a["published_date"]) <= ref_day)

    def weighted_sent(sym) -> float:
        arr = []
        for a in all_articles.get(sym, []):
            age = (ref_day - datetime.date.fromisoformat(a["published_date"])).days
            w = decay_weight(age, 7.0)
            arr.append(a["sentiment"] * w)
        if not arr:
            return 0.0
        return sum(arr)/sum(decay_weight((ref_day - datetime.date.fromisoformat(a["published_date"])).days, 7.0) for a in all_articles.get(sym, []))

    stats: Dict[str, Dict] = {}
    for sym in uni["symbol"]:
        sym = str(sym).upper()
        c3 = count_in(sym, RECENT_DAYS)                  # last 3d
        c_all = count_in(sym, LOOKBACK_DAYS)             # last ~1m
        prior_days = max(1, LOOKBACK_DAYS - RECENT_DAYS)
        prior_3d_avg = (c_all - c3) / (prior_days/RECENT_DAYS) if c_all > 0 else 0.0
        buzz_mult = (c3 / max(1.0, prior_3d_avg)) if prior_3d_avg > 0 else (1.0 if c3>0 else 0.0)

        sent = weighted_sent(sym)

        stats[sym] = {
            "count_3d": c3,
            "count_31d": c_all,
            "buzz_multiplier": round(buzz_mult, 6),
            "sentiment_weighted": round(sent, 6),
        }

    # 正規化 0..1
    buzz_vals = [v["buzz_multiplier"] for v in stats.values()]
    sent_vals = [v["sentiment_weighted"] for v in stats.values()]

    # ranks (descending for both)
    sorted_buzz = sorted([(sym, stats[sym]["buzz_multiplier"]) for sym in stats], key=lambda x: x[1], reverse=True)
    rank_buzz = {sym: i+1 for i,(sym,_) in enumerate(sorted_buzz)}

    sorted_sent = sorted([(sym, stats[sym]["sentiment_weighted"]) for sym in stats], key=lambda x: x[1], reverse=True)
    rank_sent = {sym: i+1 for i,(sym,_) in enumerate(sorted_sent)}

    for sym, rec in stats.items():
        rec["buzz_score_0_1"] = pct_rank(buzz_vals, rec["buzz_multiplier"])
        rec["sent_score_0_1"] = pct_rank(sent_vals, rec["sentiment_weighted"])
        rec["buzz_rank"] = rank_buzz.get(sym)
        rec["sent_rank"] = rank_sent.get(sym)

    # 出力
    payload = {
        "as_of": DATE,
        "window": {"recent_days": RECENT_DAYS, "lookback_days": LOOKBACK_DAYS, "keep_days": NEWS_DAYS_KEEP},
        "items": stats,
        "articles": all_articles,  # 詳細リンク保持（UI の details で使える）
    }

    latest = pathlib.Path(OUT_DIR)/"data"/"news"/"latest.json"
    todays = pathlib.Path(OUT_DIR)/"data"/DATE/"news.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    todays.parent.mkdir(parents=True, exist_ok=True)

    latest.write_text(json.dumps(payload, indent=2))
    todays.write_text(json.dumps(payload, indent=2))
    save_seen(seen)

    print(f"[NEWS] saved: {latest} and {todays} (symbols={len(stats)})")


if __name__ == "__main__":
    main()
