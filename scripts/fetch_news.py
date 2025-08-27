#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight News fetcher (Google News RSS) with safe parsing and rate-limit friendly sleeps.

- Per symbol, queries Google News RSS (company name or "<SYMBOL> stock") within lookback window.
- Aggregates "recent" (NEWS_RECENT_DAYS) vs baseline (older part of NEWS_LOOKBACK_DAYS) to compute:
    * recent_count, baseline_daily_rate, recent_daily_rate, buzz_multiplier (recent/base)
- Also assigns a 0..1 buzz_score across the universe using percentile rank of recent_count.
- Saves to:
    site/data/news/latest.json
    site/data/YYYY-MM-DD/news.json
- Robust to odd RSS fields (e.g., entry['source'] being a dict), empty feeds, and network hiccups.
- Sleep + jitter between requests to avoid rate limiting; retries with small backoff.

Env:
  OUT_DIR                 default "site"
  UNIVERSE_CSV            default "data/universe.csv"
  REPORT_DATE             default ET market date (today ET)
  NEWS_USER_AGENT         default "futuretech-stock/1.0 (news-fetcher)"
  NEWS_SLEEP              default "1.2"   (seconds between symbols)
  NEWS_JITTER             default "0.8"   (random[0..jitter] added)
  NEWS_DAYS_KEEP          default "45"    (not used here for pruning files; kept for future)
  NEWS_RECENT_DAYS        default "3"
  NEWS_LOOKBACK_DAYS      default "31"
  NEWS_MAX_PER_SYMBOL     default "60"    (max articles to retain per symbol in payload)
  NEWS_RETRIES            default "3"     (per-symbol fetch retries)
  NEWS_TIMEOUT            default "15"    (requests timeout)
"""

import os, sys, json, time, math, random, pathlib, datetime, re
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests
import feedparser

# ---------------- Env ----------------
OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
REPORT_DATE = os.getenv("REPORT_DATE")  # may be None; use ET date if so

NEWS_USER_AGENT = os.getenv("NEWS_USER_AGENT", "futuretech-stock/1.0 (news-fetcher)")
NEWS_SLEEP = float(os.getenv("NEWS_SLEEP", "1.2"))
NEWS_JITTER = float(os.getenv("NEWS_JITTER", "0.8"))
NEWS_DAYS_KEEP = int(os.getenv("NEWS_DAYS_KEEP", "45"))
NEWS_RECENT_DAYS = int(os.getenv("NEWS_RECENT_DAYS", "3"))
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "31"))
NEWS_MAX_PER_SYMBOL = int(os.getenv("NEWS_MAX_PER_SYMBOL", "60"))
NEWS_RETRIES = int(os.getenv("NEWS_RETRIES", "3"))
NEWS_TIMEOUT = int(os.getenv("NEWS_TIMEOUT", "15"))

HEADERS = {
    "User-Agent": NEWS_USER_AGENT,
    "Accept": "application/rss+xml, application/xml, text/xml, */*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# ---------------- Helpers ----------------
def et_market_date() -> str:
    if REPORT_DATE:
        return REPORT_DATE
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # after-hours partial guard is not critical for news; keep simple
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d.isoformat()

DATE = et_market_date()

def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    # normalize
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    if "name" not in df.columns:
        df["name"] = ""
    return df[["symbol","name"]]

def build_query(symbol: str, name: str) -> str:
    name = (name or "").strip()
    sym = (symbol or "").strip().upper()
    if name and name.lower() not in ("nan","none"):
        # favor company name, fall back to symbol stock
        q = f"\"{name}\" OR {sym} stock"
    else:
        q = f"{sym} stock"
    # limit by time window
    q = f"{q} when:{NEWS_LOOKBACK_DAYS}d"
    return q

def google_news_rss_url(q: str) -> str:
    base = "https://news.google.com/rss/search"
    # US/English to keep it consistent
    return f"{base}?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"

def sleep_between_symbols():
    sec = NEWS_SLEEP + random.uniform(0, NEWS_JITTER)
    time.sleep(sec)

def pct_rank(vals: List[float], x: float) -> float:
    arr = sorted([float(v) for v in vals if isinstance(v, (int,float))])
    if not arr:
        return 0.0
    # clamp x if None
    try:
        xv = float(x)
    except Exception:
        return 0.0
    import bisect
    k = bisect.bisect_right(arr, xv)
    return k / len(arr)

def clean_text(s: Any) -> str:
    """Make a safe single-line string from possibly dict/None/FeedParserDict."""
    if isinstance(s, dict):
        # Google News source often like {'title': 'Reuters', ...}
        s = s.get("title") or s.get("name") or ""
    elif hasattr(s, "get"):  # FeedParserDict
        try:
            t = s.get("title")
            if t:
                s = t
            else:
                s = ""
        except Exception:
            s = ""
    elif s is None:
        s = ""
    else:
        s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_iso(dt_struct) -> str:
    """Convert feedparser time.struct_time or string to ISO date."""
    if dt_struct is None:
        return ""
    try:
        if isinstance(dt_struct, str):
            # crude fallback: keep YYYY-MM-DD if present
            m = re.search(r"(\d{4}-\d{2}-\d{2})", dt_struct)
            if m:
                return m.group(1)
            # try RFC-2822-like date
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(dt_struct).date().isoformat()
        import time as _time
        # struct_time -> datetime
        dt = datetime.datetime(*dt_struct[:6], tzinfo=datetime.timezone.utc).date().isoformat()
        return dt
    except Exception:
        return ""

def within_days(date_iso: str, days: int, ref_date: datetime.date) -> bool:
    try:
        d = datetime.date.fromisoformat(date_iso)
    except Exception:
        return False
    return 0 <= (ref_date - d).days <= days

def fetch_feed(url: str) -> feedparser.FeedParserDict | None:
    """Fetch RSS via requests to control UA/timeout, then parse with feedparser."""
    last_err = None
    for attempt in range(1, NEWS_RETRIES+1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=NEWS_TIMEOUT)
            if resp.status_code == 429:
                # cool down a bit on 429
                sl = min(30 * attempt, 120)
                print(f"[NEWS] 429 Too Many Requests, sleeping {sl}s ...", flush=True)
                time.sleep(sl)
                continue
            if not resp.ok:
                last_err = Exception(f"HTTP {resp.status_code}")
                time.sleep(1.0 * attempt)
                continue
            fp = feedparser.parse(resp.content)
            # some feeds may parse but be empty; treat as ok
            return fp
        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)
    if last_err:
        print(f"[WARN] fetch_feed failed: {last_err}", file=sys.stderr)
    return None

def compute_buzz_metrics(dates: List[str], ref: datetime.date) -> Dict[str, float]:
    """
    dates: list of ISO dates (YYYY-MM-DD) for fetched articles within lookback.
    Return recent_count, rates and multiplier.
    """
    cutoff_recent = ref - datetime.timedelta(days=NEWS_RECENT_DAYS)
    cutoff_start  = ref - datetime.timedelta(days=NEWS_LOOKBACK_DAYS)

    recent = sum(1 for d in dates if within_days(d, NEWS_RECENT_DAYS, ref))
    old    = sum(1 for d in dates if (d >= cutoff_start.isoformat() and d < cutoff_recent.isoformat()))

    recent_days = max(1, NEWS_RECENT_DAYS)
    old_days = max(1, NEWS_LOOKBACK_DAYS - NEWS_RECENT_DAYS)

    recent_rate = recent / recent_days
    old_rate = old / old_days if old_days > 0 else 0.0
    base = old_rate if old_rate > 0 else 1e-6
    mult = recent_rate / base
    # tame extremes
    mult = max(0.25, min(8.0, mult))
    return {
        "recent_count": float(recent),
        "recent_daily_rate": float(recent_rate),
        "baseline_daily_rate": float(old_rate),
        "buzz_multiplier": float(mult),
    }

# ---------------- Main ----------------
def main():
    out_dir = pathlib.Path(OUT_DIR)
    (out_dir / "data" / DATE).mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "news").mkdir(parents=True, exist_ok=True)

    uni = load_universe()
    ref_date = datetime.date.fromisoformat(DATE)

    per_symbol: Dict[str, Dict[str, Any]] = {}
    all_recent_counts: List[float] = []
    all_multipliers: List[float] = []

    for _, row in uni.iterrows():
        sym = str(row["symbol"]).upper()
        name = str(row["name"]) if not pd.isna(row["name"]) else ""
        q = build_query(sym, name)
        url = google_news_rss_url(q)

        fp = fetch_feed(url)
        sleep_between_symbols()

        entries = (fp.entries if fp and hasattr(fp, "entries") else []) or []
        articles = []
        dates = []

        for e in entries[:NEWS_MAX_PER_SYMBOL]:
            # dates
            dt_iso = (
                to_iso(getattr(e, "published_parsed", None) or e.get("published"))
                or to_iso(getattr(e, "updated_parsed", None) or e.get("updated"))
            )
            if not dt_iso:
                continue

            # keep only within LOOKBACK
            if not within_days(dt_iso, NEWS_LOOKBACK_DAYS, ref_date):
                continue

            title = clean_text(e.get("title"))
            summary = clean_text(e.get("summary"))
            link = ""
            if "link" in e and e.get("link"):
                link = str(e.get("link"))
            elif hasattr(e, "links") and e.links:
                try:
                    link = e.links[0].get("href") or ""
                except Exception:
                    link = ""

            source = clean_text(e.get("source") or e.get("author") or "")
            if not source and hasattr(e, "source"):
                # some versions: e.source.title
                try:
                    source = clean_text(getattr(e.source, "title", "") or "")
                except Exception:
                    pass

            articles.append({
                "title": title[:280] if title else "",
                "summary": summary[:400] if summary else "",
                "url": link,
                "published": dt_iso,
                "source": source[:80] if source else "",
            })
            dates.append(dt_iso)

        metrics = compute_buzz_metrics(dates, ref_date)
        all_recent_counts.append(metrics["recent_count"])
        all_multipliers.append(metrics["buzz_multiplier"])

        per_symbol[sym] = {
            "query": q,
            "articles": articles,         # trimmed to NEWS_MAX_PER_SYMBOL in-loop
            **metrics,                    # recent_count, rates, multiplier
        }

    # Universe-level ranks/scores
    for sym, rec in per_symbol.items():
        rc = rec.get("recent_count", 0.0)
        mult = rec.get("buzz_multiplier", 0.0)

        rec["buzz_score_0_1"] = round(pct_rank(all_recent_counts, rc), 6)
        rec["mult_rank_0_1"]  = round(pct_rank(all_multipliers, mult), 6)

    # Save
    payload = {
        "as_of": DATE,
        "recent_days": NEWS_RECENT_DAYS,
        "lookback_days": NEWS_LOOKBACK_DAYS,
        "items": per_symbol,
    }

    out_latest = (out_dir / "data" / "news" / "latest.json")
    out_today  = (out_dir / "data" / DATE / "news.json")
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_today.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, indent=2))
    out_today.write_text(json.dumps(payload, indent=2))

    # Small summary
    with_items = sum(1 for v in per_symbol.values() if v.get("articles"))
    print(f"[NEWS] saved: {out_latest} and {out_today} ({len(per_symbol)} symbols; non-empty: {with_items})")

if __name__ == "__main__":
    main()
