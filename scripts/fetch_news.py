#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import feedparser
except Exception:
    feedparser = None


OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE")
NEWS_SLEEP_SEC = float(os.getenv("NEWS_SLEEP_SEC", "1.0"))
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
NEWS_MAX_PER_SYMBOL = int(os.getenv("NEWS_MAX_PER_SYMBOL", "20"))


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_universe() -> List[Dict[str, str]]:
    if not UNIVERSE_CSV.exists():
        return []
    df = pd.read_csv(UNIVERSE_CSV)
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol", list(df.columns)[0])
    name_col = cols.get("name")
    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip().upper()
        if not sym:
            continue
        nm = str(row.get(name_col, "")).strip() if name_col else ""
        out.append({"symbol": sym, "name": nm})
    return out


def parse_dt(entry: Any) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        st = getattr(entry, attr, None)
        if st:
            try:
                return datetime(*st[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def yahoo_rss_url(sym: str) -> str:
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"


def fetch_symbol_news(sym: str) -> List[Dict[str, Any]]:
    if feedparser is None:
        return []
    url = yahoo_rss_url(sym)
    feed = feedparser.parse(url)
    items: List[Dict[str, Any]] = []
    for e in getattr(feed, "entries", [])[:NEWS_MAX_PER_SYMBOL]:
        dt = parse_dt(e)
        items.append({
            "title": getattr(e, "title", "") or "",
            "link": getattr(e, "link", "") or "",
            "published_at": dt.isoformat() if dt else None,
            "source": "Yahoo Finance RSS",
        })
    return items


def score_from_headlines(headlines: List[Dict[str, Any]], ref_date: datetime) -> float:
    cutoff = ref_date - timedelta(days=NEWS_LOOKBACK_DAYS)
    recent = 0
    for h in headlines:
        try:
            dt = datetime.fromisoformat(str(h.get("published_at")))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                recent += 1
        except Exception:
            continue

    # 0件=0, 5件以上でほぼ頭打ち
    return max(0.0, min(1.0, 1.0 - math.exp(-recent / 2.5)))


def main() -> None:
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is required")

    ref_date = datetime.fromisoformat(f"{REPORT_DATE}T23:59:59+00:00")
    universe = load_universe()

    items: List[Dict[str, Any]] = []
    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")
        try:
            headlines = fetch_symbol_news(sym)
            score = score_from_headlines(headlines, ref_date)
        except Exception as e:
            log("WARN", f"{sym}: news fetch failed: {e}")
            headlines = []
            score = 0.0

        items.append({
            "symbol": sym,
            "name": nm,
            "score_0_1": round(score, 6),
            "headline_count": len(headlines),
            "headlines": headlines[:NEWS_MAX_PER_SYMBOL],
        })

        time.sleep(NEWS_SLEEP_SEC)

    payload = {"date": REPORT_DATE, "items": items}
    day_path = OUT_DIR / "data" / REPORT_DATE / "news.json"
    latest_path = OUT_DIR / "data" / "news" / "latest.json"
    write_json(day_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote News: {day_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
