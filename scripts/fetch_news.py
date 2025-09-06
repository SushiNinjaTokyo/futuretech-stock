#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import feedparser

OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE") or datetime.utcnow().date().isoformat()
LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
MAX_PER = int(os.getenv("NEWS_MAX_PER_SYMBOL", "20"))
SLEEP_SEC = float(os.getenv("NEWS_SLEEP_SEC", "1.0"))

NEWS_SOURCES = [
    "https://news.google.com/rss/search?q={sym}+stock&hl=en-US&gl=US&ceid=US:en",
]

def load_universe(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return ["NVDA","MSFT","PLTR","AI","ISRG","TER","SYM","RKLB","IRDM","VSAT",
                "INOD","SOUN","MNDY","AVAV","PERF","GDRX","ABCL","U","TEM","VRT"]
    syms: List[str] = []
    for line in csv_path.read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        syms.append(t.split(",")[0].strip())
    return syms[:20]

def sanitize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def news_score(count: int, max_count: int) -> float:
    if max_count <= 0: 
        return 0.0
    c = max(0, min(count, max_count))
    return c / float(max_count)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    day_dir = OUT_DIR / "data" / REPORT_DATE
    day_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(UNIVERSE_CSV)

    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    per_symbol_count: Dict[str, int] = {}
    max_seen = 1

    articles_by_sym: Dict[str, List[Dict]] = {s: [] for s in universe}

    for s in universe:
        total = 0
        for tmpl in NEWS_SOURCES:
            url = tmpl.format(sym=s)
            try:
                feed = feedparser.parse(url)
                for e in (feed.entries or []):
                    # pubDate のパースは feedparser 任せ（tzが飛ぶこともあるのでゆるく）
                    title = sanitize_text(getattr(e, "title", ""))
                    link = sanitize_text(getattr(e, "link", ""))
                    if not title or not link:
                        continue
                    articles_by_sym[s].append({"title": title, "link": link})
                    total += 1
                    if total >= MAX_PER:
                        break
            except Exception:
                # ソース単位で落ちても進む
                pass
            if total >= MAX_PER:
                break
        per_symbol_count[s] = total
        if total > max_seen:
            max_seen = total

    items = []
    for s in universe:
        sc = round(news_score(per_symbol_count.get(s, 0), max_seen), 12)
        items.append({
            "symbol": s,
            "score_0_1": sc,
            "components": {"news": sc},
            "articles": articles_by_sym.get(s, [])[:MAX_PER]
        })

    payload = {
        "date": REPORT_DATE,
        "items": items,
        "meta": {"lookback_days": LOOKBACK_DAYS, "max_per_symbol": MAX_PER}
    }

    (OUT_DIR / "data" / "news").mkdir(parents=True, exist_ok=True)
    latest = OUT_DIR / "data" / "news" / "latest.json"
    byday = day_dir / "news.json"
    for p in (latest, byday):
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[NEWS] saved: {latest} and {byday} (symbols={len(items)})")

if __name__ == "__main__":
    main()
