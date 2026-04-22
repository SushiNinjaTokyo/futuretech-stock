#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import feedparser
except Exception:
    feedparser = None

try:
    import numpy as np
except Exception:
    np = None


OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE")

NEWS_SLEEP_SEC = float(os.getenv("NEWS_SLEEP_SEC", "1.0"))
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "14"))  # 履歴確保用。最低8日は欲しい
NEWS_MAX_PER_SYMBOL = int(os.getenv("NEWS_MAX_PER_SYMBOL", "50"))

# スコア設計パラメータ
NEWS_BASELINE_DAYS = int(os.getenv("NEWS_BASELINE_DAYS", "7"))          # 当日除く過去7日平均
NEWS_BASELINE_FLOOR = float(os.getenv("NEWS_BASELINE_FLOOR", "0.5"))    # 0除算/過剰スパイク防止
NEWS_SPIKE_WEIGHT = float(os.getenv("NEWS_SPIKE_WEIGHT", "0.75"))
NEWS_TODAY_WEIGHT = float(os.getenv("NEWS_TODAY_WEIGHT", "0.25"))


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return {}


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def load_universe() -> List[Dict[str, str]]:
    if not UNIVERSE_CSV.exists():
        log("ERROR", f"Universe CSV missing: {UNIVERSE_CSV}")
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


def headline_day(h: Dict[str, Any]) -> Optional[date]:
    try:
        dt = datetime.fromisoformat(str(h.get("published_at")))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date()
    except Exception:
        return None


def dedupe_headlines(headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同一link優先で重複除去。
    linkが無い場合は title + published_at で代替。
    """
    seen = set()
    out: List[Dict[str, Any]] = []

    for h in headlines:
        link = str(h.get("link", "")).strip()
        title = str(h.get("title", "")).strip()
        published_at = str(h.get("published_at", "")).strip()

        key = ("link", link) if link else ("fallback", title, published_at)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)

    return out


def build_daily_counts(headlines: List[Dict[str, Any]], ref_day: date, baseline_days: int) -> Dict[str, Any]:
    """
    当日と、当日を除く過去 baseline_days 日分の日次件数を作る。
    件数が無い日は 0 件として埋める。
    """
    deduped = dedupe_headlines(headlines)

    start_day = ref_day - timedelta(days=baseline_days)
    day_counts: Dict[date, int] = {start_day + timedelta(days=i): 0 for i in range(baseline_days + 1)}

    for h in deduped:
        d = headline_day(h)
        if d is None:
            continue
        if start_day <= d <= ref_day:
            day_counts[d] = day_counts.get(d, 0) + 1

    today_count = int(day_counts.get(ref_day, 0))

    prev_days = [ref_day - timedelta(days=i) for i in range(1, baseline_days + 1)]
    prev_counts = [int(day_counts.get(d, 0)) for d in sorted(prev_days)]

    avg_prev = float(sum(prev_counts) / len(prev_counts)) if prev_counts else 0.0
    max_prev = max(prev_counts) if prev_counts else 0

    return {
        "deduped_headlines": deduped,
        "today_count": today_count,
        "prev_daily_counts": prev_counts,
        "avg_prev_7d_excl_today": round(avg_prev, 4),
        "max_prev_7d_excl_today": max_prev,
        "day_counts": {d.isoformat(): c for d, c in sorted(day_counts.items())},
    }


def robust_spike_strength(today_count: int, avg_prev: float, baseline_floor: float) -> Dict[str, float]:
    """
    その日の異常率を作る。
    floor を入れて、普段0件の銘柄が1件出ただけで暴れすぎないようにする。
    """
    baseline = max(float(avg_prev), float(baseline_floor))
    spike_ratio = float(today_count) / baseline if baseline > 0 else 0.0

    # ratioをそのまま使うと荒れやすいので log1p で圧縮
    spike_log = math.log1p(max(0.0, spike_ratio))
    today_log = math.log1p(max(0.0, float(today_count)))

    raw_strength = NEWS_SPIKE_WEIGHT * spike_log + NEWS_TODAY_WEIGHT * today_log

    return {
        "baseline_used": round(baseline, 4),
        "spike_ratio": round(spike_ratio, 6),
        "spike_log": round(spike_log, 6),
        "today_log": round(today_log, 6),
        "raw_strength": round(raw_strength, 8),
    }


def percentile_normalize(values: List[float]) -> List[float]:
    """
    min-maxではなくpercentileベース。
    上位が全部1.0になる問題を減らし、100%多発を抑える。
    """
    if not values:
        return []

    vals = [float(v) for v in values]
    sorted_vals = sorted(vals)

    out: List[float] = []
    n = len(sorted_vals)

    for v in vals:
        if n == 1:
            out.append(1.0 if v > 0 else 0.0)
            continue

        le = sum(1 for x in sorted_vals if x <= v)
        score = (le - 1) / (n - 1) if n > 1 else 0.0
        out.append(max(0.0, min(1.0, score)))

    return out


def main() -> None:
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is required")

    ref_day = date.fromisoformat(REPORT_DATE)
    universe = load_universe()
    if not universe:
        raise SystemExit("Universe is empty")

    if NEWS_BASELINE_DAYS < 1:
        raise SystemExit("NEWS_BASELINE_DAYS must be >= 1")

    raw_items: List[Dict[str, Any]] = []
    raw_strengths: List[float] = []

    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")

        try:
            headlines = fetch_symbol_news(sym)

            daily = build_daily_counts(
                headlines=headlines,
                ref_day=ref_day,
                baseline_days=NEWS_BASELINE_DAYS,
            )

            spike = robust_spike_strength(
                today_count=int(daily["today_count"]),
                avg_prev=float(daily["avg_prev_7d_excl_today"]),
                baseline_floor=NEWS_BASELINE_FLOOR,
            )

            raw_strength = float(spike["raw_strength"])

        except Exception as e:
            log("WARN", f"{sym}: news fetch/scoring failed: {e}")
            headlines = []
            daily = {
                "deduped_headlines": [],
                "today_count": 0,
                "prev_daily_counts": [0 for _ in range(NEWS_BASELINE_DAYS)],
                "avg_prev_7d_excl_today": 0.0,
                "max_prev_7d_excl_today": 0,
                "day_counts": {},
            }
            spike = {
                "baseline_used": NEWS_BASELINE_FLOOR,
                "spike_ratio": 0.0,
                "spike_log": 0.0,
                "today_log": 0.0,
                "raw_strength": 0.0,
            }
            raw_strength = 0.0

        raw_items.append({
            "symbol": sym,
            "name": nm,
            "headline_count": len(headlines),
            "deduped_headline_count": len(daily["deduped_headlines"]),
            "today_count": daily["today_count"],
            "prev_daily_counts": daily["prev_daily_counts"],
            "avg_prev_7d_excl_today": daily["avg_prev_7d_excl_today"],
            "max_prev_7d_excl_today": daily["max_prev_7d_excl_today"],
            "baseline_used": spike["baseline_used"],
            "spike_ratio": spike["spike_ratio"],
            "raw_news_strength": spike["raw_strength"],
            "headlines": daily["deduped_headlines"][:NEWS_MAX_PER_SYMBOL],
            "day_counts": daily["day_counts"],
        })
        raw_strengths.append(raw_strength)

        time.sleep(NEWS_SLEEP_SEC)

    pct_scores = percentile_normalize(raw_strengths)

    items: List[Dict[str, Any]] = []
    for row, pct_score in zip(raw_items, pct_scores):
        # 当日件数が0なら最終スコアも0寄りに抑える
        today_count = int(row["today_count"])
        today_presence = 0.0 if today_count <= 0 else min(1.0, math.log1p(today_count) / math.log1p(5.0))

        final_score = 0.85 * float(pct_score) + 0.15 * float(today_presence)
        final_score = max(0.0, min(1.0, final_score))

        items.append({
            "symbol": row["symbol"],
            "name": row["name"],
            "score_0_1": round(final_score, 6),
            "headline_count": row["headline_count"],
            "deduped_headline_count": row["deduped_headline_count"],
            "today_count": row["today_count"],
            "prev_daily_counts": row["prev_daily_counts"],
            "avg_prev_7d_excl_today": row["avg_prev_7d_excl_today"],
            "max_prev_7d_excl_today": row["max_prev_7d_excl_today"],
            "baseline_used": row["baseline_used"],
            "spike_ratio": row["spike_ratio"],
            "raw_news_strength": row["raw_news_strength"],
            "day_counts": row["day_counts"],
            "headlines": row["headlines"],
        })

    payload = {"date": REPORT_DATE, "items": items}

    day_path = OUT_DIR / "data" / REPORT_DATE / "news.json"
    latest_path = OUT_DIR / "data" / "news" / "latest.json"

    write_json(day_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote News: {day_path} ({len(items)} items)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in fetch_news: {e}")
        sys.exit(1)
