#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "10"))  # 当日+前日+直近3日平均の計算用に余裕を持たせる
NEWS_MAX_PER_SYMBOL = int(os.getenv("NEWS_MAX_PER_SYMBOL", "50"))

# スコア重み
NEWS_WEIGHT_TODAY = float(os.getenv("NEWS_WEIGHT_TODAY", "0.55"))
NEWS_WEIGHT_VS_YDAY = float(os.getenv("NEWS_WEIGHT_VS_YDAY", "0.20"))
NEWS_WEIGHT_VS_3DAY = float(os.getenv("NEWS_WEIGHT_VS_3DAY", "0.25"))

# スパイク抑制
NEWS_BASELINE_FLOOR = float(os.getenv("NEWS_BASELINE_FLOOR", "0.5"))

# 関連性判定
NEWS_STRICT_FILTER = os.getenv("NEWS_STRICT_FILTER", "true").strip().lower() in {"1", "true", "yes", "on"}


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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


def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\.\-\+\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_word_boundary_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.lower())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def build_company_aliases(symbol: str, name: str) -> Dict[str, Any]:
    """
    無関係記事除外用の別名辞書を作る。
    特に AI のような曖昧ティッカーは symbol 単独一致を禁止する。
    """
    symbol_u = symbol.upper().strip()
    name_norm = normalize_text(name)

    positive_terms: Set[str] = set()
    strong_terms: Set[str] = set()
    block_symbol_only = False

    # 会社名の基本形
    if name_norm:
        strong_terms.add(name_norm)

    # suffix除去版
    suffixes = [
        " corporation", " corp", " corp.", " inc", " inc.", " ltd", " ltd.",
        " limited", " plc", " holdings", " holdings co", " co", " co.",
        " technologies", " technology", " communications", " systems",
        " software", " therapeutics", " biotech", " biologics"
    ]

    base_name = name_norm
    for suf in suffixes:
        if base_name.endswith(suf):
            trimmed = base_name[: -len(suf)].strip()
            if len(trimmed) >= 3:
                positive_terms.add(trimmed)

    # symbol は長さや曖昧性で扱いを変える
    ambiguous_symbols = {"AI", "U", "IT", "ON", "A", "T", "C", "F", "R", "X"}
    if symbol_u in ambiguous_symbols or len(symbol_u) <= 2:
        block_symbol_only = True
    else:
        positive_terms.add(symbol_u.lower())

    # 特殊ルール
    # C3.ai は AI ティッカーが危険なので company name 系だけ通す
    if symbol_u == "AI":
        block_symbol_only = True
        strong_terms.update({
            "c3.ai",
            "c3 ai",
            "c3 code",
        })
        positive_terms.update({
            "c3.ai",
            "c3 ai",
            "c3",
        })

    elif symbol_u == "U":
        block_symbol_only = True
        strong_terms.update({
            "unity software",
            "unity software inc",
        })
        positive_terms.update({
            "unity software",
            "unity",
        })

    elif symbol_u == "SYM":
        # 英単語 "sym" は曖昧なので symbol 単独を弱く扱う
        block_symbol_only = True
        strong_terms.update({
            "symbotic",
            "symbotic inc",
        })
        positive_terms.update({
            "symbotic",
        })

    elif symbol_u == "TEM":
        block_symbol_only = True
        strong_terms.update({
            "tempus ai",
            "tempus ai inc",
        })
        positive_terms.update({
            "tempus ai",
            "tempus",
        })

    elif symbol_u == "AI":
        block_symbol_only = True

    # 一般名詞化しやすい会社名は symbol より company name 優先
    if " ai " in f" {name_norm} " or name_norm.endswith(" ai") or ".ai" in name_norm:
        block_symbol_only = True

    # 正規表現コンパイル
    strong_patterns = [make_word_boundary_pattern(x) for x in strong_terms if x]
    positive_patterns = [make_word_boundary_pattern(x) for x in positive_terms if x]

    return {
        "symbol": symbol_u,
        "name": name,
        "strong_terms": sorted(strong_terms),
        "positive_terms": sorted(positive_terms),
        "strong_patterns": strong_patterns,
        "positive_patterns": positive_patterns,
        "block_symbol_only": block_symbol_only,
    }


def is_relevant_headline(symbol: str, name: str, headline: Dict[str, Any]) -> bool:
    """
    Yahoo RSS の無関係記事除外。
    厳しめにフィルタする。
    """
    if not NEWS_STRICT_FILTER:
        return True

    alias = build_company_aliases(symbol, name)

    title = str(headline.get("title", "") or "")
    link = str(headline.get("link", "") or "")
    hay = normalize_text(f"{title} {link}")

    # 強い一致が1つでもあれば通す
    for pat in alias["strong_patterns"]:
        if pat.search(hay):
            return True

    # positive一致を数える
    hit_count = 0
    for pat in alias["positive_patterns"]:
        if pat.search(hay):
            hit_count += 1

    # block_symbol_only の場合は、symbolや短い単語だけで通さない
    if alias["block_symbol_only"]:
        # 会社名由来の positive 1件でも通すが、単なる短い曖昧ティッカーには依存しない
        company_like_hits = 0
        for term, pat in zip(alias["positive_terms"], alias["positive_patterns"]):
            if pat.search(hay):
                if len(term) >= 4 or "." in term or " " in term:
                    company_like_hits += 1
        return company_like_hits >= 1

    return hit_count >= 1


def filter_relevant_headlines(symbol: str, name: str, headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in headlines:
        if is_relevant_headline(symbol, name, h):
            out.append(h)
    return out


def build_daily_counts(headlines: List[Dict[str, Any]], ref_day: date) -> Dict[str, Any]:
    """
    使う指標:
    - 当日件数
    - 前日件数
    - 直近3日平均（当日除く）
    """
    deduped = dedupe_headlines(headlines)

    start_day = ref_day - timedelta(days=3)
    day_counts: Dict[date, int] = {start_day + timedelta(days=i): 0 for i in range(4)}

    for h in deduped:
        d = headline_day(h)
        if d is None:
            continue
        if start_day <= d <= ref_day:
            day_counts[d] = day_counts.get(d, 0) + 1

    today_count = int(day_counts.get(ref_day, 0))
    yday = ref_day - timedelta(days=1)
    prev1_count = int(day_counts.get(yday, 0))

    prev3_days = [ref_day - timedelta(days=i) for i in range(1, 4)]
    prev3_counts = [int(day_counts.get(d, 0)) for d in sorted(prev3_days)]
    avg_prev_3d_excl_today = float(sum(prev3_counts) / len(prev3_counts)) if prev3_counts else 0.0

    return {
        "deduped_headlines": deduped,
        "today_count": today_count,
        "prev1_count": prev1_count,
        "prev3_daily_counts": prev3_counts,
        "avg_prev_3d_excl_today": round(avg_prev_3d_excl_today, 4),
        "day_counts": {d.isoformat(): c for d, c in sorted(day_counts.items())},
    }


def calc_news_strength(today_count: int, prev1_count: int, avg_prev_3d: float) -> Dict[str, float]:
    """
    当日件数
    前日件数
    直近3日平均
    を使った異常値スコア
    """
    yday_base = max(float(prev1_count), NEWS_BASELINE_FLOOR)
    avg3_base = max(float(avg_prev_3d), NEWS_BASELINE_FLOOR)

    ratio_vs_yday = float(today_count) / yday_base if yday_base > 0 else 0.0
    ratio_vs_3day = float(today_count) / avg3_base if avg3_base > 0 else 0.0

    # 絶対件数も少し効かせる
    today_log = math.log1p(max(0.0, float(today_count)))
    yday_log_ratio = math.log1p(max(0.0, ratio_vs_yday))
    avg3_log_ratio = math.log1p(max(0.0, ratio_vs_3day))

    raw_strength = (
        NEWS_WEIGHT_TODAY * today_log
        + NEWS_WEIGHT_VS_YDAY * yday_log_ratio
        + NEWS_WEIGHT_VS_3DAY * avg3_log_ratio
    )

    return {
        "baseline_prev1_used": round(yday_base, 4),
        "baseline_prev3_used": round(avg3_base, 4),
        "ratio_vs_prev1": round(ratio_vs_yday, 6),
        "ratio_vs_prev3avg": round(ratio_vs_3day, 6),
        "today_log": round(today_log, 6),
        "raw_strength": round(raw_strength, 8),
    }


def percentile_normalize(values: List[float]) -> List[float]:
    if not values:
        return []

    vals = [float(v) for v in values]
    sorted_vals = sorted(vals)
    n = len(sorted_vals)

    out: List[float] = []
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

    raw_items: List[Dict[str, Any]] = []
    raw_strengths: List[float] = []

    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")

        try:
            fetched = fetch_symbol_news(sym)
            relevant = filter_relevant_headlines(sym, nm, fetched)
            daily = build_daily_counts(relevant, ref_day)

            score_parts = calc_news_strength(
                today_count=int(daily["today_count"]),
                prev1_count=int(daily["prev1_count"]),
                avg_prev_3d=float(daily["avg_prev_3d_excl_today"]),
            )

            raw_strength = float(score_parts["raw_strength"])

        except Exception as e:
            log("WARN", f"{sym}: news fetch/scoring failed: {e}")
            fetched = []
            relevant = []
            daily = {
                "deduped_headlines": [],
                "today_count": 0,
                "prev1_count": 0,
                "prev3_daily_counts": [0, 0, 0],
                "avg_prev_3d_excl_today": 0.0,
                "day_counts": {},
            }
            score_parts = {
                "baseline_prev1_used": NEWS_BASELINE_FLOOR,
                "baseline_prev3_used": NEWS_BASELINE_FLOOR,
                "ratio_vs_prev1": 0.0,
                "ratio_vs_prev3avg": 0.0,
                "today_log": 0.0,
                "raw_strength": 0.0,
            }
            raw_strength = 0.0

        raw_items.append({
            "symbol": sym,
            "name": nm,
            "headline_count_fetched": len(fetched),
            "headline_count_relevant": len(relevant),
            "deduped_headline_count": len(daily["deduped_headlines"]),
            "today_count": daily["today_count"],
            "prev1_count": daily["prev1_count"],
            "prev3_daily_counts": daily["prev3_daily_counts"],
            "avg_prev_3d_excl_today": daily["avg_prev_3d_excl_today"],
            "baseline_prev1_used": score_parts["baseline_prev1_used"],
            "baseline_prev3_used": score_parts["baseline_prev3_used"],
            "ratio_vs_prev1": score_parts["ratio_vs_prev1"],
            "ratio_vs_prev3avg": score_parts["ratio_vs_prev3avg"],
            "raw_news_strength": score_parts["raw_strength"],
            "day_counts": daily["day_counts"],
            "headlines": daily["deduped_headlines"][:NEWS_MAX_PER_SYMBOL],
        })
        raw_strengths.append(raw_strength)

        time.sleep(NEWS_SLEEP_SEC)

    pct_scores = percentile_normalize(raw_strengths)

    items: List[Dict[str, Any]] = []
    for row, pct_score in zip(raw_items, pct_scores):
        today_count = int(row["today_count"])
        today_presence = 0.0 if today_count <= 0 else min(1.0, math.log1p(today_count) / math.log1p(6.0))

        final_score = 0.85 * float(pct_score) + 0.15 * float(today_presence)
        final_score = max(0.0, min(1.0, final_score))

        items.append({
            "symbol": row["symbol"],
            "name": row["name"],
            "score_0_1": round(final_score, 6),
            "headline_count_fetched": row["headline_count_fetched"],
            "headline_count_relevant": row["headline_count_relevant"],
            "deduped_headline_count": row["deduped_headline_count"],
            "today_count": row["today_count"],
            "prev1_count": row["prev1_count"],
            "prev3_daily_counts": row["prev3_daily_counts"],
            "avg_prev_3d_excl_today": row["avg_prev_3d_excl_today"],
            "baseline_prev1_used": row["baseline_prev1_used"],
            "baseline_prev3_used": row["baseline_prev3_used"],
            "ratio_vs_prev1": row["ratio_vs_prev1"],
            "ratio_vs_prev3avg": row["ratio_vs_prev3avg"],
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
