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

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import numpy as np
except Exception:
    np = None


OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE")

NEWS_SLEEP_SEC = float(os.getenv("NEWS_SLEEP_SEC", "1.0"))
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
NEWS_MAX_PER_SYMBOL = int(os.getenv("NEWS_MAX_PER_SYMBOL", "20"))

NEWS_USE_MCAP_NORMALIZATION = os.getenv("NEWS_USE_MCAP_NORMALIZATION", "true").strip().lower() in {"1", "true", "yes", "on"}
NEWS_MCAP_FLOOR = float(os.getenv("NEWS_MCAP_FLOOR", "1000000000"))  # 10億USD
NEWS_MCAP_LOG_BASE = float(os.getenv("NEWS_MCAP_LOG_BASE", "10"))
NEWS_COUNT_EXP_SCALE = float(os.getenv("NEWS_COUNT_EXP_SCALE", "2.5"))
NEWS_MCAP_CACHE_MAX_AGE_HOURS = int(os.getenv("NEWS_MCAP_CACHE_MAX_AGE_HOURS", "168"))  # 7日
NEWS_YF_RETRIES = int(os.getenv("NEWS_YF_RETRIES", "2"))

CACHE_DIR = OUT_DIR / "cache"
MCAP_CACHE = CACHE_DIR / "market_caps.json"


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


def cache_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_mcap_cache() -> Dict[str, Dict[str, Any]]:
    raw = read_json(MCAP_CACHE)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        try:
            mcap = float(v.get("market_cap"))
            fetched_at = str(v.get("fetched_at", ""))
            if mcap > 0:
                out[str(k).upper()] = {"market_cap": mcap, "fetched_at": fetched_at}
        except Exception:
            continue
    return out


def save_mcap_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    write_json(MCAP_CACHE, cache)


def cache_is_fresh(fetched_at: str) -> bool:
    try:
        dt = datetime.fromisoformat(fetched_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - dt
        return age <= timedelta(hours=NEWS_MCAP_CACHE_MAX_AGE_HOURS)
    except Exception:
        return False


def get_market_cap_from_yf(sym: str) -> Optional[float]:
    if yf is None:
        return None

    for attempt in range(NEWS_YF_RETRIES + 1):
        try:
            t = yf.Ticker(sym)

            # fast_info 優先
            try:
                fi = t.fast_info
                mcap = None
                if fi is not None:
                    if hasattr(fi, "get"):
                        mcap = fi.get("market_cap")
                    else:
                        mcap = getattr(fi, "market_cap", None)
                if mcap is not None:
                    mcap = float(mcap)
                    if mcap > 0:
                        return mcap
            except Exception:
                pass

            # info フォールバック
            try:
                info = t.info
                if isinstance(info, dict):
                    mcap = info.get("marketCap")
                    if mcap is not None:
                        mcap = float(mcap)
                        if mcap > 0:
                            return mcap
            except Exception:
                pass

        except Exception as e:
            log("WARN", f"{sym}: yfinance market cap attempt {attempt + 1} failed: {e}")

        if attempt < NEWS_YF_RETRIES:
            time.sleep(0.8 + attempt * 0.6)

    return None


def get_market_cap(sym: str, cache: Dict[str, Dict[str, Any]]) -> Optional[float]:
    rec = cache.get(sym)
    if rec and cache_is_fresh(str(rec.get("fetched_at", ""))):
        try:
            mcap = float(rec["market_cap"])
            if mcap > 0:
                return mcap
        except Exception:
            pass

    mcap = get_market_cap_from_yf(sym)
    if mcap is not None and mcap > 0:
        cache[sym] = {
            "market_cap": float(mcap),
            "fetched_at": cache_now_iso(),
        }
        return float(mcap)

    # stale cache でも値があるなら最後の救済として使う
    if rec:
        try:
            mcap = float(rec["market_cap"])
            if mcap > 0:
                return mcap
        except Exception:
            pass

    return None


def recent_headline_stats(headlines: List[Dict[str, Any]], ref_date: datetime) -> Dict[str, Any]:
    cutoff = ref_date - timedelta(days=NEWS_LOOKBACK_DAYS)

    recent_items: List[Dict[str, Any]] = []
    unique_links = set()

    for h in headlines:
        try:
            dt = datetime.fromisoformat(str(h.get("published_at")))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                recent_items.append(h)
                link = str(h.get("link", "")).strip()
                if link:
                    unique_links.add(link)
        except Exception:
            continue

    return {
        "recent_count": len(recent_items),
        "unique_count": len(unique_links),
    }


def scaled_attention(recent_count: int) -> float:
    return max(0.0, min(1.0, 1.0 - math.exp(-recent_count / NEWS_COUNT_EXP_SCALE)))


def get_log_mcap_denom(market_cap: Optional[float], fallback_median_mcap: Optional[float]) -> Optional[float]:
    """
    market_cap が無い場合:
    1) universe内既知mcapの中央値を使う
    2) それも無い場合は None を返す
    """
    mcap = market_cap
    if mcap is None or mcap <= 0:
        mcap = fallback_median_mcap

    if mcap is None or mcap <= 0:
        return None

    mcap = max(float(mcap), NEWS_MCAP_FLOOR)

    if NEWS_MCAP_LOG_BASE == 10:
        denom = math.log10(mcap)
    else:
        denom = math.log(mcap, NEWS_MCAP_LOG_BASE)

    return max(1.0, denom)


def normalized_news_strength(recent_count: int, unique_count: int, market_cap: Optional[float], fallback_median_mcap: Optional[float]) -> float:
    """
    注目度の規模補正版 raw 指標。
    - recent_count を主軸
    - unique_count を少し加味
    - market_cap は log で補正
    """
    if recent_count <= 0 and unique_count <= 0:
        return 0.0

    base_attention = 0.75 * float(recent_count) + 0.25 * float(unique_count)

    if not NEWS_USE_MCAP_NORMALIZATION:
        return base_attention

    denom = get_log_mcap_denom(market_cap, fallback_median_mcap)
    if denom is None:
        # 最終フォールバック: floor付き生件数
        return base_attention

    return base_attention / denom


def minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []

    finite = [float(v) for v in values]
    vmin = min(finite)
    vmax = max(finite)

    if math.isclose(vmin, vmax):
        return [0.0 if v <= 0 else 1.0 for v in values]

    out = []
    for v in values:
        x = (float(v) - vmin) / (vmax - vmin)
        out.append(max(0.0, min(1.0, x)))
    return out


def median_or_none(values: List[float]) -> Optional[float]:
    vals = sorted([float(v) for v in values if v is not None and v > 0])
    if not vals:
        return None
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def main() -> None:
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is required")

    ref_date = datetime.fromisoformat(f"{REPORT_DATE}T23:59:59+00:00")
    universe = load_universe()
    if not universe:
        raise SystemExit("Universe is empty")

    mcap_cache = load_mcap_cache()

    # まず全銘柄の market cap を取りに行く
    known_mcaps: List[float] = []
    symbol_to_mcap: Dict[str, Optional[float]] = {}

    for u in universe:
        sym = u["symbol"]
        mcap = get_market_cap(sym, mcap_cache)
        symbol_to_mcap[sym] = mcap
        if mcap is not None and mcap > 0:
            known_mcaps.append(float(mcap))
        time.sleep(0.2)

    fallback_median_mcap = median_or_none(known_mcaps)
    if fallback_median_mcap is None:
        log("WARN", "No valid market caps found; news normalization will partially fall back to raw attention")
    else:
        log("INFO", f"Universe median market cap fallback: {fallback_median_mcap:.0f}")

    raw_items: List[Dict[str, Any]] = []
    raw_strengths: List[float] = []

    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")
        market_cap = symbol_to_mcap.get(sym)

        try:
            headlines = fetch_symbol_news(sym)
            stats = recent_headline_stats(headlines, ref_date)
            raw_strength = normalized_news_strength(
                recent_count=int(stats["recent_count"]),
                unique_count=int(stats["unique_count"]),
                market_cap=market_cap,
                fallback_median_mcap=fallback_median_mcap,
            )
        except Exception as e:
            log("WARN", f"{sym}: news fetch failed: {e}")
            headlines = []
            stats = {"recent_count": 0, "unique_count": 0}
            raw_strength = 0.0

        raw_items.append({
            "symbol": sym,
            "name": nm,
            "headline_count": len(headlines),
            "recent_count": int(stats["recent_count"]),
            "unique_count": int(stats["unique_count"]),
            "market_cap": market_cap,
            "raw_news_strength": raw_strength,
            "headlines": headlines[:NEWS_MAX_PER_SYMBOL],
        })
        raw_strengths.append(float(raw_strength))

        time.sleep(NEWS_SLEEP_SEC)

    save_mcap_cache(mcap_cache)

    norm_scores = minmax_normalize(raw_strengths)

    items: List[Dict[str, Any]] = []
    for row, norm_score in zip(raw_items, norm_scores):
        recent_sat = scaled_attention(int(row["recent_count"]))
        final_score = 0.75 * float(norm_score) + 0.25 * float(recent_sat)

        items.append({
            "symbol": row["symbol"],
            "name": row["name"],
            "score_0_1": round(max(0.0, min(1.0, final_score)), 6),
            "headline_count": row["headline_count"],
            "recent_count": row["recent_count"],
            "unique_count": row["unique_count"],
            "market_cap": row["market_cap"],
            "raw_news_strength": round(float(row["raw_news_strength"]), 8),
            "headlines": row["headlines"],
        })

    payload = {"date": REPORT_DATE, "items": items}

    day_path = OUT_DIR / "data" / REPORT_DATE / "news.json"
    latest_path = OUT_DIR / "data" / "news" / "latest.json"

    write_json(day_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote News: {day_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
