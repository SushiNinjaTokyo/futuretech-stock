#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
REPORT_DATE = os.getenv("REPORT_DATE")


HORIZONS = [5, 10, 20]
REGISTRY_PATH = OUT_DIR / "data" / "signals" / "registry.json"
SUMMARY_PATH = OUT_DIR / "data" / "signals" / "summary_latest.json"
OUTCOMES_PATH = OUT_DIR / "data" / "signals" / "outcomes_latest.json"


def log(level: str, msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}",
        flush=True,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def sanitize(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=sanitize),
        encoding="utf-8",
    )


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None


def pct_return(cur: Optional[float], base: Optional[float]) -> Optional[float]:
    if cur is None or base is None or base == 0:
        return None
    return round((cur / base - 1.0) * 100.0, 2)


def normalize_date_str(x: Any) -> Optional[str]:
    try:
        if x is None:
            return None
        return pd.Timestamp(str(x)).strftime("%Y-%m-%d")
    except Exception:
        return None


def pick_report_date() -> str:
    if REPORT_DATE:
        return REPORT_DATE

    latest = read_json(OUT_DIR / "data" / "top10" / "latest.json")
    if isinstance(latest, dict) and latest.get("date"):
        return str(latest["date"])

    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        raise SystemExit("site/data not found")

    candidates = sorted(
        [
            d.name
            for d in data_dir.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[:4].isdigit()
        ],
        reverse=True,
    )

    if not candidates:
        raise SystemExit("no date directories under site/data")

    return candidates[0]


def load_top10(date: str) -> List[Dict[str, Any]]:
    paths = [
        OUT_DIR / "data" / date / "top10.json",
        OUT_DIR / "data" / "top10" / "latest.json",
    ]

    for path in paths:
        j = read_json(path)
        if not j:
            continue

        payload = j.get("items", j) if isinstance(j, dict) else j
        if isinstance(payload, list):
            return [x for x in payload[:10] if isinstance(x, dict)]

    return []


def load_registry() -> Dict[str, Any]:
    j = read_json(REGISTRY_PATH)
    if isinstance(j, dict) and isinstance(j.get("signals"), list):
        return j

    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "updated_at": None,
        "signals": [],
    }


def signal_id(signal_date: str, symbol: str) -> str:
    return f"{signal_date}_{symbol.upper()}"


def classify_profile(comps: Dict[str, Any]) -> str:
    vol = to_float(comps.get("volume_anomaly")) or 0.0
    comp = to_float(comps.get("compression_release")) or 0.0
    trends = to_float(comps.get("trends_breakout")) or 0.0
    news = to_float(comps.get("news")) or 0.0

    strong = []
    if vol >= 0.75:
        strong.append("Volume")
    if comp >= 0.75:
        strong.append("Compression")
    if trends >= 0.75:
        strong.append("Trends")
    if news >= 0.75:
        strong.append("News")

    if strong:
        return " + ".join(strong)

    good = []
    if vol >= 0.55:
        good.append("Volume")
    if comp >= 0.55:
        good.append("Compression")
    if trends >= 0.55:
        good.append("Trends")
    if news >= 0.55:
        good.append("News")

    if good:
        return "Good " + " + ".join(good)

    return "Mixed / Weak"


def rank_bucket(rank: Optional[int]) -> str:
    if rank == 1:
        return "#1"
    if rank is not None and 2 <= rank <= 3:
        return "#2-3"
    if rank is not None and 4 <= rank <= 10:
        return "#4-10"
    return "Other"


def make_new_signal(item: Dict[str, Any], signal_date: str, rank: int) -> Dict[str, Any]:
    symbol = str(item.get("symbol", "")).strip().upper()
    name = str(item.get("name", "")).strip()

    comps = item.get("score_components") or {}
    weights = item.get("score_weights") or {}

    final01 = to_float(item.get("final_score_0_1")) or 0.0
    score_pts = item.get("score_pts")
    try:
        score_pts_int = int(score_pts) if score_pts is not None else int(round(final01 * 1000))
    except Exception:
        score_pts_int = int(round(final01 * 1000))

    return {
        "id": signal_id(signal_date, symbol),
        "signal_date": signal_date,
        "symbol": symbol,
        "name": name,
        "rank": rank,
        "rank_bucket": rank_bucket(rank),
        "score_pts": score_pts_int,
        "final_score_0_1": round(final01, 6),
        "score_components": {
            "volume_anomaly": to_float(comps.get("volume_anomaly")) or 0.0,
            "compression_release": to_float(comps.get("compression_release", comps.get("dii"))) or 0.0,
            "trends_breakout": to_float(comps.get("trends_breakout")) or 0.0,
            "news": to_float(comps.get("news")) or 0.0,
        },
        "score_weights": {
            "volume_anomaly": to_float(weights.get("volume_anomaly")) or 0.0,
            "compression_release": to_float(weights.get("compression_release", weights.get("dii"))) or 0.0,
            "trends_breakout": to_float(weights.get("trends_breakout")) or 0.0,
            "news": to_float(weights.get("news")) or 0.0,
        },
        "profile": classify_profile(comps),
        "signal_close": None,
        "entry": {
            "method": "next_open",
            "entry_date": None,
            "entry_price": None,
            "gap_pct": None,
        },
        "outcome": {
            "d5_return_pct": None,
            "d10_return_pct": None,
            "d20_return_pct": None,
            "max_gain_20d_pct": None,
            "max_drawdown_20d_pct": None,
            "status": "pending",
            "last_updated": None,
        },
        "source_snapshot": {
            "price_delta_1d": to_float(item.get("price_delta_1d")),
            "price_delta_1w": to_float(item.get("price_delta_1w")),
            "price_delta_1m": to_float(item.get("price_delta_1m")),
        },
    }


def add_today_signals(registry: Dict[str, Any], date: str, top10: List[Dict[str, Any]]) -> int:
    existing_ids = {str(s.get("id")) for s in registry.get("signals", []) if isinstance(s, dict)}
    added = 0

    for i, item in enumerate(top10, start=1):
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym:
            continue

        sid = signal_id(date, sym)
        if sid in existing_ids:
            continue

        registry["signals"].append(make_new_signal(item, date, i))
        existing_ids.add(sid)
        added += 1

    return added


def first_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")

    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        s = x.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    return pd.Series(dtype=float)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = pd.DataFrame(index=pd.to_datetime(df.index))
    targets = ["open", "high", "low", "close", "volume"]

    if isinstance(df.columns, pd.MultiIndex):
        chosen: Dict[str, Any] = {}

        for col in df.columns:
            parts = [str(c).strip().lower() for c in col if c is not None]
            for t in targets:
                if t in parts and t not in chosen:
                    chosen[t] = col

        for t in targets:
            if t in chosen:
                out[t.capitalize()] = first_series(df.loc[:, chosen[t]]).to_numpy()
            else:
                out[t.capitalize()] = np.nan
    else:
        src_map = {str(c).strip().lower(): c for c in df.columns}
        for t in targets:
            src = src_map.get(t)
            if src is not None:
                out[t.capitalize()] = first_series(df[src]).to_numpy()
            else:
                out[t.capitalize()] = np.nan

    out = out[["Open", "High", "Low", "Close", "Volume"]].dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def fetch_history(symbol: str, months: int = 18) -> Optional[pd.DataFrame]:
    if yf is None:
        log("WARN", "yfinance is not installed")
        return None

    for attempt in range(3):
        try:
            raw = yf.download(
                symbol,
                period=f"{months}mo",
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False,
            )

            if raw is None or raw.empty:
                raise ValueError("empty dataframe")

            df = normalize_ohlcv(raw)
            if len(df) < 30:
                raise ValueError("not enough clean rows")

            return df

        except Exception as e:
            log("WARN", f"{symbol}: fetch_history attempt {attempt + 1} failed: {e}")
            time.sleep(1.2 + attempt * 0.8)

    return None


def locate_signal_index(df: pd.DataFrame, signal_date: str) -> Optional[int]:
    if df is None or df.empty:
        return None

    target = pd.Timestamp(signal_date)
    dates = pd.to_datetime(df.index).tz_localize(None)

    # 原則：signal_date当日の足
    exact = np.where(dates == target)[0]
    if len(exact) > 0:
        return int(exact[-1])

    # 念のため、signal_date以前の最後の取引日を使う
    before = np.where(dates <= target)[0]
    if len(before) > 0:
        return int(before[-1])

    return None


def update_signal_with_history(signal: Dict[str, Any], df: pd.DataFrame) -> bool:
    signal_date = str(signal.get("signal_date", ""))
    if not signal_date:
        return False

    sig_idx = locate_signal_index(df, signal_date)
    if sig_idx is None:
        return False

    changed = False

    signal_close = to_float(df["Close"].iloc[sig_idx])
    if signal.get("signal_close") is None and signal_close is not None:
        signal["signal_close"] = round(signal_close, 4)
        changed = True

    entry_idx = sig_idx + 1
    if entry_idx >= len(df):
        return changed

    entry_date = pd.Timestamp(df.index[entry_idx]).strftime("%Y-%m-%d")
    entry_price = to_float(df["Open"].iloc[entry_idx])

    entry = signal.setdefault("entry", {})
    if entry.get("entry_price") is None and entry_price is not None:
        entry["method"] = "next_open"
        entry["entry_date"] = entry_date
        entry["entry_price"] = round(entry_price, 4)
        entry["gap_pct"] = pct_return(entry_price, signal_close)
        changed = True

    entry_price = to_float(entry.get("entry_price"))
    if entry_price is None:
        return changed

    outcome = signal.setdefault("outcome", {})

    completed_any = False
    for h in HORIZONS:
        target_idx = entry_idx + h
        key = f"d{h}_return_pct"

        if target_idx < len(df):
            target_close = to_float(df["Close"].iloc[target_idx])
            ret = pct_return(target_close, entry_price)
            if ret is not None and outcome.get(key) is None:
                outcome[key] = ret
                completed_any = True
                changed = True

    max_window_end = min(entry_idx + 20, len(df) - 1)
    if max_window_end >= entry_idx:
        high_window = df["High"].iloc[entry_idx:max_window_end + 1]
        low_window = df["Low"].iloc[entry_idx:max_window_end + 1]

        max_high = to_float(high_window.max())
        min_low = to_float(low_window.min())

        max_gain = pct_return(max_high, entry_price)
        max_dd = pct_return(min_low, entry_price)

        if max_gain is not None:
            if outcome.get("max_gain_20d_pct") is None or max_window_end >= entry_idx + 20:
                outcome["max_gain_20d_pct"] = max_gain
                changed = True

        if max_dd is not None:
            if outcome.get("max_drawdown_20d_pct") is None or max_window_end >= entry_idx + 20:
                outcome["max_drawdown_20d_pct"] = max_dd
                changed = True

    if outcome.get("d20_return_pct") is not None:
        status = "completed_20d"
    elif outcome.get("d10_return_pct") is not None:
        status = "completed_10d"
    elif outcome.get("d5_return_pct") is not None:
        status = "completed_5d"
    else:
        status = "pending"

    if outcome.get("status") != status:
        outcome["status"] = status
        changed = True

    if changed or completed_any:
        outcome["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    return changed


def needs_update(signal: Dict[str, Any]) -> bool:
    outcome = signal.get("outcome") or {}
    return outcome.get("status") != "completed_20d"


def flatten_recent_outcomes(signals: List[Dict[str, Any]], limit: int = 80) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for s in signals:
        outcome = s.get("outcome") or {}
        if (
            outcome.get("d5_return_pct") is None
            and outcome.get("d10_return_pct") is None
            and outcome.get("d20_return_pct") is None
        ):
            continue

        rows.append({
            "id": s.get("id"),
            "signal_date": s.get("signal_date"),
            "symbol": s.get("symbol"),
            "name": s.get("name"),
            "rank": s.get("rank"),
            "rank_bucket": s.get("rank_bucket"),
            "profile": s.get("profile"),
            "score_pts": s.get("score_pts"),
            "entry_date": (s.get("entry") or {}).get("entry_date"),
            "entry_price": (s.get("entry") or {}).get("entry_price"),
            "gap_pct": (s.get("entry") or {}).get("gap_pct"),
            "d5_return_pct": outcome.get("d5_return_pct"),
            "d10_return_pct": outcome.get("d10_return_pct"),
            "d20_return_pct": outcome.get("d20_return_pct"),
            "max_gain_20d_pct": outcome.get("max_gain_20d_pct"),
            "max_drawdown_20d_pct": outcome.get("max_drawdown_20d_pct"),
            "status": outcome.get("status"),
        })

    rows.sort(key=lambda r: str(r.get("signal_date") or ""), reverse=True)
    return rows[:limit]


def mean(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return round(float(np.mean(vals)), 2)


def median(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return round(float(np.median(vals)), 2)


def win_rate(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return round(sum(1 for x in vals if x > 0) / len(vals), 4)


def summarize_group(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    d5 = [to_float(r.get("d5_return_pct")) for r in rows]
    d10 = [to_float(r.get("d10_return_pct")) for r in rows]
    d20 = [to_float(r.get("d20_return_pct")) for r in rows]
    gain = [to_float(r.get("max_gain_20d_pct")) for r in rows]
    dd = [to_float(r.get("max_drawdown_20d_pct")) for r in rows]

    return {
        "label": label,
        "count": len(rows),
        "completed_5d": sum(1 for x in d5 if x is not None),
        "completed_10d": sum(1 for x in d10 if x is not None),
        "completed_20d": sum(1 for x in d20 if x is not None),
        "win_rate_5d": win_rate([x for x in d5 if x is not None]),
        "win_rate_10d": win_rate([x for x in d10 if x is not None]),
        "win_rate_20d": win_rate([x for x in d20 if x is not None]),
        "avg_return_5d": mean([x for x in d5 if x is not None]),
        "avg_return_10d": mean([x for x in d10 if x is not None]),
        "avg_return_20d": mean([x for x in d20 if x is not None]),
        "median_return_20d": median([x for x in d20 if x is not None]),
        "avg_max_gain_20d": mean([x for x in gain if x is not None]),
        "avg_max_drawdown_20d": mean([x for x in dd if x is not None]),
    }


def build_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = flatten_recent_outcomes(signals, limit=100000)

    d5_rows = [r for r in rows if r.get("d5_return_pct") is not None]
    d10_rows = [r for r in rows if r.get("d10_return_pct") is not None]
    d20_rows = [r for r in rows if r.get("d20_return_pct") is not None]

    d5 = [to_float(r.get("d5_return_pct")) for r in d5_rows]
    d10 = [to_float(r.get("d10_return_pct")) for r in d10_rows]
    d20 = [to_float(r.get("d20_return_pct")) for r in d20_rows]
    gain = [to_float(r.get("max_gain_20d_pct")) for r in rows if r.get("max_gain_20d_pct") is not None]
    dd = [to_float(r.get("max_drawdown_20d_pct")) for r in rows if r.get("max_drawdown_20d_pct") is not None]

    buckets: List[Dict[str, Any]] = []
    for b in ["#1", "#2-3", "#4-10", "Other"]:
        br = [r for r in rows if r.get("rank_bucket") == b]
        if br:
            buckets.append(summarize_group(br, b))

    profile_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        p = str(r.get("profile") or "Unknown")
        profile_map.setdefault(p, []).append(r)

    profiles = [
        summarize_group(v, k)
        for k, v in profile_map.items()
        if len(v) >= 2
    ]
    profiles.sort(
        key=lambda x: (
            x.get("completed_20d") or 0,
            x.get("avg_return_20d") if x.get("avg_return_20d") is not None else -999,
        ),
        reverse=True,
    )

    as_of = None
    signal_dates = [normalize_date_str(s.get("signal_date")) for s in signals]
    signal_dates = [d for d in signal_dates if d]
    if signal_dates:
        as_of = max(signal_dates)

    return {
        "as_of": as_of,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "total_signals": len(signals),
        "completed_5d": len(d5_rows),
        "completed_10d": len(d10_rows),
        "completed_20d": len(d20_rows),
        "win_rate_5d": win_rate([x for x in d5 if x is not None]),
        "win_rate_10d": win_rate([x for x in d10 if x is not None]),
        "win_rate_20d": win_rate([x for x in d20 if x is not None]),
        "avg_return_5d": mean([x for x in d5 if x is not None]),
        "avg_return_10d": mean([x for x in d10 if x is not None]),
        "avg_return_20d": mean([x for x in d20 if x is not None]),
        "median_return_20d": median([x for x in d20 if x is not None]),
        "avg_max_gain_20d": mean([x for x in gain if x is not None]),
        "avg_max_drawdown_20d": mean([x for x in dd if x is not None]),
        "rank_buckets": buckets,
        "profiles": profiles[:20],
    }


def main() -> None:
    date = pick_report_date()
    top10 = load_top10(date)

    if not top10:
        raise SystemExit(f"No top10 items found for {date}")

    registry = load_registry()
    added = add_today_signals(registry, date, top10)

    signals = registry.get("signals", [])
    if not isinstance(signals, list):
        raise SystemExit("registry signals is not a list")

    pending_symbols = sorted({
        str(s.get("symbol", "")).upper()
        for s in signals
        if isinstance(s, dict) and s.get("symbol") and needs_update(s)
    })

    log("INFO", f"Added today signals: {added}")
    log("INFO", f"Pending symbols to update: {len(pending_symbols)}")

    history_cache: Dict[str, Optional[pd.DataFrame]] = {}

    for sym in pending_symbols:
        history_cache[sym] = fetch_history(sym, months=18)

    changed = 0
    for s in signals:
        if not isinstance(s, dict):
            continue

        sym = str(s.get("symbol", "")).upper()
        if not sym or not needs_update(s):
            continue

        df = history_cache.get(sym)
        if df is None or df.empty:
            continue

        if update_signal_with_history(s, df):
            changed += 1

    registry["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    registry["signals"] = signals

    recent = flatten_recent_outcomes(signals, limit=80)
    summary = build_summary(signals)

    write_json(REGISTRY_PATH, registry)
    write_json(OUTCOMES_PATH, {"items": recent})
    write_json(SUMMARY_PATH, summary)

    log("INFO", f"Wrote registry: {REGISTRY_PATH}")
    log("INFO", f"Wrote outcomes: {OUTCOMES_PATH}")
    log("INFO", f"Wrote summary: {SUMMARY_PATH}")
    log("INFO", f"Signals changed: {changed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in track_signal_outcomes: {e}")
        sys.exit(1)