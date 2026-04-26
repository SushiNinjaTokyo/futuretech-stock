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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
REPORT_DATE = os.getenv("REPORT_DATE")

REGISTRY_PATH = OUT_DIR / "data" / "signals" / "registry.json"
SUMMARY_PATH = OUT_DIR / "data" / "signals" / "summary_latest.json"
OUTCOMES_PATH = OUT_DIR / "data" / "signals" / "outcomes_latest.json"

# Top3だけ追跡
TRACK_TOP_N = 3

# 取引日ベース
HORIZONS = {
    "1w": 5,
    "1m": 20,
    "3m": 63,
}


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


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
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
        "policy": {
            "track_top_n": TRACK_TOP_N,
            "entry_method": "next_open",
            "horizons": HORIZONS,
            "stop_tracking_after": "3m",
        },
        "signals": [],
    }


def signal_id(signal_date: str, symbol: str) -> str:
    return f"{signal_date}_{symbol.upper()}"


def rank_bucket(rank: Optional[int]) -> str:
    if rank == 1:
        return "#1"
    if rank is not None and 2 <= rank <= 3:
        return "#2-3"
    return "Other"


def classify_profile(comps: Dict[str, Any]) -> str:
    vol = to_float(comps.get("volume_anomaly")) or 0.0
    comp = to_float(comps.get("compression_release", comps.get("dii"))) or 0.0
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


def normalize_components(item: Dict[str, Any]) -> Dict[str, float]:
    comps = item.get("score_components") or {}
    return {
        "volume_anomaly": to_float(comps.get("volume_anomaly")) or 0.0,
        "compression_release": to_float(comps.get("compression_release", comps.get("dii"))) or 0.0,
        "trends_breakout": to_float(comps.get("trends_breakout")) or 0.0,
        "news": to_float(comps.get("news")) or 0.0,
    }


def normalize_weights(item: Dict[str, Any]) -> Dict[str, float]:
    weights = item.get("score_weights") or {}
    return {
        "volume_anomaly": to_float(weights.get("volume_anomaly")) or 0.0,
        "compression_release": to_float(weights.get("compression_release", weights.get("dii"))) or 0.0,
        "trends_breakout": to_float(weights.get("trends_breakout")) or 0.0,
        "news": to_float(weights.get("news")) or 0.0,
    }


def make_new_signal(item: Dict[str, Any], signal_date: str, rank: int) -> Dict[str, Any]:
    symbol = str(item.get("symbol", "")).strip().upper()
    name = str(item.get("name", "")).strip()

    comps = normalize_components(item)
    weights = normalize_weights(item)

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
        "score_components": comps,
        "score_weights": weights,
        "profile": classify_profile(comps),
        "signal_close": None,
        "entry": {
            "method": "next_open",
            "entry_date": None,
            "entry_price": None,
            "gap_pct": None,
        },
        "outcome": {
            "return_1w_pct": None,
            "return_1m_pct": None,
            "return_3m_pct": None,
            "current_return_pct": None,
            "current_close": None,
            "current_date": None,
            "max_gain_since_entry_pct": None,
            "max_drawdown_since_entry_pct": None,
            "status": "pending_entry",
            "last_updated": None,

            # 旧Backtestテンプレート互換用
            "d5_return_pct": None,
            "d10_return_pct": None,
            "d20_return_pct": None,
            "max_gain_20d_pct": None,
            "max_drawdown_20d_pct": None,
        },
        "source_snapshot": {
            "price_delta_1d": to_float(item.get("price_delta_1d")),
            "price_delta_1w": to_float(item.get("price_delta_1w")),
            "price_delta_1m": to_float(item.get("price_delta_1m")),
        },
    }


def cleanup_existing_registry_to_top3(registry: Dict[str, Any]) -> int:
    """
    既にTop10で登録済みの初期ログをTop3だけに整理する。
    まだ初期段階なので、rank > 3 は削除してよい設計。
    """
    signals = registry.get("signals", [])
    if not isinstance(signals, list):
        registry["signals"] = []
        return 0

    before = len(signals)
    cleaned = []

    for s in signals:
        if not isinstance(s, dict):
            continue
        r = to_int(s.get("rank"))
        if r is not None and r <= TRACK_TOP_N:
            s["rank_bucket"] = rank_bucket(r)
            cleaned.append(s)

    registry["signals"] = cleaned
    return before - len(cleaned)


def add_today_signals(registry: Dict[str, Any], date: str, top10: List[Dict[str, Any]]) -> int:
    existing_ids = {str(s.get("id")) for s in registry.get("signals", []) if isinstance(s, dict)}
    added = 0

    for i, item in enumerate(top10[:TRACK_TOP_N], start=1):
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

    exact = np.where(dates == target)[0]
    if len(exact) > 0:
        return int(exact[-1])

    before = np.where(dates <= target)[0]
    if len(before) > 0:
        return int(before[-1])

    return None


def get_status(outcome: Dict[str, Any], has_entry: bool) -> str:
    if not has_entry:
        return "pending_entry"
    if outcome.get("return_3m_pct") is not None:
        return "completed_3m"
    if outcome.get("return_1m_pct") is not None:
        return "completed_1m"
    if outcome.get("return_1w_pct") is not None:
        return "completed_1w"
    return "active"


def needs_update(signal: Dict[str, Any]) -> bool:
    outcome = signal.get("outcome") or {}
    return outcome.get("status") != "completed_3m"


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
        outcome = signal.setdefault("outcome", {})
        if outcome.get("status") != "pending_entry":
            outcome["status"] = "pending_entry"
            changed = True
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

    # 1W / 1M / 3M の固定評価
    for label, h in HORIZONS.items():
        target_idx = entry_idx + h
        key = f"return_{label}_pct"

        if target_idx < len(df):
            target_close = to_float(df["Close"].iloc[target_idx])
            ret = pct_return(target_close, entry_price)
            if ret is not None and outcome.get(key) is None:
                outcome[key] = ret
                changed = True

    # current は3M到達までは最新値、3M到達後は3M時点で固定
    if outcome.get("return_3m_pct") is not None and entry_idx + HORIZONS["3m"] < len(df):
        current_idx = entry_idx + HORIZONS["3m"]
    else:
        current_idx = len(df) - 1

    current_close = to_float(df["Close"].iloc[current_idx])
    current_date = pd.Timestamp(df.index[current_idx]).strftime("%Y-%m-%d")
    current_return = pct_return(current_close, entry_price)

    if current_close is not None and outcome.get("current_close") != round(current_close, 4):
        outcome["current_close"] = round(current_close, 4)
        changed = True

    if outcome.get("current_date") != current_date:
        outcome["current_date"] = current_date
        changed = True

    if current_return is not None and outcome.get("current_return_pct") != current_return:
        outcome["current_return_pct"] = current_return
        changed = True

    # Max Gain / Max Drawdown
    # 3M到達前はentryから現在まで、3M到達後はentryから3Mまでで固定
    max_window_end = current_idx
    if max_window_end >= entry_idx:
        high_window = df["High"].iloc[entry_idx:max_window_end + 1]
        low_window = df["Low"].iloc[entry_idx:max_window_end + 1]

        max_high = to_float(high_window.max())
        min_low = to_float(low_window.min())

        max_gain = pct_return(max_high, entry_price)
        max_dd = pct_return(min_low, entry_price)

        if max_gain is not None and outcome.get("max_gain_since_entry_pct") != max_gain:
            outcome["max_gain_since_entry_pct"] = max_gain
            changed = True

        if max_dd is not None and outcome.get("max_drawdown_since_entry_pct") != max_dd:
            outcome["max_drawdown_since_entry_pct"] = max_dd
            changed = True

    # 旧Backtestテンプレート互換
    # d5 = 1W, d20 = 1M として出す。d10は廃止扱いでNone。
    if outcome.get("d5_return_pct") != outcome.get("return_1w_pct"):
        outcome["d5_return_pct"] = outcome.get("return_1w_pct")
        changed = True

    if outcome.get("d10_return_pct") is not None:
        outcome["d10_return_pct"] = None
        changed = True

    if outcome.get("d20_return_pct") != outcome.get("return_1m_pct"):
        outcome["d20_return_pct"] = outcome.get("return_1m_pct")
        changed = True

    if outcome.get("max_gain_20d_pct") != outcome.get("max_gain_since_entry_pct"):
        outcome["max_gain_20d_pct"] = outcome.get("max_gain_since_entry_pct")
        changed = True

    if outcome.get("max_drawdown_20d_pct") != outcome.get("max_drawdown_since_entry_pct"):
        outcome["max_drawdown_20d_pct"] = outcome.get("max_drawdown_since_entry_pct")
        changed = True

    status = get_status(outcome, has_entry=True)
    if outcome.get("status") != status:
        outcome["status"] = status
        changed = True

    if changed:
        outcome["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    return changed


def flatten_recent_outcomes(signals: List[Dict[str, Any]], limit: int = 120) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for s in signals:
        outcome = s.get("outcome") or {}
        entry = s.get("entry") or {}

        # entryがまだないものも表示対象にする
        rows.append({
            "id": s.get("id"),
            "signal_date": s.get("signal_date"),
            "symbol": s.get("symbol"),
            "name": s.get("name"),
            "rank": s.get("rank"),
            "rank_bucket": s.get("rank_bucket"),
            "profile": s.get("profile"),
            "score_pts": s.get("score_pts"),
            "entry_date": entry.get("entry_date"),
            "entry_price": entry.get("entry_price"),
            "gap_pct": entry.get("gap_pct"),

            "return_1w_pct": outcome.get("return_1w_pct"),
            "return_1m_pct": outcome.get("return_1m_pct"),
            "return_3m_pct": outcome.get("return_3m_pct"),
            "current_return_pct": outcome.get("current_return_pct"),
            "current_close": outcome.get("current_close"),
            "current_date": outcome.get("current_date"),
            "max_gain_since_entry_pct": outcome.get("max_gain_since_entry_pct"),
            "max_drawdown_since_entry_pct": outcome.get("max_drawdown_since_entry_pct"),
            "status": outcome.get("status"),

            # 旧Backtestテンプレート互換
            "d5_return_pct": outcome.get("d5_return_pct"),
            "d10_return_pct": outcome.get("d10_return_pct"),
            "d20_return_pct": outcome.get("d20_return_pct"),
            "max_gain_20d_pct": outcome.get("max_gain_20d_pct"),
            "max_drawdown_20d_pct": outcome.get("max_drawdown_20d_pct"),
        })

    rows.sort(
        key=lambda r: (
            str(r.get("signal_date") or ""),
            int(r.get("rank") or 999),
        ),
        reverse=True,
    )
    return rows[:limit]


def mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return round(float(np.mean(vals)), 2)


def median(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return round(float(np.median(vals)), 2)


def win_rate(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return round(sum(1 for x in vals if x > 0) / len(vals), 4)


def summarize_group(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    r1w = [to_float(r.get("return_1w_pct")) for r in rows]
    r1m = [to_float(r.get("return_1m_pct")) for r in rows]
    r3m = [to_float(r.get("return_3m_pct")) for r in rows]
    cur = [to_float(r.get("current_return_pct")) for r in rows]
    gain = [to_float(r.get("max_gain_since_entry_pct")) for r in rows]
    dd = [to_float(r.get("max_drawdown_since_entry_pct")) for r in rows]

    return {
        "label": label,
        "count": len(rows),

        "completed_1w": sum(1 for x in r1w if x is not None),
        "completed_1m": sum(1 for x in r1m if x is not None),
        "completed_3m": sum(1 for x in r3m if x is not None),

        "win_rate_1w": win_rate(r1w),
        "win_rate_1m": win_rate(r1m),
        "win_rate_3m": win_rate(r3m),

        "avg_return_1w": mean(r1w),
        "avg_return_1m": mean(r1m),
        "avg_return_3m": mean(r3m),
        "median_return_3m": median(r3m),

        "avg_current_return": mean(cur),
        "avg_max_gain": mean(gain),
        "avg_max_drawdown": mean(dd),

        # 旧テンプレート互換
        "completed_5d": sum(1 for x in r1w if x is not None),
        "completed_10d": 0,
        "completed_20d": sum(1 for x in r1m if x is not None),
        "win_rate_5d": win_rate(r1w),
        "win_rate_10d": None,
        "win_rate_20d": win_rate(r1m),
        "avg_return_5d": mean(r1w),
        "avg_return_10d": None,
        "avg_return_20d": mean(r1m),
        "median_return_20d": median(r1m),
        "avg_max_gain_20d": mean(gain),
        "avg_max_drawdown_20d": mean(dd),
    }


def build_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = flatten_recent_outcomes(signals, limit=100000)

    r1w = [to_float(r.get("return_1w_pct")) for r in rows]
    r1m = [to_float(r.get("return_1m_pct")) for r in rows]
    r3m = [to_float(r.get("return_3m_pct")) for r in rows]
    cur = [to_float(r.get("current_return_pct")) for r in rows]
    gain = [to_float(r.get("max_gain_since_entry_pct")) for r in rows]
    dd = [to_float(r.get("max_drawdown_since_entry_pct")) for r in rows]

    completed_1w = sum(1 for x in r1w if x is not None)
    completed_1m = sum(1 for x in r1m if x is not None)
    completed_3m = sum(1 for x in r3m if x is not None)

    buckets: List[Dict[str, Any]] = []
    for b in ["#1", "#2-3", "Other"]:
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
            x.get("completed_3m") or 0,
            x.get("avg_return_3m") if x.get("avg_return_3m") is not None else -999,
            x.get("count") or 0,
        ),
        reverse=True,
    )

    signal_dates = [normalize_date_str(s.get("signal_date")) for s in signals]
    signal_dates = [d for d in signal_dates if d]
    as_of = max(signal_dates) if signal_dates else None

    active_count = sum(
        1
        for s in signals
        if (s.get("outcome") or {}).get("status") != "completed_3m"
    )

    summary = {
        "as_of": as_of,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "tracking_policy": {
            "track_top_n": TRACK_TOP_N,
            "entry_method": "next_open",
            "horizons": {
                "1w_trading_days": HORIZONS["1w"],
                "1m_trading_days": HORIZONS["1m"],
                "3m_trading_days": HORIZONS["3m"],
            },
            "stop_tracking_after": "3m",
        },

        "total_signals": len(signals),
        "active_signals": active_count,

        "completed_1w": completed_1w,
        "completed_1m": completed_1m,
        "completed_3m": completed_3m,

        "win_rate_1w": win_rate(r1w),
        "win_rate_1m": win_rate(r1m),
        "win_rate_3m": win_rate(r3m),

        "avg_return_1w": mean(r1w),
        "avg_return_1m": mean(r1m),
        "avg_return_3m": mean(r3m),
        "median_return_3m": median(r3m),

        "avg_current_return": mean(cur),
        "avg_max_gain": mean(gain),
        "avg_max_drawdown": mean(dd),

        "rank_buckets": buckets,
        "profiles": profiles[:20],

        # 旧Backtestテンプレート互換
        "completed_5d": completed_1w,
        "completed_10d": 0,
        "completed_20d": completed_1m,
        "win_rate_5d": win_rate(r1w),
        "win_rate_10d": None,
        "win_rate_20d": win_rate(r1m),
        "avg_return_5d": mean(r1w),
        "avg_return_10d": None,
        "avg_return_20d": mean(r1m),
        "median_return_20d": median(r1m),
        "avg_max_gain_20d": mean(gain),
        "avg_max_drawdown_20d": mean(dd),
    }

    return summary


def main() -> None:
    date = pick_report_date()
    top10 = load_top10(date)

    if not top10:
        raise SystemExit(f"No top10 items found for {date}")

    registry = load_registry()

    removed = cleanup_existing_registry_to_top3(registry)
    if removed:
        log("INFO", f"Removed non-Top3 existing signals: {removed}")

    added = add_today_signals(registry, date, top10)

    signals = registry.get("signals", [])
    if not isinstance(signals, list):
        raise SystemExit("registry signals is not a list")

    pending_symbols = sorted({
        str(s.get("symbol", "")).upper()
        for s in signals
        if isinstance(s, dict) and s.get("symbol") and needs_update(s)
    })

    log("INFO", f"Added today Top{TRACK_TOP_N} signals: {added}")
    log("INFO", f"Symbols to update until 3M completion: {len(pending_symbols)}")

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
    registry["policy"] = {
        "track_top_n": TRACK_TOP_N,
        "entry_method": "next_open",
        "horizons": HORIZONS,
        "stop_tracking_after": "3m",
    }
    registry["signals"] = signals

    recent = flatten_recent_outcomes(signals, limit=120)
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