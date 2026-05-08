#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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

REGISTRY_PATH = OUT_DIR / "data" / "signals" / "registry.json"
SUMMARY_PATH = OUT_DIR / "data" / "signals" / "summary_latest.json"
OUTCOMES_PATH = OUT_DIR / "data" / "signals" / "outcomes_latest.json"

POLICY_VERSION = "score_filtered_v1"

MIN_TOP3_SCORE = int(os.getenv("SIGNAL_MIN_TOP3_SCORE", "750"))
MIN_VOLUME_SCORE = float(os.getenv("SIGNAL_MIN_VOLUME_SCORE", "0.55"))
MIN_VOLUME_COMBO_SCORE = int(os.getenv("SIGNAL_MIN_VOLUME_COMBO_SCORE", "700"))
MIN_ANY_RANK_SCORE = int(os.getenv("SIGNAL_MIN_ANY_RANK_SCORE", "800"))
REPORT_LEGACY_MIN_SCORE = int(os.getenv("SIGNAL_REPORT_LEGACY_MIN_SCORE", "750"))

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


def valid_date_dir_name(name: str) -> bool:
    if len(name) != 10:
        return False
    try:
        pd.Timestamp(name)
        return name[:4].isdigit() and name[4] == "-" and name[7] == "-"
    except Exception:
        return False


def signal_id(signal_date: str, symbol: str) -> str:
    return f"{signal_date}_{symbol.upper()}"


def rank_bucket(rank: Optional[int]) -> str:
    if rank == 1:
        return "#1"
    if rank is not None and 2 <= rank <= 3:
        return "#2-3"
    if rank is not None and 4 <= rank <= 10:
        return "#4-10"
    return "Other"


def score_bucket(score_pts: Optional[int]) -> str:
    if score_pts is None:
        return "Unknown"
    if score_pts >= 850:
        return "850+"
    if score_pts >= 800:
        return "800-849"
    if score_pts >= 750:
        return "750-799"
    if score_pts >= 700:
        return "700-749"
    return "<700"


def current_policy() -> Dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "entry_method": "next_open",
        "horizons": {
            "1w_trading_days": HORIZONS["1w"],
            "1m_trading_days": HORIZONS["1m"],
            "3m_trading_days": HORIZONS["3m"],
        },
        "stop_tracking_after": "3m",
        "registration_rules": [
            f"rank <= 3 and score_pts >= {MIN_TOP3_SCORE}",
            f"score_pts >= {MIN_VOLUME_COMBO_SCORE} and volume_anomaly >= {MIN_VOLUME_SCORE}",
            f"rank > 3 and score_pts >= {MIN_ANY_RANK_SCORE}",
        ],
        "legacy_reporting": {
            "keep_existing_registry": True,
            "legacy_min_score_for_report": REPORT_LEGACY_MIN_SCORE,
        },
    }


def backup_registry() -> Optional[Path]:
    if not REGISTRY_PATH.exists():
        log("INFO", "No existing registry.json to backup")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = REGISTRY_PATH.parent / f"registry.backup_{ts}.json"
    ensure_dir(backup_path.parent)
    shutil.copy2(REGISTRY_PATH, backup_path)
    log("INFO", f"Backed up registry: {backup_path}")
    return backup_path


def load_registry() -> Dict[str, Any]:
    j = read_json(REGISTRY_PATH)
    if isinstance(j, dict) and isinstance(j.get("signals"), list):
        return j

    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "updated_at": None,
        "policy": current_policy(),
        "signals": [],
    }


def load_top10_for_date(date: str) -> List[Dict[str, Any]]:
    path = OUT_DIR / "data" / date / "top10.json"
    j = read_json(path)
    if not j:
        return []

    payload = j.get("items", j) if isinstance(j, dict) else j
    if not isinstance(payload, list):
        return []

    return [x for x in payload[:10] if isinstance(x, dict)]


def list_rebuild_dates(start_date: str, end_date: Optional[str] = None) -> List[str]:
    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        raise SystemExit("site/data not found")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else None

    dates: List[str] = []
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        if not valid_date_dir_name(d.name):
            continue
        if not (d / "top10.json").exists():
            continue

        cur = pd.Timestamp(d.name)
        if cur < start_ts:
            continue
        if end_ts is not None and cur > end_ts:
            continue
        dates.append(d.name)

    dates.sort()
    return dates


def preserve_signals_before_start(registry: Dict[str, Any], start_date: str) -> List[Dict[str, Any]]:
    start_ts = pd.Timestamp(start_date)
    preserved: List[Dict[str, Any]] = []

    for s in registry.get("signals", []):
        if not isinstance(s, dict):
            continue

        d = normalize_date_str(s.get("signal_date"))
        if not d:
            continue

        if pd.Timestamp(d) < start_ts:
            preserved.append(s)

    return preserved


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


def get_item_rank(item: Dict[str, Any], fallback_rank: int) -> int:
    r = to_int(item.get("rank"))
    if r is not None:
        return r
    return fallback_rank


def get_score_pts(item: Dict[str, Any]) -> int:
    final01 = to_float(item.get("final_score_0_1")) or 0.0
    score_pts = item.get("score_pts")
    try:
        return int(score_pts) if score_pts is not None else int(round(final01 * 1000))
    except Exception:
        return int(round(final01 * 1000))


def get_volume_score(item: Dict[str, Any]) -> float:
    comps = normalize_components(item)
    return float(comps.get("volume_anomaly") or 0.0)


def evaluate_signal_eligibility(item: Dict[str, Any], rank: int) -> Tuple[bool, str, str]:
    score_pts = get_score_pts(item)
    volume_anomaly = get_volume_score(item)

    if rank <= 3 and score_pts >= MIN_TOP3_SCORE:
        return True, "top3_score_750", "Signal"

    if rank > 3 and score_pts >= MIN_ANY_RANK_SCORE:
        return True, "score_800_any_rank", "Strong Signal"

    if score_pts >= MIN_VOLUME_COMBO_SCORE and volume_anomaly >= MIN_VOLUME_SCORE:
        return True, "score_700_volume_055", "Volume Signal"

    return False, "not_eligible", "Watch"


def make_new_signal(item: Dict[str, Any], signal_date: str, rank: int, rule: str, quality: str) -> Dict[str, Any]:
    symbol = str(item.get("symbol", "")).strip().upper()
    name = str(item.get("name", "")).strip()

    comps = normalize_components(item)
    weights = normalize_weights(item)

    final01 = to_float(item.get("final_score_0_1")) or 0.0
    score_pts_int = get_score_pts(item)

    return {
        "id": signal_id(signal_date, symbol),
        "signal_date": signal_date,
        "symbol": symbol,
        "name": name,
        "rank": rank,
        "rank_bucket": rank_bucket(rank),
        "score_bucket": score_bucket(score_pts_int),
        "score_pts": score_pts_int,
        "final_score_0_1": round(final01, 6),
        "score_components": comps,
        "score_weights": weights,
        "profile": classify_profile(comps),

        "policy_version": POLICY_VERSION,
        "signal_eligible": True,
        "eligibility_rule": rule,
        "signal_quality": quality,

        "signal_close": None,
        "entry": {
            "method": "next_open",
            "entry_date": None,
            "entry_price": None,
            "gap_pct": None,
        },
        "outcome": empty_outcome(),
        "source_snapshot": {
            "price_delta_1d": to_float(item.get("price_delta_1d")),
            "price_delta_1w": to_float(item.get("price_delta_1w")),
            "price_delta_1m": to_float(item.get("price_delta_1m")),
        },
    }


def empty_outcome() -> Dict[str, Any]:
    return {
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

        # 旧Backtestテンプレート互換
        "d5_return_pct": None,
        "d10_return_pct": None,
        "d20_return_pct": None,
        "max_gain_20d_pct": None,
        "max_drawdown_20d_pct": None,
    }


def enrich_legacy_signal(signal: Dict[str, Any]) -> bool:
    changed = False

    rank = to_int(signal.get("rank"))
    score = to_int(signal.get("score_pts"))
    comps = signal.get("score_components") or {}

    if signal.get("rank_bucket") != rank_bucket(rank):
        signal["rank_bucket"] = rank_bucket(rank)
        changed = True

    if signal.get("score_bucket") != score_bucket(score):
        signal["score_bucket"] = score_bucket(score)
        changed = True

    if "policy_version" not in signal:
        signal["policy_version"] = "legacy_top3"
        changed = True

    if "signal_eligible" not in signal:
        signal["signal_eligible"] = score is not None and score >= REPORT_LEGACY_MIN_SCORE
        changed = True

    if "eligibility_rule" not in signal:
        if signal.get("signal_eligible"):
            signal["eligibility_rule"] = f"legacy_score_{REPORT_LEGACY_MIN_SCORE}+"
        else:
            signal["eligibility_rule"] = "legacy_below_threshold"
        changed = True

    if "signal_quality" not in signal:
        if score is not None and score >= 800:
            signal["signal_quality"] = "Strong Signal"
        elif score is not None and score >= 750:
            signal["signal_quality"] = "Signal"
        else:
            signal["signal_quality"] = "Legacy / Weak"
        changed = True

    if not signal.get("profile"):
        signal["profile"] = classify_profile(comps if isinstance(comps, dict) else {})
        changed = True

    return changed


def rebuild_signals_from_top10_dates(dates: List[str]) -> List[Dict[str, Any]]:
    rebuilt: List[Dict[str, Any]] = []
    existing_ids = set()

    for date in dates:
        top10 = load_top10_for_date(date)
        if not top10:
            log("WARN", f"No top10 items for {date}")
            continue

        day_added = 0

        for fallback_rank, item in enumerate(top10[:10], start=1):
            sym = str(item.get("symbol", "")).strip().upper()
            if not sym:
                continue

            rank = get_item_rank(item, fallback_rank)
            eligible, rule, quality = evaluate_signal_eligibility(item, rank)

            if not eligible:
                continue

            sid = signal_id(date, sym)
            if sid in existing_ids:
                continue

            rebuilt.append(make_new_signal(item, date, rank, rule, quality))
            existing_ids.add(sid)
            day_added += 1

        log("INFO", f"{date}: rebuilt eligible signals={day_added}")

    return rebuilt


def is_reportable_signal(signal: Dict[str, Any]) -> bool:
    if signal.get("signal_eligible") is True:
        return True

    score = to_int(signal.get("score_pts"))
    return score is not None and score >= REPORT_LEGACY_MIN_SCORE


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


def months_between(start_date: str, end: Optional[pd.Timestamp] = None) -> int:
    start = pd.Timestamp(start_date)
    end_ts = end or pd.Timestamp.utcnow().tz_localize(None)
    days = max(1, int((end_ts - start).days))
    return max(18, int(math.ceil(days / 30.0)) + 6)


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


def reset_signal_outcome(signal: Dict[str, Any]) -> None:
    signal["signal_close"] = None
    signal["entry"] = {
        "method": "next_open",
        "entry_date": None,
        "entry_price": None,
        "gap_pct": None,
    }
    signal["outcome"] = empty_outcome()


def update_signal_with_history(signal: Dict[str, Any], df: pd.DataFrame, force_recalc: bool = True) -> bool:
    signal_date = str(signal.get("signal_date", ""))
    if not signal_date:
        return False

    if force_recalc:
        reset_signal_outcome(signal)

    sig_idx = locate_signal_index(df, signal_date)
    if sig_idx is None:
        return False

    changed = False

    signal_close = to_float(df["Close"].iloc[sig_idx])
    if signal_close is not None:
        signal["signal_close"] = round(signal_close, 4)
        changed = True

    entry_idx = sig_idx + 1
    if entry_idx >= len(df):
        outcome = signal.setdefault("outcome", {})
        outcome["status"] = "pending_entry"
        outcome["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        return True

    entry_date = pd.Timestamp(df.index[entry_idx]).strftime("%Y-%m-%d")
    entry_price = to_float(df["Open"].iloc[entry_idx])

    entry = signal.setdefault("entry", {})
    if entry_price is not None:
        entry["method"] = "next_open"
        entry["entry_date"] = entry_date
        entry["entry_price"] = round(entry_price, 4)
        entry["gap_pct"] = pct_return(entry_price, signal_close)
        changed = True

    entry_price = to_float(entry.get("entry_price"))
    if entry_price is None:
        return changed

    outcome = signal.setdefault("outcome", {})

    for label, h in HORIZONS.items():
        target_idx = entry_idx + h
        key = f"return_{label}_pct"

        if target_idx < len(df):
            target_close = to_float(df["Close"].iloc[target_idx])
            ret = pct_return(target_close, entry_price)
            outcome[key] = ret
            changed = True

    if outcome.get("return_3m_pct") is not None and entry_idx + HORIZONS["3m"] < len(df):
        current_idx = entry_idx + HORIZONS["3m"]
    else:
        current_idx = len(df) - 1

    current_close = to_float(df["Close"].iloc[current_idx])
    current_date = pd.Timestamp(df.index[current_idx]).strftime("%Y-%m-%d")
    current_return = pct_return(current_close, entry_price)

    outcome["current_close"] = None if current_close is None else round(current_close, 4)
    outcome["current_date"] = current_date
    outcome["current_return_pct"] = current_return
    changed = True

    max_window_end = current_idx
    if max_window_end >= entry_idx:
        high_window = df["High"].iloc[entry_idx:max_window_end + 1]
        low_window = df["Low"].iloc[entry_idx:max_window_end + 1]

        max_high = to_float(high_window.max())
        min_low = to_float(low_window.min())

        outcome["max_gain_since_entry_pct"] = pct_return(max_high, entry_price)
        outcome["max_drawdown_since_entry_pct"] = pct_return(min_low, entry_price)
        changed = True

    outcome["d5_return_pct"] = outcome.get("return_1w_pct")
    outcome["d10_return_pct"] = None
    outcome["d20_return_pct"] = outcome.get("return_1m_pct")
    outcome["max_gain_20d_pct"] = outcome.get("max_gain_since_entry_pct")
    outcome["max_drawdown_20d_pct"] = outcome.get("max_drawdown_since_entry_pct")

    outcome["status"] = get_status(outcome, has_entry=True)
    outcome["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    return changed


def flatten_recent_outcomes(signals: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for s in signals:
        if not is_reportable_signal(s):
            continue

        outcome = s.get("outcome") or {}
        entry = s.get("entry") or {}
        comps = s.get("score_components") or {}

        rows.append({
            "id": s.get("id"),
            "signal_date": s.get("signal_date"),
            "symbol": s.get("symbol"),
            "name": s.get("name"),
            "rank": s.get("rank"),
            "rank_bucket": s.get("rank_bucket"),
            "score_bucket": s.get("score_bucket"),
            "profile": s.get("profile"),
            "score_pts": s.get("score_pts"),
            "final_score_0_1": s.get("final_score_0_1"),

            "volume_anomaly": comps.get("volume_anomaly"),
            "compression_release": comps.get("compression_release"),
            "trends_breakout": comps.get("trends_breakout"),
            "news": comps.get("news"),

            "policy_version": s.get("policy_version"),
            "signal_eligible": s.get("signal_eligible"),
            "eligibility_rule": s.get("eligibility_rule"),
            "signal_quality": s.get("signal_quality"),

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
        "avg_current_return": mean(cur),
    }


def build_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    reportable_signals = [s for s in signals if isinstance(s, dict) and is_reportable_signal(s)]
    rows = flatten_recent_outcomes(reportable_signals, limit=100000)

    r1w = [to_float(r.get("return_1w_pct")) for r in rows]
    r1m = [to_float(r.get("return_1m_pct")) for r in rows]
    r3m = [to_float(r.get("return_3m_pct")) for r in rows]
    cur = [to_float(r.get("current_return_pct")) for r in rows]
    gain = [to_float(r.get("max_gain_since_entry_pct")) for r in rows]
    dd = [to_float(r.get("max_drawdown_since_entry_pct")) for r in rows]

    completed_1w = sum(1 for x in r1w if x is not None)
    completed_1m = sum(1 for x in r1m if x is not None)
    completed_3m = sum(1 for x in r3m if x is not None)

    rank_map: Dict[str, List[Dict[str, Any]]] = {}
    score_map: Dict[str, List[Dict[str, Any]]] = {}
    rule_map: Dict[str, List[Dict[str, Any]]] = {}
    profile_map: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        rank_map.setdefault(str(r.get("rank_bucket") or "Other"), []).append(r)
        score_map.setdefault(str(r.get("score_bucket") or "Unknown"), []).append(r)
        rule_map.setdefault(str(r.get("eligibility_rule") or "unknown"), []).append(r)
        profile_map.setdefault(str(r.get("profile") or "Unknown"), []).append(r)

    rank_buckets = [summarize_group(v, k) for k, v in rank_map.items()]
    rank_buckets.sort(
        key=lambda x: (
            x.get("count") or 0,
            x.get("avg_return_3m") if x.get("avg_return_3m") is not None else -999,
        ),
        reverse=True,
    )

    score_buckets = [summarize_group(v, k) for k, v in score_map.items()]
    score_order = {
        "850+": 0,
        "800-849": 1,
        "750-799": 2,
        "700-749": 3,
        "<700": 4,
        "Unknown": 9,
    }
    score_buckets.sort(key=lambda x: score_order.get(str(x.get("label")), 99))

    rule_buckets = [summarize_group(v, k) for k, v in rule_map.items()]
    rule_buckets.sort(key=lambda x: (x.get("count") or 0), reverse=True)

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

    signal_dates = [normalize_date_str(s.get("signal_date")) for s in reportable_signals]
    signal_dates = [d for d in signal_dates if d]
    as_of = max(signal_dates) if signal_dates else None

    active_count = sum(
        1
        for s in reportable_signals
        if (s.get("outcome") or {}).get("status") != "completed_3m"
    )

    return {
        "as_of": as_of,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "tracking_policy": current_policy(),

        "total_signals": len(reportable_signals),
        "raw_total_signals_in_registry": len(signals),
        "hidden_legacy_signals": max(0, len(signals) - len(reportable_signals)),
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

        "rank_buckets": rank_buckets,
        "score_buckets": score_buckets,
        "rule_buckets": rule_buckets,
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


def earliest_signal_date(signals: List[Dict[str, Any]]) -> Optional[str]:
    dates = [normalize_date_str(s.get("signal_date")) for s in signals if isinstance(s, dict)]
    dates = [d for d in dates if d]
    return min(dates) if dates else None


def update_all_signal_outcomes(signals: List[Dict[str, Any]]) -> int:
    symbols = sorted({
        str(s.get("symbol", "")).upper()
        for s in signals
        if isinstance(s, dict) and s.get("symbol")
    })

    symbol_earliest: Dict[str, str] = {}
    for s in signals:
        sym = str(s.get("symbol", "")).upper()
        d = normalize_date_str(s.get("signal_date"))
        if not sym or not d:
            continue
        if sym not in symbol_earliest or d < symbol_earliest[sym]:
            symbol_earliest[sym] = d

    log("INFO", f"Fetching history for symbols: {len(symbols)}")

    history_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for sym in symbols:
        months = months_between(symbol_earliest.get(sym, datetime.now(timezone.utc).strftime("%Y-%m-%d")))
        history_cache[sym] = fetch_history(sym, months=months)

    changed = 0
    for s in signals:
        if not isinstance(s, dict):
            continue

        sym = str(s.get("symbol", "")).upper()
        if not sym:
            continue

        df = history_cache.get(sym)
        if df is None or df.empty:
            continue

        if update_signal_with_history(s, df, force_recalc=True):
            changed += 1

    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild signal registry/outcomes from historical daily top10.json files."
    )
    parser.add_argument(
        "--start-date",
        default=os.getenv("REBUILD_START_DATE", ""),
        help="Rebuild signals from this date, YYYY-MM-DD. Required unless REBUILD_START_DATE is set.",
    )
    parser.add_argument(
        "--end-date",
        default=os.getenv("REBUILD_END_DATE", ""),
        help="Optional end date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create registry backup before rebuild.",
    )
    return parser.parse_args()


def validate_date(value: str, label: str) -> str:
    if not value:
        raise SystemExit(f"{label} is required")
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        raise SystemExit(f"Invalid {label}: {value}. Expected YYYY-MM-DD.")


def main() -> None:
    args = parse_args()
    start_date = validate_date(args.start_date, "start_date")
    end_date = validate_date(args.end_date, "end_date") if args.end_date else None

    log("INFO", f"Rebuild start_date={start_date}, end_date={end_date or 'latest'}")

    if not args.no_backup:
        backup_registry()

    old_registry = load_registry()
    preserved = preserve_signals_before_start(old_registry, start_date)

    for s in preserved:
        if isinstance(s, dict):
            enrich_legacy_signal(s)

    rebuild_dates = list_rebuild_dates(start_date, end_date)
    if not rebuild_dates:
        raise SystemExit(f"No top10.json files found from {start_date} to {end_date or 'latest'}")

    log("INFO", f"Dates to rebuild: {len(rebuild_dates)}")

    rebuilt = rebuild_signals_from_top10_dates(rebuild_dates)

    # preserved と rebuilt の重複を避ける
    combined: List[Dict[str, Any]] = []
    seen = set()

    for s in preserved + rebuilt:
        sid = str(s.get("id") or "")
        if not sid:
            d = normalize_date_str(s.get("signal_date"))
            sym = str(s.get("symbol", "")).upper()
            if d and sym:
                sid = signal_id(d, sym)
                s["id"] = sid

        if not sid or sid in seen:
            continue

        combined.append(s)
        seen.add(sid)

    combined.sort(
        key=lambda s: (
            str(s.get("signal_date") or ""),
            int(s.get("rank") or 999),
            str(s.get("symbol") or ""),
        )
    )

    outcome_changed = update_all_signal_outcomes(combined)

    registry = {
        "created_at": old_registry.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "policy": current_policy(),
        "rebuild": {
            "rebuilt_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "start_date": start_date,
            "end_date": end_date,
            "rebuilt_dates": len(rebuild_dates),
            "preserved_signals_before_start": len(preserved),
            "rebuilt_signals_from_top10": len(rebuilt),
            "combined_signals": len(combined),
        },
        "signals": combined,
    }

    recent = flatten_recent_outcomes(combined, limit=200)
    summary = build_summary(combined)

    write_json(REGISTRY_PATH, registry)
    write_json(OUTCOMES_PATH, {"items": recent})
    write_json(SUMMARY_PATH, summary)

    log("INFO", f"Wrote registry: {REGISTRY_PATH}")
    log("INFO", f"Wrote outcomes: {OUTCOMES_PATH}")
    log("INFO", f"Wrote summary: {SUMMARY_PATH}")
    log("INFO", f"Preserved signals before start: {len(preserved)}")
    log("INFO", f"Rebuilt signals from top10: {len(rebuilt)}")
    log("INFO", f"Combined signals: {len(combined)}")
    log("INFO", f"Outcome recalculated signals: {outcome_changed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in rebuild_signal_outcomes: {e}")
        sys.exit(1)
