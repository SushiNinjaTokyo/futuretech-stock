#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))

REGISTRY_PATH = OUT_DIR / "data" / "signals" / "registry.json"
SUMMARY_PATH = OUT_DIR / "data" / "signals" / "summary_latest.json"
OUTCOMES_PATH = OUT_DIR / "data" / "signals" / "outcomes_latest.json"

POLICY_VERSION = "daily_event_lab_v2"

MIN_TOP3_SCORE = int(os.getenv("SIGNAL_MIN_TOP3_SCORE", "750"))
MIN_VOLUME_SCORE = float(os.getenv("SIGNAL_MIN_VOLUME_SCORE", "0.55"))
MIN_VOLUME_COMBO_SCORE = int(os.getenv("SIGNAL_MIN_VOLUME_COMBO_SCORE", "700"))
MIN_ANY_RANK_SCORE = int(os.getenv("SIGNAL_MIN_ANY_RANK_SCORE", "800"))
REPORT_LEGACY_MIN_SCORE = int(os.getenv("SIGNAL_REPORT_LEGACY_MIN_SCORE", "750"))

BENCHMARKS = [x.strip().upper() for x in os.getenv("DAILY_BENCHMARKS", "SPY,QQQ").split(",") if x.strip()]
PRIMARY_BENCHMARK = BENCHMARKS[0] if BENCHMARKS else "SPY"
SECONDARY_BENCHMARK = BENCHMARKS[1] if len(BENCHMARKS) > 1 else "QQQ"

# Daily event horizons: trading-day offsets from next-open entry.
HORIZONS = {
    "1d": 1,
    "3d": 3,
    "5d": 5,
    "10d": 10,
    "20d": 20,
    # Legacy compatibility with the old page.
    "1w": 5,
    "1m": 20,
    "3m": 63,
}
EVENT_FREEZE_HORIZON = int(os.getenv("DAILY_EVENT_FREEZE_DAYS", "20"))
FOLLOW_THROUGH_DAYS = int(os.getenv("DAILY_FOLLOW_THROUGH_DAYS", "3"))
FOLLOW_THROUGH_PCT = float(os.getenv("DAILY_FOLLOW_THROUGH_PCT", "5"))
STOP_LEVELS = [3.0, 5.0, 8.0]
VOLUME_BUCKETS = [0.0, 0.35, 0.55, 0.75, 1.01]
SCORE_BUCKETS = [(850, 10_000, "850+"), (800, 849, "800-849"), (750, 799, "750-799"), (700, 749, "700-749"), (0, 699, "<700")]


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        return int(float(x))
    except Exception:
        return None


def safe_round(x: Any, ndigits: int = 2) -> Optional[float]:
    v = to_float(x)
    return round(v, ndigits) if v is not None else None


def pct_return(cur: Any, base: Any) -> Optional[float]:
    c = to_float(cur)
    b = to_float(base)
    if c is None or b is None or b == 0:
        return None
    return round((c / b - 1.0) * 100.0, 2)


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


def volume_bucket(volume: Optional[float]) -> str:
    v = to_float(volume)
    if v is None:
        return "Unknown"
    if v >= 0.75:
        return "0.75+"
    if v >= 0.55:
        return "0.55-0.74"
    if v >= 0.35:
        return "0.35-0.54"
    return "<0.35"


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
        strong.append("Trend")
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
        good.append("Trend")
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
    return to_int(item.get("rank")) or fallback_rank


def get_score_pts(item: Dict[str, Any]) -> int:
    score_pts = item.get("score_pts")
    final01 = to_float(item.get("final_score_0_1")) or 0.0
    try:
        return int(score_pts) if score_pts is not None else int(round(final01 * 1000))
    except Exception:
        return int(round(final01 * 1000))


def get_volume_score(item: Dict[str, Any]) -> float:
    return float(normalize_components(item).get("volume_anomaly") or 0.0)


def evaluate_signal_eligibility(item: Dict[str, Any], rank: int) -> Tuple[bool, str, str]:
    score_pts = get_score_pts(item)
    volume_anomaly = get_volume_score(item)
    if rank <= 3 and score_pts >= MIN_TOP3_SCORE:
        return True, "top3_score_750", "Top3 High Score"
    if rank > 3 and score_pts >= MIN_ANY_RANK_SCORE:
        return True, "score_800_any_rank", "High Score Any Rank"
    if score_pts >= MIN_VOLUME_COMBO_SCORE and volume_anomaly >= MIN_VOLUME_SCORE:
        return True, "score_700_volume_055", "Volume Momentum"
    return False, "not_eligible", "Watch"


def event_archetype(signal: Dict[str, Any]) -> str:
    comps = signal.get("score_components") or {}
    vol = to_float(comps.get("volume_anomaly")) or 0.0
    news = to_float(comps.get("news")) or 0.0
    trend = to_float(comps.get("trends_breakout")) or 0.0
    comp = to_float(comps.get("compression_release")) or 0.0
    score = to_int(signal.get("score_pts")) or 0

    if vol >= 0.75 and trend >= 0.55:
        return "Volume Trend Breakout"
    if news >= 0.75 and vol >= 0.55:
        return "News Volume Spike"
    if comp >= 0.75 and vol >= 0.55:
        return "Compression Release"
    if score >= 800:
        return "High Score Momentum"
    if vol >= 0.55:
        return "Volume Follow-through"
    return "Mixed Event"


def current_policy() -> Dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "entry_method": "next_open",
        "horizons": {k: v for k, v in HORIZONS.items() if k in {"1d", "3d", "5d", "10d", "20d"}},
        "legacy_horizons": {"1w": HORIZONS["1w"], "1m": HORIZONS["1m"], "3m": HORIZONS["3m"]},
        "current_return_policy": f"updates until {EVENT_FREEZE_HORIZON} trading days, then freezes",
        "benchmarks": BENCHMARKS,
        "follow_through": f"MFE >= +{FOLLOW_THROUGH_PCT}% within {FOLLOW_THROUGH_DAYS} trading days",
        "registration_rules": [
            f"rank <= 3 and score_pts >= {MIN_TOP3_SCORE}",
            f"score_pts >= {MIN_VOLUME_COMBO_SCORE} and volume_anomaly >= {MIN_VOLUME_SCORE}",
            f"rank > 3 and score_pts >= {MIN_ANY_RANK_SCORE}",
        ],
        "reporting": {"legacy_min_score_for_report": REPORT_LEGACY_MIN_SCORE},
    }


def empty_outcome() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "pending_entry",
        "last_updated": None,
        "current_close": None,
        "current_date": None,
        "current_return_pct": None,
        "mfe_pct": None,
        "mae_pct": None,
        "mfe_mae_ratio": None,
        "peak_day": None,
        "peak_return_pct": None,
        "follow_through_3d_5pct": None,
        "stop_hit_3pct": None,
        "stop_hit_5pct": None,
        "stop_hit_8pct": None,
        "path": [],
        "benchmarks": {},
        "regime": "Unknown",
    }
    for h in HORIZONS:
        out[f"return_{h}_pct"] = None
    out.update({
        "max_gain_since_entry_pct": None,
        "max_drawdown_since_entry_pct": None,
        "d5_return_pct": None,
        "d10_return_pct": None,
        "d20_return_pct": None,
        "max_gain_20d_pct": None,
        "max_drawdown_20d_pct": None,
    })
    return out


def make_new_signal(item: Dict[str, Any], signal_date: str, rank: int, rule: str, quality: str) -> Dict[str, Any]:
    symbol = str(item.get("symbol", "")).strip().upper()
    name = str(item.get("name", "")).strip()
    comps = normalize_components(item)
    weights = normalize_weights(item)
    score_pts_int = get_score_pts(item)
    final01 = to_float(item.get("final_score_0_1")) or (score_pts_int / 1000.0)
    signal = {
        "id": signal_id(signal_date, symbol),
        "signal_date": signal_date,
        "symbol": symbol,
        "name": name,
        "rank": rank,
        "rank_bucket": rank_bucket(rank),
        "score_bucket": score_bucket(score_pts_int),
        "volume_bucket": volume_bucket(comps.get("volume_anomaly")),
        "score_pts": score_pts_int,
        "final_score_0_1": round(final01, 6),
        "score_components": comps,
        "score_weights": weights,
        "profile": classify_profile(comps),
        "event_archetype": "Mixed Event",
        "policy_version": POLICY_VERSION,
        "signal_eligible": True,
        "eligibility_rule": rule,
        "signal_quality": quality,
        "signal_close": None,
        "entry": {"method": "next_open", "entry_date": None, "entry_price": None, "gap_pct": None},
        "outcome": empty_outcome(),
        "source_snapshot": {
            "price_delta_1d": to_float(item.get("price_delta_1d")),
            "price_delta_1w": to_float(item.get("price_delta_1w")),
            "price_delta_1m": to_float(item.get("price_delta_1m")),
        },
    }
    signal["event_archetype"] = event_archetype(signal)
    return signal


def enrich_signal_defaults(signal: Dict[str, Any]) -> Dict[str, Any]:
    rank = to_int(signal.get("rank"))
    score = to_int(signal.get("score_pts"))
    comps = signal.get("score_components") if isinstance(signal.get("score_components"), dict) else {}
    signal.setdefault("id", signal_id(normalize_date_str(signal.get("signal_date")) or "unknown", str(signal.get("symbol", "")).upper()))
    signal["rank_bucket"] = signal.get("rank_bucket") or rank_bucket(rank)
    signal["score_bucket"] = signal.get("score_bucket") or score_bucket(score)
    signal["volume_bucket"] = signal.get("volume_bucket") or volume_bucket(comps.get("volume_anomaly"))
    signal["profile"] = signal.get("profile") or classify_profile(comps)
    signal["event_archetype"] = signal.get("event_archetype") or event_archetype(signal)
    signal.setdefault("policy_version", "legacy")
    if "signal_eligible" not in signal:
        signal["signal_eligible"] = score is not None and score >= REPORT_LEGACY_MIN_SCORE
    signal.setdefault("eligibility_rule", "legacy_score_filtered")
    signal.setdefault("signal_quality", "Legacy Signal" if signal.get("signal_eligible") else "Legacy / Weak")
    signal.setdefault("entry", {"method": "next_open", "entry_date": None, "entry_price": None, "gap_pct": None})
    outcome = signal.get("outcome") if isinstance(signal.get("outcome"), dict) else {}
    merged = empty_outcome()
    merged.update(outcome)
    signal["outcome"] = merged
    return signal


def is_reportable_signal(signal: Dict[str, Any]) -> bool:
    if signal.get("signal_eligible") is True:
        return True
    score = to_int(signal.get("score_pts"))
    return score is not None and score >= REPORT_LEGACY_MIN_SCORE


def backup_registry() -> Optional[Path]:
    if not REGISTRY_PATH.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = REGISTRY_PATH.with_name(f"registry.backup.{stamp}.json")
    backup.write_text(REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    log("INFO", f"Registry backup created: {backup}")
    return backup


def load_registry() -> Dict[str, Any]:
    j = read_json(REGISTRY_PATH)
    if isinstance(j, dict) and isinstance(j.get("signals"), list):
        for s in j.get("signals", []):
            if isinstance(s, dict):
                enrich_signal_defaults(s)
        return j
    return {"created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"), "updated_at": None, "policy": current_policy(), "signals": []}


def load_top10_for_date(date: str) -> List[Dict[str, Any]]:
    paths = [OUT_DIR / "data" / date / "top10.json", OUT_DIR / "data" / "top10" / "latest.json"]
    for path in paths:
        j = read_json(path)
        if not j:
            continue
        payload = j.get("items", j) if isinstance(j, dict) else j
        if isinstance(payload, list):
            return [x for x in payload[:10] if isinstance(x, dict)]
    return []


def list_rebuild_dates(start_date: str, end_date: Optional[str] = None) -> List[str]:
    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        return []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else None
    dates: List[str] = []
    for d in data_dir.iterdir():
        if not d.is_dir() or not valid_date_dir_name(d.name):
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
    preserved = []
    for s in registry.get("signals", []):
        if not isinstance(s, dict):
            continue
        d = normalize_date_str(s.get("signal_date"))
        if d and pd.Timestamp(d) < start_ts:
            preserved.append(enrich_signal_defaults(s))
    return preserved


def rebuild_signals_from_top10_dates(dates: List[str]) -> List[Dict[str, Any]]:
    rebuilt: List[Dict[str, Any]] = []
    seen = set()
    for date in dates:
        top10 = load_top10_for_date(date)
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
            if sid in seen:
                continue
            rebuilt.append(make_new_signal(item, date, rank, rule, quality))
            seen.add(sid)
            day_added += 1
        log("INFO", f"{date}: rebuilt eligible signals={day_added}")
    return rebuilt


def first_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        return pd.to_numeric(x.iloc[:, 0], errors="coerce")
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
            out[t.capitalize()] = first_series(df.loc[:, chosen[t]]).to_numpy() if t in chosen else np.nan
    else:
        src_map = {str(c).strip().lower(): c for c in df.columns}
        for t in targets:
            src = src_map.get(t)
            out[t.capitalize()] = first_series(df[src]).to_numpy() if src is not None else np.nan
    out = out[["Open", "High", "Low", "Close", "Volume"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def months_between(start_date: str, end: Optional[pd.Timestamp] = None) -> int:
    try:
        start = pd.Timestamp(start_date)
        end_ts = end or pd.Timestamp.utcnow()
        days = max(120, (end_ts - start).days + 120)
        return max(8, min(72, int(days / 30) + 2))
    except Exception:
        return 18


def fetch_history(symbol: str, months: int = 18) -> Optional[pd.DataFrame]:
    if yf is None:
        log("WARN", "yfinance unavailable")
        return None
    for attempt in range(3):
        try:
            raw = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False, threads=False, auto_adjust=False)
            df = normalize_ohlcv(raw)
            if not df.empty:
                return df
        except Exception as e:
            log("WARN", f"fetch failed {symbol} attempt={attempt+1}: {e}")
    return None


def locate_signal_index(df: pd.DataFrame, signal_date: str) -> Optional[int]:
    if df.empty:
        return None
    target = pd.Timestamp(signal_date).normalize()
    candidates = [i for i, idx in enumerate(df.index) if pd.Timestamp(idx).normalize() <= target]
    if candidates:
        return candidates[-1]
    return None


def entry_index_after_signal(df: pd.DataFrame, signal_date: str) -> Optional[int]:
    target = pd.Timestamp(signal_date).normalize()
    for i, idx in enumerate(df.index):
        if pd.Timestamp(idx).normalize() > target:
            return i
    return None


def status_from_outcome(outcome: Dict[str, Any], has_entry: bool) -> str:
    if not has_entry:
        return "pending_entry"
    if outcome.get("return_20d_pct") is not None:
        return "completed_20d"
    if outcome.get("return_10d_pct") is not None:
        return "completed_10d"
    if outcome.get("return_5d_pct") is not None:
        return "completed_5d"
    if outcome.get("return_3d_pct") is not None:
        return "completed_3d"
    if outcome.get("return_1d_pct") is not None:
        return "completed_1d"
    return "active"


def benchmark_returns_for_entry(df: Optional[pd.DataFrame], entry_date: str, entry_price_source: str = "Open") -> Dict[str, Any]:
    out = {"entry_date": None, "entry_price": None, "returns": {}, "current_return_pct": None, "path": []}
    if df is None or df.empty:
        return out
    idx = None
    target = pd.Timestamp(entry_date).normalize()
    for i, ts in enumerate(df.index):
        if pd.Timestamp(ts).normalize() >= target:
            idx = i
            break
    if idx is None or idx >= len(df):
        return out
    entry_price = to_float(df[entry_price_source].iloc[idx]) or to_float(df["Close"].iloc[idx])
    if entry_price is None or entry_price <= 0:
        return out
    out["entry_date"] = pd.Timestamp(df.index[idx]).strftime("%Y-%m-%d")
    out["entry_price"] = round(entry_price, 4)
    max_idx = min(idx + EVENT_FREEZE_HORIZON, len(df) - 1)
    for label, offset in HORIZONS.items():
        target_idx = idx + offset
        out["returns"][label] = pct_return(df["Close"].iloc[target_idx], entry_price) if target_idx < len(df) else None
    out["current_return_pct"] = pct_return(df["Close"].iloc[max_idx], entry_price)
    path = []
    for day in [0, 1, 3, 5, 10, 20]:
        j = idx + day
        path.append({"day": day, "return_pct": pct_return(df["Close"].iloc[j], entry_price) if j < len(df) else None})
    out["path"] = path
    return out


def market_regime(benchmarks: Dict[str, Optional[pd.DataFrame]], entry_date: Optional[str]) -> str:
    if not entry_date:
        return "Unknown"
    checks = []
    for sym in [PRIMARY_BENCHMARK, SECONDARY_BENCHMARK]:
        df = benchmarks.get(sym)
        if df is None or df.empty or len(df) < 50:
            continue
        d = df.copy()
        d["SMA20"] = d["Close"].rolling(20).mean()
        d["SMA50"] = d["Close"].rolling(50).mean()
        target = pd.Timestamp(entry_date).normalize()
        rows = [i for i, idx in enumerate(d.index) if pd.Timestamp(idx).normalize() <= target]
        if not rows:
            continue
        row = d.iloc[rows[-1]]
        close = to_float(row.get("Close"))
        sma20 = to_float(row.get("SMA20"))
        sma50 = to_float(row.get("SMA50"))
        if close is None or sma20 is None or sma50 is None:
            continue
        checks.append((close > sma20, close > sma50))
    if not checks:
        return "Unknown"
    if all(a and b for a, b in checks):
        return "Risk-on"
    if any(not b for _, b in checks):
        return "Risk-off"
    return "Neutral"


def update_signal_with_history(signal: Dict[str, Any], df: pd.DataFrame, benchmarks: Dict[str, Optional[pd.DataFrame]], force_recalc: bool = True) -> bool:
    signal = enrich_signal_defaults(signal)
    signal_date = normalize_date_str(signal.get("signal_date"))
    if not signal_date or df is None or df.empty:
        return False
    sig_idx = locate_signal_index(df, signal_date)
    ent_idx = entry_index_after_signal(df, signal_date)
    if ent_idx is None or ent_idx >= len(df):
        signal["outcome"]["status"] = "pending_entry"
        signal["outcome"]["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        return True
    signal_close = to_float(df["Close"].iloc[sig_idx]) if sig_idx is not None else None
    signal["signal_close"] = None if signal_close is None else round(signal_close, 4)
    entry_price = to_float(df["Open"].iloc[ent_idx]) or to_float(df["Close"].iloc[ent_idx])
    if entry_price is None or entry_price <= 0:
        return False
    entry_date = pd.Timestamp(df.index[ent_idx]).strftime("%Y-%m-%d")
    signal["entry"] = {
        "method": "next_open",
        "entry_date": entry_date,
        "entry_price": round(entry_price, 4),
        "gap_pct": pct_return(entry_price, signal_close),
    }
    outcome = empty_outcome()
    # Horizons
    for label, offset in HORIZONS.items():
        target_idx = ent_idx + offset
        outcome[f"return_{label}_pct"] = pct_return(df["Close"].iloc[target_idx], entry_price) if target_idx < len(df) else None
    current_idx = min(ent_idx + EVENT_FREEZE_HORIZON, len(df) - 1)
    current_close = to_float(df["Close"].iloc[current_idx])
    outcome["current_close"] = None if current_close is None else round(current_close, 4)
    outcome["current_date"] = pd.Timestamp(df.index[current_idx]).strftime("%Y-%m-%d")
    outcome["current_return_pct"] = pct_return(current_close, entry_price)

    window = df.iloc[ent_idx:current_idx + 1]
    if not window.empty:
        high = to_float(window["High"].max())
        low = to_float(window["Low"].min())
        outcome["mfe_pct"] = pct_return(high, entry_price)
        outcome["mae_pct"] = pct_return(low, entry_price)
        outcome["max_gain_since_entry_pct"] = outcome["mfe_pct"]
        outcome["max_drawdown_since_entry_pct"] = outcome["mae_pct"]
        if outcome["mfe_pct"] is not None and outcome["mae_pct"] is not None and outcome["mae_pct"] != 0:
            outcome["mfe_mae_ratio"] = round(abs(outcome["mfe_pct"] / outcome["mae_pct"]), 2)
        # Peak day from entry to event horizon.
        try:
            hi_pos = int(np.argmax(window["High"].to_numpy(dtype=float)))
            outcome["peak_day"] = hi_pos
            outcome["peak_return_pct"] = pct_return(window["High"].iloc[hi_pos], entry_price)
        except Exception:
            pass
        # Stop and follow-through diagnostics.
        follow_end = min(ent_idx + FOLLOW_THROUGH_DAYS, len(df) - 1)
        ft_window = df.iloc[ent_idx:follow_end + 1]
        ft_high = to_float(ft_window["High"].max()) if not ft_window.empty else None
        outcome["follow_through_3d_5pct"] = (pct_return(ft_high, entry_price) or -999) >= FOLLOW_THROUGH_PCT if ft_high is not None else None
        for stop in STOP_LEVELS:
            stop_price = entry_price * (1.0 - stop / 100.0)
            hit = bool((window["Low"] <= stop_price).any())
            outcome[f"stop_hit_{int(stop)}pct"] = hit
    path = []
    for day in [0, 1, 3, 5, 10, 20]:
        j = ent_idx + day
        path.append({"day": day, "return_pct": pct_return(df["Close"].iloc[j], entry_price) if j < len(df) else None})
    outcome["path"] = path

    bdata = {}
    for sym, bdf in benchmarks.items():
        b = benchmark_returns_for_entry(bdf, entry_date)
        bdata[sym] = b
        for h in ["1d", "3d", "5d", "10d", "20d"]:
            r = outcome.get(f"return_{h}_pct")
            br = (b.get("returns") or {}).get(h)
            outcome[f"{sym.lower()}_return_{h}_pct"] = br
            outcome[f"{sym.lower()}_alpha_{h}_pct"] = round(float(r) - float(br), 2) if r is not None and br is not None else None
        outcome[f"{sym.lower()}_current_return_pct"] = b.get("current_return_pct")
        outcome[f"{sym.lower()}_current_alpha_pct"] = round(float(outcome["current_return_pct"]) - float(b.get("current_return_pct")), 2) if outcome.get("current_return_pct") is not None and b.get("current_return_pct") is not None else None
    outcome["benchmarks"] = bdata
    outcome["regime"] = market_regime(benchmarks, entry_date)

    outcome["d5_return_pct"] = outcome.get("return_5d_pct")
    outcome["d10_return_pct"] = outcome.get("return_10d_pct")
    outcome["d20_return_pct"] = outcome.get("return_20d_pct")
    outcome["max_gain_20d_pct"] = outcome.get("mfe_pct")
    outcome["max_drawdown_20d_pct"] = outcome.get("mae_pct")
    outcome["status"] = status_from_outcome(outcome, has_entry=True)
    outcome["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    signal["outcome"] = outcome
    return True


def flatten_recent_outcomes(signals: List[Dict[str, Any]], limit: int = 300) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in signals:
        if not isinstance(s, dict) or not is_reportable_signal(s):
            continue
        s = enrich_signal_defaults(s)
        o = s.get("outcome") or {}
        e = s.get("entry") or {}
        comps = s.get("score_components") or {}
        row = {
            "id": s.get("id"), "signal_date": s.get("signal_date"), "symbol": s.get("symbol"), "name": s.get("name"),
            "rank": s.get("rank"), "rank_bucket": s.get("rank_bucket"), "score_bucket": s.get("score_bucket"), "volume_bucket": s.get("volume_bucket"),
            "profile": s.get("profile"), "event_archetype": s.get("event_archetype"), "score_pts": s.get("score_pts"), "final_score_0_1": s.get("final_score_0_1"),
            "volume_anomaly": comps.get("volume_anomaly"), "compression_release": comps.get("compression_release"), "trends_breakout": comps.get("trends_breakout"), "news": comps.get("news"),
            "policy_version": s.get("policy_version"), "signal_eligible": s.get("signal_eligible"), "eligibility_rule": s.get("eligibility_rule"), "signal_quality": s.get("signal_quality"),
            "entry_date": e.get("entry_date"), "entry_price": e.get("entry_price"), "gap_pct": e.get("gap_pct"),
            "status": o.get("status"), "current_close": o.get("current_close"), "current_date": o.get("current_date"), "current_return_pct": o.get("current_return_pct"),
            "mfe_pct": o.get("mfe_pct"), "mae_pct": o.get("mae_pct"), "mfe_mae_ratio": o.get("mfe_mae_ratio"), "peak_day": o.get("peak_day"), "peak_return_pct": o.get("peak_return_pct"),
            "follow_through_3d_5pct": o.get("follow_through_3d_5pct"), "stop_hit_3pct": o.get("stop_hit_3pct"), "stop_hit_5pct": o.get("stop_hit_5pct"), "stop_hit_8pct": o.get("stop_hit_8pct"),
            "regime": o.get("regime"), "path": o.get("path") or [],
            "return_1d_pct": o.get("return_1d_pct"), "return_3d_pct": o.get("return_3d_pct"), "return_5d_pct": o.get("return_5d_pct"), "return_10d_pct": o.get("return_10d_pct"), "return_20d_pct": o.get("return_20d_pct"),
            "return_1w_pct": o.get("return_1w_pct"), "return_1m_pct": o.get("return_1m_pct"), "return_3m_pct": o.get("return_3m_pct"),
            "max_gain_since_entry_pct": o.get("max_gain_since_entry_pct"), "max_drawdown_since_entry_pct": o.get("max_drawdown_since_entry_pct"),
            "d5_return_pct": o.get("d5_return_pct"), "d10_return_pct": o.get("d10_return_pct"), "d20_return_pct": o.get("d20_return_pct"),
        }
        for sym in BENCHMARKS:
            sl = sym.lower()
            for h in ["1d", "3d", "5d", "10d", "20d"]:
                row[f"{sl}_return_{h}_pct"] = o.get(f"{sl}_return_{h}_pct")
                row[f"{sl}_alpha_{h}_pct"] = o.get(f"{sl}_alpha_{h}_pct")
            row[f"{sl}_current_return_pct"] = o.get(f"{sl}_current_return_pct")
            row[f"{sl}_current_alpha_pct"] = o.get(f"{sl}_current_alpha_pct")
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("signal_date") or ""), -(to_int(r.get("rank")) or 999)), reverse=True)
    return rows[:limit]


def vals(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(v) for r in rows for v in [to_float(r.get(key))] if v is not None and math.isfinite(float(v))]


def mean_key(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    xs = vals(rows, key)
    return round(float(np.mean(xs)), 2) if xs else None


def median_key(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    xs = vals(rows, key)
    return round(float(np.median(xs)), 2) if xs else None


def rate_from_bools(xs: List[Any]) -> Optional[float]:
    valid = [x for x in xs if x is not None]
    if not valid:
        return None
    return round(sum(1 for x in valid if bool(x)) / len(valid), 4)


def win_rate_key(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    xs = vals(rows, key)
    if not xs:
        return None
    return round(sum(1 for x in xs if x > 0) / len(xs), 4)


def avg_mfe_mae_ratio(rows: List[Dict[str, Any]]) -> Optional[float]:
    mfe = mean_key(rows, "mfe_pct")
    mae = mean_key(rows, "mae_pct")
    if mfe is None or mae is None or mae == 0:
        return None
    return round(abs(mfe / mae), 2)


def summarize_rows(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    out = {"label": label, "count": len(rows)}
    for h in ["1d", "3d", "5d", "10d", "20d"]:
        out[f"completed_{h}"] = len(vals(rows, f"return_{h}_pct"))
        out[f"avg_return_{h}"] = mean_key(rows, f"return_{h}_pct")
        out[f"win_rate_{h}"] = win_rate_key(rows, f"return_{h}_pct")
        for sym in BENCHMARKS:
            sl = sym.lower()
            out[f"avg_{sl}_return_{h}"] = mean_key(rows, f"{sl}_return_{h}_pct")
            out[f"avg_{sl}_alpha_{h}"] = mean_key(rows, f"{sl}_alpha_{h}_pct")
    out.update({
        "avg_current_return": mean_key(rows, "current_return_pct"),
        "avg_mfe": mean_key(rows, "mfe_pct"),
        "avg_mae": mean_key(rows, "mae_pct"),
        "mfe_mae_ratio": avg_mfe_mae_ratio(rows),
        "follow_through_rate": rate_from_bools([r.get("follow_through_3d_5pct") for r in rows]),
        "stop_hit_3pct_rate": rate_from_bools([r.get("stop_hit_3pct") for r in rows]),
        "stop_hit_5pct_rate": rate_from_bools([r.get("stop_hit_5pct") for r in rows]),
        "stop_hit_8pct_rate": rate_from_bools([r.get("stop_hit_8pct") for r in rows]),
        "avg_peak_day": mean_key(rows, "peak_day"),
        "avg_peak_return": mean_key(rows, "peak_return_pct"),
    })
    for sym in BENCHMARKS:
        sl = sym.lower()
        out[f"avg_{sl}_current_alpha"] = mean_key(rows, f"{sl}_current_alpha_pct")
    return out


def group_summary(rows: List[Dict[str, Any]], key: str, min_count: int = 1) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(str(r.get(key) or "Unknown"), []).append(r)
    out = [summarize_rows(v, k) for k, v in buckets.items() if len(v) >= min_count]
    out.sort(key=lambda x: (x.get("count") or 0, x.get("avg_return_5d") if x.get("avg_return_5d") is not None else -999), reverse=True)
    return out


def build_signal_path(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for day in [0, 1, 3, 5, 10, 20]:
        sig_vals, spy_vals, qqq_vals = [], [], []
        for r in rows:
            for point in r.get("path") or []:
                if int(point.get("day") or -1) == day and point.get("return_pct") is not None:
                    sig_vals.append(float(point.get("return_pct")))
            for sym, arr in [(PRIMARY_BENCHMARK, spy_vals), (SECONDARY_BENCHMARK, qqq_vals)]:
                key = f"{sym.lower()}_return_{day}d_pct" if day else None
                if day == 0:
                    arr.append(0.0)
                elif key and r.get(key) is not None:
                    arr.append(float(r.get(key)))
        out.append({
            "day": day,
            "signal_avg": round(float(np.mean(sig_vals)), 2) if sig_vals else (0.0 if day == 0 else None),
            "spy_avg": round(float(np.mean(spy_vals)), 2) if spy_vals else (0.0 if day == 0 else None),
            "qqq_avg": round(float(np.mean(qqq_vals)), 2) if qqq_vals else (0.0 if day == 0 else None),
        })
    return out


def build_peak_distribution(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = [(0, 1, "Day 0-1"), (2, 3, "Day 2-3"), (4, 5, "Day 4-5"), (6, 10, "Day 6-10"), (11, 20, "Day 11-20")]
    total = sum(1 for r in rows if r.get("peak_day") is not None)
    out = []
    for lo, hi, label in buckets:
        count = sum(1 for r in rows if r.get("peak_day") is not None and lo <= int(r.get("peak_day")) <= hi)
        out.append({"label": label, "count": count, "share": round(count / total, 4) if total else None})
    return out


def build_quality_matrix(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    score_labels = ["700-749", "750-799", "800-849", "850+"]
    vol_labels = ["<0.35", "0.35-0.54", "0.55-0.74", "0.75+"]
    out = []
    for vol in vol_labels:
        for score in score_labels:
            subset = [r for r in rows if r.get("score_bucket") == score and r.get("volume_bucket") == vol]
            base = summarize_rows(subset, f"{vol} / {score}") if subset else {"label": f"{vol} / {score}", "count": 0, "avg_return_5d": None, "win_rate_5d": None, "follow_through_rate": None, "stop_hit_5pct_rate": None}
            base.update({"volume_bucket": vol, "score_bucket": score})
            out.append(base)
    return out


def simulate_strategy(rows: List[Dict[str, Any]], name: str, mode: str) -> Dict[str, Any]:
    """Lightweight daily event strategy comparison from row-level path data.
    The goal is not a full cash portfolio; it is a consistent event-lot rules comparison.
    Each row is equally weighted, using available MFE/MAE/path diagnostics.
    """
    returns = []
    wins = 0
    stop_hits = 0
    for r in rows:
        def ret(key: str) -> Optional[float]:
            return to_float(r.get(key))
        mae = to_float(r.get("mae_pct"))
        mfe = to_float(r.get("mfe_pct"))
        val: Optional[float]
        if mode == "hold_5d":
            val = ret("return_5d_pct")
        elif mode == "hold_10d":
            val = ret("return_10d_pct")
        elif mode == "hold_20d":
            val = ret("return_20d_pct")
        elif mode.startswith("stop_"):
            stop = float(mode.split("_")[1])
            base = ret("return_10d_pct") if ret("return_10d_pct") is not None else ret("current_return_pct")
            if mae is not None and mae <= -stop:
                val = -stop
                stop_hits += 1
            else:
                val = base
        elif mode == "profit_lock_10_5":
            base = ret("return_20d_pct") if ret("return_20d_pct") is not None else ret("current_return_pct")
            if mfe is not None and mfe >= 10 and mae is not None and mae <= -5:
                val = 5.0
            else:
                val = base
        elif mode == "profit_lock_15_7":
            base = ret("return_20d_pct") if ret("return_20d_pct") is not None else ret("current_return_pct")
            if mfe is not None and mfe >= 15 and mae is not None and mae <= -7:
                val = 8.0
            else:
                val = base
        elif mode == "score_volume_only":
            if (to_int(r.get("score_pts")) or 0) >= 700 and (to_float(r.get("volume_anomaly")) or 0) >= 0.55:
                val = ret("return_5d_pct")
            else:
                continue
        elif mode == "trade_triage":
            if (to_int(r.get("score_pts")) or 0) >= 750 and (to_float(r.get("volume_anomaly")) or 0) >= 0.55 and str(r.get("regime")) != "Risk-off":
                val = ret("return_5d_pct")
            else:
                continue
        else:
            val = ret("return_5d_pct")
        if val is None:
            continue
        returns.append(float(val))
        if val > 0:
            wins += 1
    avg_ret = round(float(np.mean(returns)), 2) if returns else None
    max_dd = round(float(min(returns)), 2) if returns else None
    return_dd = round(avg_ret / abs(max_dd), 2) if avg_ret is not None and max_dd not in (None, 0) else None
    return {
        "name": name,
        "mode": mode,
        "avg_return": avg_ret,
        "max_adverse_result": max_dd,
        "return_dd_ratio": return_dd,
        "win_rate": round(wins / len(returns), 4) if returns else None,
        "trades": len(returns),
        "stop_hit_count": stop_hits,
    }


def build_strategy_comparison(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    variants = [
        ("Hold 5D", "hold_5d"),
        ("Hold 10D", "hold_10d"),
        ("Hold 20D", "hold_20d"),
        ("Stop -3%", "stop_3"),
        ("Stop -5%", "stop_5"),
        ("Stop -8%", "stop_8"),
        ("Profit Lock 10/5", "profit_lock_10_5"),
        ("Profit Lock 15/7", "profit_lock_15_7"),
        ("Score+Volume Only", "score_volume_only"),
        ("Trade Triage", "trade_triage"),
    ]
    return [simulate_strategy(rows, name, mode) for name, mode in variants]


def build_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    losers = [r for r in rows if (to_float(r.get("return_5d_pct")) or 0) < 0]
    winners = [r for r in rows if (to_float(r.get("return_5d_pct")) or 0) > 0]
    def avg_of(sub: List[Dict[str, Any]], key: str) -> Optional[float]:
        return mean_key(sub, key)
    return {
        "loser_count": len(losers),
        "winner_count": len(winners),
        "loser_avg_gap": avg_of(losers, "gap_pct"),
        "loser_avg_volume": avg_of(losers, "volume_anomaly"),
        "loser_stop_hit_5pct": rate_from_bools([r.get("stop_hit_5pct") for r in losers]),
        "winner_avg_volume": avg_of(winners, "volume_anomaly"),
        "winner_follow_through": rate_from_bools([r.get("follow_through_3d_5pct") for r in winners]),
        "notes": [
            "High stop-hit rates indicate either noisy entries or insufficient stop distance.",
            "Follow-through rate is the core daily event-quality diagnostic.",
            "Compare SPY and QQQ alpha before trusting raw event returns.",
        ],
    }


def build_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    reportable = [enrich_signal_defaults(s) for s in signals if isinstance(s, dict) and is_reportable_signal(s)]
    rows = flatten_recent_outcomes(reportable, limit=100000)
    active_count = sum(1 for r in rows if str(r.get("status")) not in {"completed_20d", "completed_3m"})
    as_of_dates = [normalize_date_str(r.get("signal_date")) for r in rows]
    as_of_dates = [d for d in as_of_dates if d]
    summary = summarize_rows(rows, "All Signals")
    summary.update({
        "as_of": max(as_of_dates) if as_of_dates else None,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "tracking_policy": current_policy(),
        "total_signals": len(rows),
        "raw_total_signals_in_registry": len(signals),
        "hidden_legacy_signals": max(0, len(signals) - len(rows)),
        "active_signals": active_count,
        "primary_benchmark": PRIMARY_BENCHMARK,
        "secondary_benchmark": SECONDARY_BENCHMARK,
        "rank_buckets": group_summary(rows, "rank_bucket"),
        "score_buckets": group_summary(rows, "score_bucket"),
        "volume_buckets": group_summary(rows, "volume_bucket"),
        "rule_buckets": group_summary(rows, "eligibility_rule"),
        "profiles": group_summary(rows, "profile", min_count=1)[:20],
        "archetypes": group_summary(rows, "event_archetype", min_count=1)[:20],
        "regime_buckets": group_summary(rows, "regime", min_count=1),
        "quality_matrix": build_quality_matrix(rows),
        "signal_path": build_signal_path(rows),
        "peak_distribution": build_peak_distribution(rows),
        "strategy_comparison": build_strategy_comparison(rows),
        "diagnostics": build_diagnostics(rows),
        # Compatibility keys for old template/actions.
        "completed_1w": len(vals(rows, "return_1w_pct")),
        "completed_1m": len(vals(rows, "return_1m_pct")),
        "completed_3m": len(vals(rows, "return_3m_pct")),
        "win_rate_1w": win_rate_key(rows, "return_1w_pct"),
        "win_rate_1m": win_rate_key(rows, "return_1m_pct"),
        "win_rate_3m": win_rate_key(rows, "return_3m_pct"),
        "avg_return_1w": mean_key(rows, "return_1w_pct"),
        "avg_return_1m": mean_key(rows, "return_1m_pct"),
        "avg_return_3m": mean_key(rows, "return_3m_pct"),
        "median_return_3m": median_key(rows, "return_3m_pct"),
        "avg_current_return": mean_key(rows, "current_return_pct"),
        "avg_max_gain": mean_key(rows, "mfe_pct"),
        "avg_max_drawdown": mean_key(rows, "mae_pct"),
        "completed_5d": len(vals(rows, "return_5d_pct")),
        "completed_10d": len(vals(rows, "return_10d_pct")),
        "completed_20d": len(vals(rows, "return_20d_pct")),
        "win_rate_5d": win_rate_key(rows, "return_5d_pct"),
        "win_rate_10d": win_rate_key(rows, "return_10d_pct"),
        "win_rate_20d": win_rate_key(rows, "return_20d_pct"),
        "avg_return_5d": mean_key(rows, "return_5d_pct"),
        "avg_return_10d": mean_key(rows, "return_10d_pct"),
        "avg_return_20d": mean_key(rows, "return_20d_pct"),
        "median_return_20d": median_key(rows, "return_20d_pct"),
        "avg_max_gain_20d": mean_key(rows, "mfe_pct"),
        "avg_max_drawdown_20d": mean_key(rows, "mae_pct"),
    })
    return summary


def earliest_signal_date(signals: List[Dict[str, Any]]) -> Optional[str]:
    dates = [normalize_date_str(s.get("signal_date")) for s in signals if isinstance(s, dict)]
    dates = [d for d in dates if d]
    return min(dates) if dates else None


def update_all_signal_outcomes(signals: List[Dict[str, Any]]) -> int:
    symbols = sorted({str(s.get("symbol", "")).upper() for s in signals if isinstance(s, dict) and s.get("symbol")})
    symbol_earliest: Dict[str, str] = {}
    for s in signals:
        sym = str(s.get("symbol", "")).upper()
        d = normalize_date_str(s.get("signal_date"))
        if not sym or not d:
            continue
        if sym not in symbol_earliest or d < symbol_earliest[sym]:
            symbol_earliest[sym] = d
    log("INFO", f"Fetching price history for symbols={len(symbols)}, benchmarks={','.join(BENCHMARKS)}")
    history_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for sym in symbols:
        months = months_between(symbol_earliest.get(sym, datetime.now(timezone.utc).strftime("%Y-%m-%d")))
        history_cache[sym] = fetch_history(sym, months=months)
    earliest = min(symbol_earliest.values()) if symbol_earliest else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    benchmark_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for b in BENCHMARKS:
        benchmark_cache[b] = fetch_history(b, months=months_between(earliest))
    changed = 0
    for s in signals:
        if not isinstance(s, dict):
            continue
        sym = str(s.get("symbol", "")).upper()
        df = history_cache.get(sym)
        if df is None or df.empty:
            continue
        if update_signal_with_history(s, df, benchmark_cache, force_recalc=True):
            changed += 1
    return changed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild Daily Event Backtest signal registry/outcomes from historical daily top10.json files.")
    p.add_argument("--start-date", default=os.getenv("REBUILD_START_DATE", ""), help="YYYY-MM-DD")
    p.add_argument("--end-date", default=os.getenv("REBUILD_END_DATE", ""), help="optional YYYY-MM-DD")
    p.add_argument("--no-backup", action="store_true", help="Do not create registry backup before rebuild")
    return p.parse_args()


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
    log("INFO", f"Daily Event Lab rebuild start={start_date}, end={end_date or 'latest'}")
    if not args.no_backup:
        backup_registry()
    old_registry = load_registry()
    preserved = preserve_signals_before_start(old_registry, start_date)
    rebuild_dates = list_rebuild_dates(start_date, end_date)
    if not rebuild_dates:
        raise SystemExit(f"No top10.json files found from {start_date} to {end_date or 'latest'}")
    rebuilt = rebuild_signals_from_top10_dates(rebuild_dates)
    combined: List[Dict[str, Any]] = []
    seen = set()
    for s in preserved + rebuilt:
        if not isinstance(s, dict):
            continue
        s = enrich_signal_defaults(s)
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
    combined.sort(key=lambda s: (str(s.get("signal_date") or ""), int(s.get("rank") or 999), str(s.get("symbol") or "")))
    outcome_changed = update_all_signal_outcomes(combined)
    registry = {
        "created_at": old_registry.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "policy": current_policy(),
        "rebuild": {"rebuilt_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"), "start_date": start_date, "end_date": end_date, "rebuilt_dates": len(rebuild_dates), "preserved_signals_before_start": len(preserved), "rebuilt_signals_from_top10": len(rebuilt), "combined_signals": len(combined)},
        "signals": combined,
    }
    recent = flatten_recent_outcomes(combined, limit=300)
    summary = build_summary(combined)
    write_json(REGISTRY_PATH, registry)
    write_json(OUTCOMES_PATH, {"items": recent, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")})
    write_json(SUMMARY_PATH, summary)
    log("INFO", f"Wrote registry/outcomes/summary. combined={len(combined)}, updated={outcome_changed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in rebuild_signal_outcomes: {e}")
        sys.exit(1)
