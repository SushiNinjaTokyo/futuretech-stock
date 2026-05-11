#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_weekly_screening as weekly  # noqa: E402

OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
BACKTEST_WEEKS = int(os.getenv("WEEKLY_BACKTEST_WEEKS", "4"))
BACKTEST_END_DATE = os.getenv("WEEKLY_BACKTEST_END_DATE", "").strip()
MIN_SIGNAL_LEVEL = os.getenv("WEEKLY_BACKTEST_MIN_SIGNAL", "B").strip().upper() or "B"
BENCHMARK = os.getenv("WEEKLY_BACKTEST_BENCHMARK", "SPY").strip().upper() or "SPY"

# Cache benchmark price windows by as-of date so SPY is not downloaded for every signal.
_BENCHMARK_CACHE: Dict[str, pd.DataFrame] = {}

HORIZONS = {
    "1w": 5,
    "2w": 10,
    "4w": 20,
    "8w": 40,
}

QUALIFIED_SIGNALS = {
    "A": {"A+ Fresh Breakout", "A Leader"},
    "B": {"A+ Fresh Breakout", "A Leader", "B Constructive Setup"},
    "C": {"A+ Fresh Breakout", "A Leader", "B Constructive Setup", "C Early Watch"},
}


def log(level: str, msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}",
        flush=True,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


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


def safe_round(x: Any, ndigits: int = 2) -> Optional[float]:
    v = to_float(x)
    return round(v, ndigits) if v is not None else None


def pct(cur: Any, prev: Any) -> Optional[float]:
    c = to_float(cur)
    p = to_float(prev)
    if c is None or p is None or p == 0:
        return None
    return (c / p - 1.0) * 100.0


def diff_pct(a: Any, b: Any) -> Optional[float]:
    av = to_float(a)
    bv = to_float(b)
    if av is None or bv is None:
        return None
    return av - bv


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def last_saturday_on_or_before(d: datetime) -> datetime:
    return d - timedelta(days=(d.weekday() - 5) % 7)


def get_backtest_saturdays(n: int, end_date: Optional[str]) -> List[str]:
    if end_date:
        end = parse_date(end_date)
    else:
        end = datetime.now(timezone.utc).replace(tzinfo=None)

    last_sat = last_saturday_on_or_before(end)
    sats = [last_sat - timedelta(days=7 * i) for i in range(n)]
    return [x.strftime("%Y-%m-%d") for x in reversed(sats)]


def fetch_history_window(symbol: str, start_date: str, end_date_exclusive: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Add yfinance to requirements.txt.")

    try:
        raw = yf.download(
            symbol,
            start=start_date,
            end=end_date_exclusive,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return weekly.extract_ohlcv(raw, symbol)
    except Exception as e:
        log("WARN", f"history fetch failed: {symbol}: {e}")
        return pd.DataFrame()


def fetch_for_scoring(symbol: str, as_of_saturday: str) -> pd.DataFrame:
    # Saturday snapshot should only use data available up to that weekend.
    # yfinance end is exclusive, so Saturday + 1 day captures the prior Friday close.
    end_dt = parse_date(as_of_saturday) + timedelta(days=1)
    start_dt = end_dt - timedelta(days=560)
    return fetch_history_window(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))


def fetch_for_outcome(symbol: str, as_of_saturday: str) -> pd.DataFrame:
    # Need past data for context and future data for outcome measurement.
    start_dt = parse_date(as_of_saturday) - timedelta(days=560)
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=2)
    return fetch_history_window(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))


def fetch_benchmark_for_outcome(as_of_saturday: str) -> pd.DataFrame:
    """Fetch benchmark price history for the same outcome window, cached by as-of date."""
    cache_key = f"{BENCHMARK}:{as_of_saturday}"
    if cache_key in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE[cache_key]

    df = fetch_for_outcome(BENCHMARK, as_of_saturday)
    _BENCHMARK_CACHE[cache_key] = df
    return df


def build_snapshot(as_of: str) -> Dict[str, Any]:
    candidates = [c for c in weekly.load_candidates() if not c.exclude]
    if not candidates:
        raise SystemExit("No weekly candidates found. Add rows to data/weekly_candidates.csv")

    log("INFO", f"Build weekly backtest snapshot as_of={as_of}, candidates={len(candidates)}")

    raw_items: List[Dict[str, Any]] = []
    rs_values: Dict[str, Optional[float]] = {}

    for c in candidates:
        df = fetch_for_scoring(c.symbol, as_of)
        if df.empty or len(df) < 210:
            log("WARN", f"skip {c.symbol} at {as_of}: insufficient history rows={len(df)}")
            continue

        d = weekly.add_indicators(df).dropna(subset=["Close", "Volume"])
        if len(d) < 210:
            log("WARN", f"skip {c.symbol} at {as_of}: insufficient indicator rows={len(d)}")
            continue

        trend_score, trend = weekly.score_trend(d)
        breakout_score, breakout = weekly.score_breakout(d)
        setup_score, setup = weekly.score_setup(d)
        volume_score, volume = weekly.score_volume(d)
        fund_score, fund = weekly.score_fundamental(weekly.get_info(c.symbol))
        risk_score, risk = weekly.score_risk(d, breakout)
        returns = weekly.calc_returns(d)
        rs_comp = weekly.rs_composite(returns)
        rs_values[c.symbol] = rs_comp

        raw_items.append({
            "symbol": c.symbol,
            "name": c.name,
            "source": c.source,
            "theme": c.theme,
            "trigger": c.trigger,
            "note": c.note,
            "priority": c.priority,
            "trend_score": trend_score,
            "breakout_score": breakout_score,
            "setup_score": setup_score,
            "volume_score": volume_score,
            "fundamental_score": fund_score,
            "risk_score": risk_score,
            "trend": trend,
            "breakout": breakout,
            "setup": setup,
            "volume": volume,
            "fundamental": fund,
            "risk": risk,
            "returns": returns,
            "rs_composite": safe_round(rs_comp, 4),
        })

    if not raw_items:
        return {
            "as_of": as_of,
            "items": [],
            "qualified": [],
            "summary": {
                "valid_items": 0,
                "qualified_signals": 0,
                "fresh_breakouts": 0,
                "leaders": 0,
                "constructive_setups": 0,
            },
        }

    percentiles = weekly.percentile_scores(rs_values)

    for item in raw_items:
        sym = item["symbol"]
        rs_pct = percentiles.get(sym, 0.0)
        rs_score = weekly.map_rs_score(rs_pct)

        item["rs_score"] = rs_score
        item["rs_percentile"] = round(rs_pct, 1)

        total = int(
            item["trend_score"]
            + item["rs_score"]
            + item["breakout_score"]
            + item["setup_score"]
            + item["volume_score"]
            + item["fundamental_score"]
            + item["risk_score"]
        )

        item["weekly_score"] = total
        item["signal"] = weekly.make_signal(
            total,
            item["trend_score"],
            item["rs_score"],
            item["breakout_score"],
            item["setup_score"],
            item["volume_score"],
            item["risk"],
        )
        item["comment"] = weekly.make_comment(item)

    signal_order = {
        "A+ Fresh Breakout": 0,
        "A Leader": 1,
        "B Constructive Setup": 2,
        "C Early Watch": 3,
        "D Extended": 4,
        "E Avoid": 5,
    }

    items = sorted(raw_items, key=lambda x: (signal_order.get(x["signal"], 9), -x["weekly_score"], -x.get("rs_percentile", 0)))

    for i, item in enumerate(items, 1):
        item["rank"] = i

    allowed = QUALIFIED_SIGNALS.get(MIN_SIGNAL_LEVEL, QUALIFIED_SIGNALS["B"])
    qualified = [x for x in items if x["signal"] in allowed]

    return {
        "as_of": as_of,
        "items": items,
        "qualified": qualified,
        "summary": {
            "valid_items": len(items),
            "qualified_signals": len(qualified),
            "fresh_breakouts": sum(1 for x in qualified if x["signal"] == "A+ Fresh Breakout"),
            "leaders": sum(1 for x in qualified if x["signal"] == "A Leader"),
            "constructive_setups": sum(1 for x in qualified if x["signal"] == "B Constructive Setup"),
        },
    }


def next_trading_row_after(df: pd.DataFrame, as_of: str) -> Optional[int]:
    if df.empty:
        return None

    dt = pd.Timestamp(as_of)

    for i, ts in enumerate(df.index):
        if pd.Timestamp(ts).normalize() > dt:
            return i

    return None


def empty_outcome(status: str) -> Dict[str, Any]:
    return {
        "entry_date": None,
        "entry_price": None,
        "status": status,
        "return_1w_pct": None,
        "return_2w_pct": None,
        "return_4w_pct": None,
        "return_8w_pct": None,
        "current_return_pct": None,
        "max_gain_since_entry_pct": None,
        "max_drawdown_since_entry_pct": None,
    }


def outcome_for_signal(row: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    symbol = row["symbol"]
    df = fetch_for_outcome(symbol, as_of)

    if df.empty:
        return empty_outcome("missing_prices")

    entry_pos = next_trading_row_after(df, as_of)
    if entry_pos is None or entry_pos >= len(df):
        return empty_outcome("pending_entry")

    entry = df.iloc[entry_pos]
    entry_price = to_float(entry.get("Open")) or to_float(entry.get("Close"))

    if entry_price is None or entry_price <= 0:
        out = empty_outcome("missing_entry")
        out["entry_date"] = df.index[entry_pos].date().isoformat()
        return out

    out: Dict[str, Any] = {"entry_date": df.index[entry_pos].date().isoformat(), "entry_price": safe_round(entry_price, 2)}

    completed = []
    for label, offset in HORIZONS.items():
        key = f"return_{label}_pct"
        target_pos = entry_pos + offset

        if target_pos < len(df):
            out[key] = safe_round(pct(df["Close"].iloc[target_pos], entry_price), 2)
            completed.append(label)
        else:
            out[key] = None

    freeze_pos = entry_pos + HORIZONS["8w"]
    current_pos = min(freeze_pos, len(df) - 1)
    out["current_return_pct"] = safe_round(pct(df["Close"].iloc[current_pos], entry_price), 2)

    window = df.iloc[entry_pos: current_pos + 1]
    if not window.empty:
        hi = window["High"].max() if "High" in window else window["Close"].max()
        lo = window["Low"].min() if "Low" in window else window["Close"].min()
        out["max_gain_since_entry_pct"] = safe_round(pct(hi, entry_price), 2)
        out["max_drawdown_since_entry_pct"] = safe_round(pct(lo, entry_price), 2)
    else:
        out["max_gain_since_entry_pct"] = None
        out["max_drawdown_since_entry_pct"] = None

    if "8w" in completed:
        status = "completed_8w"
    elif "4w" in completed:
        status = "completed_4w"
    elif "2w" in completed:
        status = "completed_2w"
    elif "1w" in completed:
        status = "completed_1w"
    else:
        status = "active"

    out["status"] = status
    return out


def benchmark_outcome_for_signal(as_of: str) -> Dict[str, Any]:
    df = fetch_benchmark_for_outcome(as_of)
    empty = {
        "benchmark_symbol": BENCHMARK,
        "benchmark_entry_date": None,
        "benchmark_entry_price": None,
        "benchmark_return_1w_pct": None,
        "benchmark_return_2w_pct": None,
        "benchmark_return_4w_pct": None,
        "benchmark_return_8w_pct": None,
        "benchmark_current_return_pct": None,
    }

    if df.empty:
        return empty

    entry_pos = next_trading_row_after(df, as_of)
    if entry_pos is None or entry_pos >= len(df):
        return empty

    entry = df.iloc[entry_pos]
    entry_price = to_float(entry.get("Open")) or to_float(entry.get("Close"))
    if entry_price is None or entry_price <= 0:
        out = dict(empty)
        out["benchmark_entry_date"] = df.index[entry_pos].date().isoformat()
        return out

    out = {
        "benchmark_symbol": BENCHMARK,
        "benchmark_entry_date": df.index[entry_pos].date().isoformat(),
        "benchmark_entry_price": safe_round(entry_price, 2),
        "benchmark_return_1w_pct": None,
        "benchmark_return_2w_pct": None,
        "benchmark_return_4w_pct": None,
        "benchmark_return_8w_pct": None,
        "benchmark_current_return_pct": None,
    }

    for label, offset in HORIZONS.items():
        key = f"benchmark_return_{label}_pct"
        target_pos = entry_pos + offset
        if target_pos < len(df):
            out[key] = safe_round(pct(df["Close"].iloc[target_pos], entry_price), 2)

    freeze_pos = entry_pos + HORIZONS["8w"]
    current_pos = min(freeze_pos, len(df) - 1)
    out["benchmark_current_return_pct"] = safe_round(pct(df["Close"].iloc[current_pos], entry_price), 2)
    return out


def signal_to_backtest_row(snapshot_row: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    outcome = outcome_for_signal(snapshot_row, as_of)
    benchmark = benchmark_outcome_for_signal(as_of)
    bo = snapshot_row.get("breakout") or {}

    return {
        "signal_date": as_of,
        "rank": snapshot_row.get("rank"),
        "weekly_score": snapshot_row.get("weekly_score"),
        "symbol": snapshot_row.get("symbol"),
        "name": snapshot_row.get("name"),
        "theme": snapshot_row.get("theme"),
        "source": snapshot_row.get("source"),
        "trigger": snapshot_row.get("trigger"),
        "signal": snapshot_row.get("signal"),
        "trend_score": snapshot_row.get("trend_score"),
        "rs_score": snapshot_row.get("rs_score"),
        "rs_percentile": snapshot_row.get("rs_percentile"),
        "breakout_score": snapshot_row.get("breakout_score"),
        "setup_score": snapshot_row.get("setup_score"),
        "volume_score": snapshot_row.get("volume_score"),
        "fundamental_score": snapshot_row.get("fundamental_score"),
        "risk_score": snapshot_row.get("risk_score"),
        "breakout_type": bo.get("breakout_type"),
        "days_since_breakout": bo.get("days_since_breakout"),
        "price_from_breakout_pct": bo.get("price_from_breakout_pct"),
        **outcome,
        **benchmark,
    }


def avg(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def win_rate(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(1 for v in vals if v > 0) / len(vals)


def bucket_summary(rows: List[Dict[str, Any]], *args: Any) -> List[Dict[str, Any]]:
    if len(args) == 1:
        labels = args[0]
    elif len(args) == 2:
        _key = args[0]
        labels = args[1]
    else:
        raise TypeError("bucket_summary expects rows + labels, or rows + key + labels")

    out = []
    for label, predicate in labels:
        subset = [r for r in rows if predicate(r)]
        if not subset:
            continue
        out.append({
            "label": label,
            "count": len(subset),
            "avg_current_return": safe_round(avg([r.get("current_return_pct") for r in subset]), 2),
            "avg_return_1w": safe_round(avg([r.get("return_1w_pct") for r in subset]), 2),
            "avg_return_2w": safe_round(avg([r.get("return_2w_pct") for r in subset]), 2),
            "avg_return_4w": safe_round(avg([r.get("return_4w_pct") for r in subset]), 2),
            "avg_return_8w": safe_round(avg([r.get("return_8w_pct") for r in subset]), 2),
            "avg_benchmark_current_return": safe_round(avg([r.get("benchmark_current_return_pct") for r in subset]), 2),
            "avg_benchmark_return_1w": safe_round(avg([r.get("benchmark_return_1w_pct") for r in subset]), 2),
            "avg_benchmark_return_2w": safe_round(avg([r.get("benchmark_return_2w_pct") for r in subset]), 2),
            "avg_benchmark_return_4w": safe_round(avg([r.get("benchmark_return_4w_pct") for r in subset]), 2),
            "avg_benchmark_return_8w": safe_round(avg([r.get("benchmark_return_8w_pct") for r in subset]), 2),
            "avg_alpha_current": safe_round(avg([diff_pct(r.get("current_return_pct"), r.get("benchmark_current_return_pct")) for r in subset]), 2),
            "avg_alpha_1w": safe_round(avg([diff_pct(r.get("return_1w_pct"), r.get("benchmark_return_1w_pct")) for r in subset]), 2),
            "avg_alpha_4w": safe_round(avg([diff_pct(r.get("return_4w_pct"), r.get("benchmark_return_4w_pct")) for r in subset]), 2),
            "win_rate_1w": safe_round(win_rate([r.get("return_1w_pct") for r in subset]), 4),
            "win_rate_4w": safe_round(win_rate([r.get("return_4w_pct") for r in subset]), 4),
            "win_rate_8w": safe_round(win_rate([r.get("return_8w_pct") for r in subset]), 4),
        })
    return out


def build_summary(rows: List[Dict[str, Any]], snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    entered_rows = [
        r for r in rows
        if r.get("entry_price") is not None and r.get("status") not in {"pending_entry", "missing_entry", "missing_prices"}
    ]
    pending_rows = [r for r in rows if r.get("status") == "pending_entry" or r.get("entry_price") is None]

    return {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "snapshot_count": len(snapshots),
        "total_signals": total,
        "entered_signals": len(entered_rows),
        "pending_entries": len(pending_rows),
        "active_signals": sum(1 for r in entered_rows if str(r.get("status", "")).startswith("active")),
        "benchmark_symbol": BENCHMARK,
        "avg_current_return": safe_round(avg([r.get("current_return_pct") for r in entered_rows]), 2),
        "avg_return_1w": safe_round(avg([r.get("return_1w_pct") for r in entered_rows]), 2),
        "avg_return_2w": safe_round(avg([r.get("return_2w_pct") for r in entered_rows]), 2),
        "avg_return_4w": safe_round(avg([r.get("return_4w_pct") for r in entered_rows]), 2),
        "avg_return_8w": safe_round(avg([r.get("return_8w_pct") for r in entered_rows]), 2),
        "avg_benchmark_current_return": safe_round(avg([r.get("benchmark_current_return_pct") for r in entered_rows]), 2),
        "avg_benchmark_return_1w": safe_round(avg([r.get("benchmark_return_1w_pct") for r in entered_rows]), 2),
        "avg_benchmark_return_2w": safe_round(avg([r.get("benchmark_return_2w_pct") for r in entered_rows]), 2),
        "avg_benchmark_return_4w": safe_round(avg([r.get("benchmark_return_4w_pct") for r in entered_rows]), 2),
        "avg_benchmark_return_8w": safe_round(avg([r.get("benchmark_return_8w_pct") for r in entered_rows]), 2),
        "avg_alpha_current": safe_round(avg([diff_pct(r.get("current_return_pct"), r.get("benchmark_current_return_pct")) for r in entered_rows]), 2),
        "avg_alpha_1w": safe_round(avg([diff_pct(r.get("return_1w_pct"), r.get("benchmark_return_1w_pct")) for r in entered_rows]), 2),
        "avg_alpha_2w": safe_round(avg([diff_pct(r.get("return_2w_pct"), r.get("benchmark_return_2w_pct")) for r in entered_rows]), 2),
        "avg_alpha_4w": safe_round(avg([diff_pct(r.get("return_4w_pct"), r.get("benchmark_return_4w_pct")) for r in entered_rows]), 2),
        "avg_alpha_8w": safe_round(avg([diff_pct(r.get("return_8w_pct"), r.get("benchmark_return_8w_pct")) for r in entered_rows]), 2),
        "win_rate_1w": safe_round(win_rate([r.get("return_1w_pct") for r in entered_rows]), 4),
        "win_rate_4w": safe_round(win_rate([r.get("return_4w_pct") for r in entered_rows]), 4),
        "win_rate_8w": safe_round(win_rate([r.get("return_8w_pct") for r in entered_rows]), 4),
        "completed_1w": sum(1 for r in entered_rows if r.get("return_1w_pct") is not None),
        "completed_2w": sum(1 for r in entered_rows if r.get("return_2w_pct") is not None),
        "completed_4w": sum(1 for r in entered_rows if r.get("return_4w_pct") is not None),
        "completed_8w": sum(1 for r in entered_rows if r.get("return_8w_pct") is not None),
        "avg_max_gain": safe_round(avg([r.get("max_gain_since_entry_pct") for r in entered_rows]), 2),
        "avg_max_drawdown": safe_round(avg([r.get("max_drawdown_since_entry_pct") for r in entered_rows]), 2),
        "signal_buckets": bucket_summary(entered_rows, "signal", [
            ("A+ Fresh Breakout", lambda r: r.get("signal") == "A+ Fresh Breakout"),
            ("A Leader", lambda r: r.get("signal") == "A Leader"),
            ("B Constructive Setup", lambda r: r.get("signal") == "B Constructive Setup"),
        ]),
        "score_buckets": bucket_summary(entered_rows, "score", [
            ("850+", lambda r: (to_float(r.get("weekly_score")) or 0) >= 850),
            ("800-849", lambda r: 800 <= (to_float(r.get("weekly_score")) or 0) < 850),
            ("750-799", lambda r: 750 <= (to_float(r.get("weekly_score")) or 0) < 800),
            ("700-749", lambda r: 700 <= (to_float(r.get("weekly_score")) or 0) < 750),
            ("<700", lambda r: (to_float(r.get("weekly_score")) or 0) < 700),
        ]),
        "rank_buckets": bucket_summary(entered_rows, "rank", [
            ("Rank 1", lambda r: r.get("rank") == 1),
            ("Rank 2-3", lambda r: r.get("rank") in {2, 3}),
            ("Rank 4-10", lambda r: isinstance(r.get("rank"), int) and 4 <= r.get("rank") <= 10),
            ("Rank 11+", lambda r: isinstance(r.get("rank"), int) and r.get("rank") >= 11),
        ]),
    }


def main() -> None:
    saturdays = get_backtest_saturdays(BACKTEST_WEEKS, BACKTEST_END_DATE or None)
    log("INFO", f"Weekly backtest weeks={BACKTEST_WEEKS}, snapshots={saturdays}, min_signal={MIN_SIGNAL_LEVEL}, benchmark={BENCHMARK}")
    snapshots = []
    rows: List[Dict[str, Any]] = []
    for as_of in saturdays:
        snapshot = build_snapshot(as_of)
        snapshots.append({"as_of": as_of, "summary": snapshot.get("summary", {})})
        for q in snapshot.get("qualified", []):
            rows.append(signal_to_backtest_row(q, as_of))
    rows = sorted(rows, key=lambda r: (r.get("signal_date") or "", -(r.get("rank") or 999)), reverse=True)
    summary = build_summary(rows, snapshots)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "methodology": {
            "name": "Weekly Minervini-style Signal Backtest",
            "min_signal": MIN_SIGNAL_LEVEL,
            "benchmark": BENCHMARK,
            "qualified_signals": sorted(list(QUALIFIED_SIGNALS.get(MIN_SIGNAL_LEVEL, QUALIFIED_SIGNALS["B"]))),
            "entry": "next trading day open after Saturday snapshot",
            "horizons": HORIZONS,
            "current_return_policy": "updates until 8W completion, then freezes at 8W close",
            "aggregate_policy": "pending and missing-entry signals are shown in the table but excluded from averages and buckets",
            "benchmark_policy": "benchmark references use the same entry date and holding window as each signal",
        },
        "snapshots": snapshots,
        "summary": summary,
        "recent": rows,
    }
    out_dir = OUT_DIR / "data" / "weekly" / "backtest"
    latest_path = out_dir / "latest.json"
    dated_path = out_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    write_json(latest_path, payload)
    write_json(dated_path, payload)
    log("INFO", f"Wrote {latest_path}")
    log("INFO", f"Wrote {dated_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in rebuild_weekly_backtest: {e}")
        sys.exit(1)
