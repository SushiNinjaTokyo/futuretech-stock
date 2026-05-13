#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
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
BENCHMARKS = {"spy": os.getenv("DAILY_V2_BENCHMARK_SPY", "SPY"), "qqq": os.getenv("DAILY_V2_BENCHMARK_QQQ", "QQQ")}
HORIZONS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "20d": 20}
STOP_LEVELS = [3.0, 5.0, 8.0]
FOLLOW_THROUGH_PCT = float(os.getenv("DAILY_V2_FOLLOW_THROUGH_PCT", "5.0"))
FOLLOW_THROUGH_DAYS = int(os.getenv("DAILY_V2_FOLLOW_THROUGH_DAYS", "3"))
SLEEP_SECONDS = float(os.getenv("DAILY_V2_BACKTEST_SLEEP_SECONDS", "0.6"))


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def safe_round(x: Any, n: int = 2) -> Optional[float]:
    v = to_float(x)
    return round(v, n) if v is not None else None


def pct(cur: Any, prev: Any) -> Optional[float]:
    c = to_float(cur)
    p = to_float(prev)
    if c is None or p is None or p == 0:
        return None
    return (c / p - 1.0) * 100.0


def avg(vals: List[Any]) -> Optional[float]:
    xs = [float(v) for v in vals if to_float(v) is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def rate(vals: List[Any]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return sum(1 for v in xs if bool(v)) / len(xs)


def normalize_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol]
        elif symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, level=-1, axis=1)
        elif "Close" in df.columns.get_level_values(0):
            tickers = list(df.columns.get_level_values(1).unique())
            df = df.xs(tickers[0], level=1, axis=1)
        else:
            df = df.droplevel(0, axis=1)
    cols = {str(c).lower(): c for c in df.columns}
    out = pd.DataFrame(index=pd.to_datetime(df.index))
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        src = cols.get(c.lower())
        out[c] = pd.to_numeric(df[src], errors="coerce") if src is not None else np.nan
    return out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    s = pd.Timestamp(start) - pd.Timedelta(days=10)
    e = pd.Timestamp(end) + pd.Timedelta(days=45)
    for attempt in range(3):
        try:
            raw = yf.download(symbol, start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"), interval="1d", progress=False, auto_adjust=False, threads=False)
            df = normalize_ohlcv(raw, symbol)
            if not df.empty:
                return df
            raise ValueError("empty history")
        except Exception as exc:
            log("WARN", f"{symbol}: fetch attempt {attempt+1} failed: {exc}")
            time.sleep(1.2 + attempt)
    return pd.DataFrame()


def next_pos_after(df: pd.DataFrame, signal_date: str) -> Optional[int]:
    d = pd.Timestamp(signal_date).normalize()
    for i, ts in enumerate(df.index):
        if pd.Timestamp(ts).normalize() > d:
            return i
    return None


def load_daily_v2_dates(start: str, end: Optional[str]) -> List[Path]:
    base = OUT_DIR / "data" / "daily-v2"
    if not base.exists():
        return []
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize() if end else pd.Timestamp.utcnow().normalize()
    paths = []
    for p in sorted(base.glob("????-??-??/top10.json")):
        try:
            d = pd.Timestamp(p.parent.name).normalize()
        except Exception:
            continue
        if s <= d <= e:
            paths.append(p)
    return paths


def should_register(row: Dict[str, Any]) -> bool:
    return row.get("triage") in {"Trade", "Watch"} and (to_float(row.get("score_pts")) or 0) >= 650


def benchmark_returns(df: pd.DataFrame, entry_pos: int, entry_price: float) -> Dict[str, Optional[float]]:
    out = {}
    for label, off in HORIZONS.items():
        idx = entry_pos + off
        out[f"return_{label}_pct"] = safe_round(pct(df["Close"].iloc[idx], entry_price), 2) if idx < len(df) else None
    return out


def build_path(df: pd.DataFrame, entry_pos: int, entry_price: float, max_days: int = 20) -> List[Optional[float]]:
    vals = []
    for i in range(max_days + 1):
        idx = entry_pos + i
        vals.append(safe_round(pct(df["Close"].iloc[idx], entry_price), 2) if idx < len(df) else None)
    return vals


def outcome_for(row: Dict[str, Any], df: pd.DataFrame, bm: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    sig_date = row.get("signal_date") or row.get("date") or row.get("as_of")
    empty = {"entry_date": None, "entry_price": None, "status": "pending_entry"}
    if df.empty or not sig_date:
        return {**empty, "status": "missing_prices"}
    entry_pos = next_pos_after(df, sig_date)
    if entry_pos is None:
        return empty
    entry = df.iloc[entry_pos]
    entry_price = to_float(entry.get("Open")) or to_float(entry.get("Close"))
    if entry_price is None or entry_price <= 0:
        return {**empty, "status": "missing_entry", "entry_date": df.index[entry_pos].date().isoformat()}
    out: Dict[str, Any] = {
        "entry_date": df.index[entry_pos].date().isoformat(),
        "entry_price": safe_round(entry_price, 2),
        "status": "active",
    }
    completed = []
    for label, off in HORIZONS.items():
        idx = entry_pos + off
        key = f"return_{label}_pct"
        if idx < len(df):
            out[key] = safe_round(pct(df["Close"].iloc[idx], entry_price), 2)
            completed.append(label)
        else:
            out[key] = None
    end_pos = min(entry_pos + 20, len(df) - 1)
    window = df.iloc[entry_pos:end_pos + 1]
    max_high = float(window["High"].max()) if not window.empty else entry_price
    min_low = float(window["Low"].min()) if not window.empty else entry_price
    out["mfe_pct"] = safe_round(pct(max_high, entry_price), 2)
    out["mae_pct"] = safe_round(pct(min_low, entry_price), 2)
    out["max_gain_since_entry_pct"] = out["mfe_pct"]
    out["max_drawdown_since_entry_pct"] = out["mae_pct"]
    out["stop_hit_3_pct"] = bool((window["Low"] <= entry_price * 0.97).any()) if not window.empty else False
    out["stop_hit_5_pct"] = bool((window["Low"] <= entry_price * 0.95).any()) if not window.empty else False
    out["stop_hit_8_pct"] = bool((window["Low"] <= entry_price * 0.92).any()) if not window.empty else False
    early = df.iloc[entry_pos:min(entry_pos + FOLLOW_THROUGH_DAYS + 1, len(df))]
    out["follow_through"] = bool((early["High"] >= entry_price * (1 + FOLLOW_THROUGH_PCT / 100)).any()) if not early.empty else False
    if not window.empty:
        peak_i = int(window["High"].values.argmax())
        out["peak_day"] = peak_i
        out["peak_return_pct"] = safe_round(pct(window["High"].iloc[peak_i], entry_price), 2)
    out["path_pct"] = build_path(df, entry_pos, entry_price, 20)

    for bkey, bdf in bm.items():
        bpos = next_pos_after(bdf, sig_date) if not bdf.empty else None
        if bpos is None:
            continue
        bentry = to_float(bdf.iloc[bpos].get("Open")) or to_float(bdf.iloc[bpos].get("Close"))
        if not bentry:
            continue
        bret = benchmark_returns(bdf, bpos, bentry)
        for label in HORIZONS:
            rv = out.get(f"return_{label}_pct")
            bv = bret.get(f"return_{label}_pct")
            out[f"{bkey}_return_{label}_pct"] = bv
            out[f"alpha_{bkey}_{label}_pct"] = safe_round(rv - bv, 2) if rv is not None and bv is not None else None
        out[f"{bkey}_path_pct"] = build_path(bdf, bpos, bentry, 20)
    if "20d" in completed:
        out["status"] = "completed_20d"
    elif completed:
        out["status"] = f"completed_{completed[-1]}"
    return out


def strategy_return(row: Dict[str, Any], name: str) -> Optional[float]:
    path = row.get("path_pct") or []
    mfe = to_float(row.get("mfe_pct"))
    if name == "Hold 5D":
        return row.get("return_5d_pct")
    if name == "Hold 10D":
        return row.get("return_10d_pct")
    if name == "Hold 20D":
        return row.get("return_20d_pct")
    if name == "Stop -3%":
        return -3.0 if row.get("stop_hit_3_pct") else row.get("return_10d_pct")
    if name == "Stop -5%":
        return -5.0 if row.get("stop_hit_5_pct") else row.get("return_10d_pct")
    if name == "Stop -8%":
        return -8.0 if row.get("stop_hit_8_pct") else row.get("return_10d_pct")
    if name == "Profit Lock 10/5":
        if mfe is not None and mfe >= 10:
            # Conservative approximation: keep at least +5% once +10% was available.
            return max(5.0, to_float(row.get("return_10d_pct")) or 5.0)
        return row.get("return_10d_pct")
    if name == "Profit Lock 15/7":
        if mfe is not None and mfe >= 15:
            return max(7.0, to_float(row.get("return_10d_pct")) or 7.0)
        return row.get("return_10d_pct")
    if name == "Score+Volume Only":
        if (to_float(row.get("score_pts")) or 0) >= 750 and (to_float(row.get("volume_anomaly")) or 0) >= 0.55:
            return row.get("return_10d_pct")
        return None
    if name == "Trade Triage":
        if row.get("triage") == "Trade":
            return row.get("return_10d_pct")
        return None
    return None


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    entered = [r for r in rows if r.get("entry_price") is not None]
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "total_signals": len(rows),
        "entered_signals": len(entered),
        "pending_entries": len(rows) - len(entered),
        "avg_return_1d": safe_round(avg([r.get("return_1d_pct") for r in entered]), 2),
        "avg_return_3d": safe_round(avg([r.get("return_3d_pct") for r in entered]), 2),
        "avg_return_5d": safe_round(avg([r.get("return_5d_pct") for r in entered]), 2),
        "avg_return_10d": safe_round(avg([r.get("return_10d_pct") for r in entered]), 2),
        "avg_return_20d": safe_round(avg([r.get("return_20d_pct") for r in entered]), 2),
        "avg_spy_alpha_5d": safe_round(avg([r.get("alpha_spy_5d_pct") for r in entered]), 2),
        "avg_qqq_alpha_5d": safe_round(avg([r.get("alpha_qqq_5d_pct") for r in entered]), 2),
        "follow_through_rate": safe_round(rate([r.get("follow_through") for r in entered]), 4),
        "stop_hit_5_rate": safe_round(rate([r.get("stop_hit_5_pct") for r in entered]), 4),
        "avg_mfe": safe_round(avg([r.get("mfe_pct") for r in entered]), 2),
        "avg_mae": safe_round(avg([r.get("mae_pct") for r in entered]), 2),
    }
    if summary["avg_mfe"] is not None and summary["avg_mae"] not in (None, 0):
        summary["mfe_mae_ratio"] = safe_round(abs(summary["avg_mfe"] / summary["avg_mae"]), 2)
    else:
        summary["mfe_mae_ratio"] = None
    return summary


def bucket(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or "Unknown")].append(r)
    out = []
    for label, xs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append({
            "label": label,
            "count": len(xs),
            "avg_5d": safe_round(avg([x.get("return_5d_pct") for x in xs]), 2),
            "avg_10d": safe_round(avg([x.get("return_10d_pct") for x in xs]), 2),
            "spy_alpha_5d": safe_round(avg([x.get("alpha_spy_5d_pct") for x in xs]), 2),
            "qqq_alpha_5d": safe_round(avg([x.get("alpha_qqq_5d_pct") for x in xs]), 2),
            "follow_through_rate": safe_round(rate([x.get("follow_through") for x in xs]), 4),
            "stop_hit_rate": safe_round(rate([x.get("stop_hit_5_pct") for x in xs]), 4),
            "avg_mfe": safe_round(avg([x.get("mfe_pct") for x in xs]), 2),
            "avg_mae": safe_round(avg([x.get("mae_pct") for x in xs]), 2),
        })
    return out


def heatmap(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    score_bins = [("650-699", 650, 700), ("700-749", 700, 750), ("750-799", 750, 800), ("800+", 800, 10_000)]
    vol_bins = [("<0.45", 0, 0.45), ("0.45-0.54", 0.45, 0.55), ("0.55-0.74", 0.55, 0.75), ("0.75+", 0.75, 99)]
    out = []
    for sl, slo, shi in score_bins:
        for vl, vlo, vhi in vol_bins:
            xs = [r for r in rows if slo <= (to_float(r.get("score_pts")) or 0) < shi and vlo <= (to_float(r.get("volume_anomaly")) or 0) < vhi]
            out.append({"score_band": sl, "volume_band": vl, "count": len(xs), "avg_5d": safe_round(avg([x.get("return_5d_pct") for x in xs]), 2), "follow_through_rate": safe_round(rate([x.get("follow_through") for x in xs]), 4)})
    return out


def average_path(rows: List[Dict[str, Any]], path_key: str) -> List[Dict[str, Any]]:
    out = []
    for day in range(21):
        vals = []
        for r in rows:
            p = r.get(path_key) or []
            if len(p) > day and p[day] is not None:
                vals.append(p[day])
        out.append({"day": day, "value": safe_round(avg(vals), 2)})
    return out


def strategy_comparison(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = ["Hold 5D", "Hold 10D", "Hold 20D", "Stop -3%", "Stop -5%", "Stop -8%", "Profit Lock 10/5", "Profit Lock 15/7", "Score+Volume Only", "Trade Triage"]
    out = []
    for name in names:
        vals = [strategy_return(r, name) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            out.append({
                "name": name,
                "trades": 0,
                "avg_return_pct": None,
                "max_drawdown_pct": None,
                "win_rate": None,
                "return_drawdown_ratio": None,
            })
            continue
        max_dd = min(vals)
        ret = avg(vals)
        out.append({
            "name": name,
            "trades": len(vals),
            "avg_return_pct": safe_round(ret, 2),
            "max_drawdown_pct": safe_round(max_dd, 2),
            "win_rate": safe_round(sum(1 for v in vals if v > 0) / len(vals), 4),
            "return_drawdown_ratio": safe_round((ret / abs(max_dd)) if ret is not None and max_dd < 0 else None, 2),
        })
    return out


def peak_distribution(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = [("Day 0-1", 0, 1), ("Day 2-3", 2, 3), ("Day 4-5", 4, 5), ("Day 6-10", 6, 10), ("Day 11-20", 11, 20)]
    out = []
    total = len([r for r in rows if r.get("peak_day") is not None])
    for label, lo, hi in buckets:
        c = sum(1 for r in rows if r.get("peak_day") is not None and lo <= int(r.get("peak_day")) <= hi)
        out.append({"label": label, "count": c, "rate": safe_round(c / total if total else None, 4)})
    return out


def loser_diagnostics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    losers = [r for r in rows if (to_float(r.get("return_5d_pct")) or 0) < 0]
    tests = [
        ("News-heavy", lambda r: (r.get("archetype") or "").find("Catalyst") >= 0),
        ("Weak close", lambda r: (to_float(r.get("range_pos")) or 1) < 0.5),
        ("Extended", lambda r: (to_float(r.get("extension_sma20_pct")) or 0) > 15),
        ("Risk-off/Neutral", lambda r: r.get("regime") != "Risk-on"),
        ("Low compression", lambda r: (to_float(r.get("compression_release")) or 0) < 0.45),
    ]
    out = []
    for label, fn in tests:
        c = sum(1 for r in losers if fn(r))
        out.append({"label": label, "count": c, "rate": safe_round(c / len(losers) if losers else None, 4)})
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", default="")
    args = ap.parse_args(argv)
    paths = load_daily_v2_dates(args.start_date, args.end_date or None)
    if not paths:
        raise SystemExit("No daily-v2 top10.json files found for requested range")
    # collect signals
    signals: List[Dict[str, Any]] = []
    for p in paths:
        j = read_json(p) or {}
        date = j.get("date") or p.parent.name
        for item in j.get("items", [])[:25]:
            if not isinstance(item, dict) or not should_register(item):
                continue
            signals.append({
                "signal_date": date,
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score_pts": item.get("score_pts"),
                "triage": item.get("triage"),
                "archetype": item.get("archetype"),
                "regime": item.get("regime"),
                "reason": item.get("reason"),
                "volume_anomaly": (item.get("score_components") or {}).get("volume_anomaly"),
                "compression_release": (item.get("score_components") or {}).get("compression_release"),
                "setup_quality": (item.get("score_components") or {}).get("trends_breakout"),
                "news": (item.get("score_components") or {}).get("news"),
                "range_pos": (item.get("metrics") or {}).get("range_pos"),
                "extension_sma20_pct": (item.get("metrics") or {}).get("extension_sma20_pct"),
            })
    symbols = sorted({s["symbol"] for s in signals if s.get("symbol")})
    start = min(s["signal_date"] for s in signals)
    end = max(s["signal_date"] for s in signals)
    histories: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        log("INFO", f"fetch outcome history {i}/{len(symbols)} {sym}")
        histories[sym] = fetch_history(sym, start, end)
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
    bm = {k: fetch_history(sym, start, end) for k, sym in BENCHMARKS.items()}
    rows: List[Dict[str, Any]] = []
    for sig in signals:
        sym = sig.get("symbol")
        out = outcome_for(sig, histories.get(sym, pd.DataFrame()), bm)
        rows.append({**sig, **out})
    rows.sort(key=lambda r: (r.get("signal_date") or "", -(r.get("score_pts") or 0)), reverse=True)
    entered = [r for r in rows if r.get("entry_price") is not None]
    payload = {
        "version": "daily_event_score_v2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "range": {"start_date": args.start_date, "end_date": args.end_date or end, "source_files": len(paths)},
        "methodology": {"name": "Daily Event Signal Lab", "horizons": HORIZONS, "benchmarks": BENCHMARKS, "follow_through": {"pct": FOLLOW_THROUGH_PCT, "days": FOLLOW_THROUGH_DAYS}},
        "summary": summarize_rows(rows),
        "recent": rows,
        "rule_buckets": bucket(entered, "triage"),
        "profile_buckets": bucket(entered, "archetype"),
        "regime_buckets": bucket(entered, "regime"),
        "score_volume_heatmap": heatmap(entered),
        "average_signal_path": average_path(entered, "path_pct"),
        "average_spy_path": average_path(entered, "spy_path_pct"),
        "average_qqq_path": average_path(entered, "qqq_path_pct"),
        "strategy_comparison": strategy_comparison(entered),
        "peak_distribution": peak_distribution(entered),
        "loser_diagnostics": loser_diagnostics(entered),
    }
    out_dir = OUT_DIR / "data" / "signals-v2"
    write_json(out_dir / "outcomes_latest.json", payload)
    write_json(out_dir / "summary_latest.json", payload.get("summary", {}))
    write_json(out_dir / "registry.json", {"version": "daily_event_score_v2", "signals": [{"signal_date": r.get("signal_date"), "symbol": r.get("symbol"), "score_pts": r.get("score_pts"), "triage": r.get("triage")} for r in rows]})
    # Legacy-compatible copy for existing links/debug, but v1 registry is not overwritten.
    log("INFO", f"Wrote {out_dir / 'outcomes_latest.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in rebuild_signal_outcomes_v2: {e}")
        sys.exit(1)
