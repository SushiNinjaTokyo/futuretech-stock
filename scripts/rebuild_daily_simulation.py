#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
if not OUT_DIR.is_absolute():
    OUT_DIR = (ROOT / OUT_DIR).resolve()
else:
    OUT_DIR = OUT_DIR.resolve()

DAILY_V2_DIR = OUT_DIR / "data" / "daily-v2"
SIM_DIR = OUT_DIR / "data" / "daily" / "simulation"

START_DATE = os.getenv("DAILY_SIM_START_DATE", os.getenv("START_DATE", "")).strip()
END_DATE = os.getenv("DAILY_SIM_END_DATE", os.getenv("END_DATE", "")).strip()
BENCHMARK_1 = os.getenv("DAILY_SIM_BENCHMARK", "QQQ").strip().upper() or "QQQ"
BENCHMARK_2 = os.getenv("DAILY_SIM_SECONDARY_BENCHMARK", "SPY").strip().upper() or "SPY"
INITIAL_CAPITAL = float(os.getenv("DAILY_SIM_INITIAL_CAPITAL", "10000") or "10000")
BUY_SLIPPAGE = float(os.getenv("DAILY_SIM_BUY_SLIPPAGE_PCT", "0.10") or "0.10") / 100.0
SELL_SLIPPAGE = float(os.getenv("DAILY_SIM_SELL_SLIPPAGE_PCT", "0.10") or "0.10") / 100.0
COMMISSION = float(os.getenv("DAILY_SIM_COMMISSION", "0") or "0")
SLEEP = float(os.getenv("DAILY_SIM_FETCH_SLEEP_SECONDS", "0.4") or "0.4")
MAX_CLOSED_ROWS = int(os.getenv("DAILY_SIM_MAX_CLOSED_ROWS", "250") or "250")
MIN_TRADE_AMOUNT = float(os.getenv("DAILY_SIM_MIN_TRADE_AMOUNT", "25") or "25")


# -----------------------------
# Utilities
# -----------------------------

def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log("WARN", f"read_json failed: {path}: {exc}")
        return None


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


def avg(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if to_float(v) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def win_rate(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if to_float(v) is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v > 0) / len(vals) * 100.0


def is_date_dir(name: str) -> bool:
    if len(name) != 10:
        return False
    try:
        pd.Timestamp(name)
        return name[4] == "-" and name[7] == "-"
    except Exception:
        return False


def date_range_from_daily_v2() -> Tuple[str, str]:
    dates = (
        sorted(
            d.name
            for d in DAILY_V2_DIR.iterdir()
            if d.is_dir() and is_date_dir(d.name) and (d / "top10.json").exists()
        )
        if DAILY_V2_DIR.exists()
        else []
    )
    if not dates:
        raise SystemExit("No daily-v2 top10.json files found.")
    return dates[0], dates[-1]


def selected_daily_files() -> List[Path]:
    default_start, default_end = date_range_from_daily_v2()
    start = START_DATE or default_start
    end = END_DATE or default_end

    out: List[Path] = []
    for d in sorted(DAILY_V2_DIR.iterdir() if DAILY_V2_DIR.exists() else []):
        if not d.is_dir() or not is_date_dir(d.name):
            continue
        if d.name < start or d.name > end:
            continue
        p = d / "top10.json"
        if p.exists():
            out.append(p)

    if not out:
        raise SystemExit(f"No daily-v2 top10.json files selected. start={start}, end={end}")
    return out


# -----------------------------
# Price data
# -----------------------------

def extract_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return either (Price, Ticker) or (Ticker, Price).
        try:
            if symbol in df.columns.get_level_values(0):
                df = df[symbol]
            elif symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1)
            elif df.columns.nlevels > 1:
                # Single ticker with redundant second level.
                df = df.droplevel(1, axis=1)
        except Exception:
            pass

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    need = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in need):
        log("WARN", f"missing OHLC columns for {symbol}: columns={list(df.columns)}")
        return pd.DataFrame()

    cols = need + (["Volume"] if "Volume" in df.columns else [])
    out = df[cols].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=need)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["SMA20"] = out["Close"].rolling(20, min_periods=5).mean()
    return out


def fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Add yfinance to requirements.txt.")
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return extract_ohlcv(raw, symbol)
    except Exception as exc:
        log("WARN", f"history fetch failed: {symbol}: {exc}")
        return pd.DataFrame()


def fetch_histories(symbols: List[str], first_date: str, last_date: str) -> Dict[str, pd.DataFrame]:
    start = (pd.Timestamp(first_date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(last_date) + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    out: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        log("INFO", f"fetch daily simulation history {i}/{len(symbols)} {sym}")
        out[sym] = fetch_history(sym, start, end)
        if SLEEP > 0 and i < len(symbols):
            time.sleep(SLEEP)
    return out


def next_pos_after(df: pd.DataFrame, date_str: str) -> Optional[int]:
    if df.empty:
        return None
    dt = pd.Timestamp(date_str).normalize()
    for i, ts in enumerate(df.index):
        if pd.Timestamp(ts).normalize() > dt:
            return i
    return None


def pos_on_or_before(df: pd.DataFrame, date_str: str) -> Optional[int]:
    if df.empty:
        return None
    dt = pd.Timestamp(date_str).normalize()
    idx = [i for i, ts in enumerate(df.index) if pd.Timestamp(ts).normalize() <= dt]
    return idx[-1] if idx else None


def pos_on_or_after(df: pd.DataFrame, date_str: str) -> Optional[int]:
    if df.empty:
        return None
    dt = pd.Timestamp(date_str).normalize()
    for i, ts in enumerate(df.index):
        if pd.Timestamp(ts).normalize() >= dt:
            return i
    return None


def price_on_or_before(hist: pd.DataFrame, date_str: str, field: str = "Close") -> Optional[float]:
    pos = pos_on_or_before(hist, date_str)
    if pos is None:
        return None
    return to_float(hist.iloc[pos].get(field))


def price_on_or_after(hist: pd.DataFrame, date_str: str, field: str = "Open") -> Optional[float]:
    pos = pos_on_or_after(hist, date_str)
    if pos is None:
        return None
    return to_float(hist.iloc[pos].get(field))


def calendar_between(histories: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> List[str]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = set()
    for df in histories.values():
        if df is None or df.empty:
            continue
        for ts in df.index:
            t = pd.Timestamp(ts).normalize()
            if start <= t <= end:
                dates.add(t.date().isoformat())
    if not dates:
        return []
    return sorted(dates)


# -----------------------------
# Signal normalization
# -----------------------------

def clean_triage(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s == "trade":
        return "Trade"
    if s == "watch":
        return "Watch"
    if s == "ignore":
        return "Ignore"
    return "Watch"


def components(item: Dict[str, Any]) -> Dict[str, Any]:
    v = item.get("v2_components")
    if isinstance(v, dict):
        return v
    v = item.get("score_components")
    return v if isinstance(v, dict) else {}


def event_is_news_heavy(item: Dict[str, Any]) -> bool:
    arch = str(item.get("archetype") or "").lower()
    comps = components(item)
    catalyst = to_float(comps.get("catalyst_confirmation")) or 0.0
    setup = to_float(comps.get("breakout_setup_quality")) or 0.0
    comp = to_float(comps.get("compression_release")) or 0.0
    return ("catalyst" in arch or "news" in arch) and catalyst >= 0.55 and max(setup, comp) < 0.65


def event_is_volume_compression(item: Dict[str, Any]) -> bool:
    arch = str(item.get("archetype") or "").lower()
    comps = components(item)
    volume = to_float(comps.get("volume_liquidity_shock")) or 0.0
    comp = to_float(comps.get("compression_release")) or 0.0
    return ("compression" in arch and volume >= 0.55) or (volume >= 0.65 and comp >= 0.60)


def theme_of(item: Dict[str, Any]) -> str:
    sym = str(item.get("symbol") or "").upper()
    themes = {
        "RKLB": "Space", "VSAT": "Space", "LUNR": "Space", "ASTS": "Space", "IRDM": "Space",
        "TER": "Robotics", "ISRG": "Robotics", "SYM": "Robotics", "ROK": "Robotics", "CGNX": "Robotics", "PRCT": "Robotics", "SERV": "Robotics", "PDYN": "Robotics",
        "NVDA": "AI", "MSFT": "AI", "PLTR": "AI", "AI": "AI", "SOUN": "AI", "U": "AI", "HIMS": "AI", "TEM": "AI", "RXRX": "AI", "ABCL": "AI",
        "ALAB": "AI Infrastructure", "CRDO": "AI Infrastructure", "VRT": "AI Infrastructure", "NBIS": "AI Infrastructure", "RBRK": "AI Infrastructure", "ESTC": "AI Infrastructure", "MNDY": "AI Infrastructure", "S": "AI Infrastructure",
        "OII": "Energy", "FTI": "Energy", "LNG": "Energy", "PR": "Energy", "MTDR": "Energy", "OXY": "Energy",
    }
    return themes.get(sym, "Other")


def load_signals() -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for path in selected_daily_files():
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        date = str(payload.get("date") or path.parent.name)
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol") or "").upper().strip()
            if not sym:
                continue
            sig = dict(item)
            sig["signal_date"] = date
            sig["symbol"] = sym
            sig["triage"] = clean_triage(sig.get("triage"))
            sig["theme"] = theme_of(sig)
            sig["is_news_heavy"] = event_is_news_heavy(sig)
            sig["is_volume_compression"] = event_is_volume_compression(sig)
            signals.append(sig)

    signals.sort(key=lambda x: (x.get("signal_date", ""), int(x.get("rank") or 999), x.get("symbol", "")))
    return signals


# -----------------------------
# Strategy and exit simulation
# -----------------------------

@dataclass
class Strategy:
    name: str
    note: str
    triage: Tuple[str, ...] = ("Trade",)
    fixed_amount: Optional[float] = None
    allocation_pct: float = 0.10
    compound: bool = True
    stop_pct: Optional[float] = 8.0
    profit_trigger_pct: Optional[float] = 15.0
    profit_lock_pct: Optional[float] = 7.0
    max_hold_days: int = 10
    trend_fail: bool = True
    max_symbol_pct: float = 25.0
    max_theme_pct: float = 40.0
    exclude_news_heavy: bool = False
    volume_compression_only: bool = False


def strategies() -> List[Strategy]:
    return [
        Strategy("Default Daily Compound", "Trade only · 10% cash allocation · stop -8% · profit lock 15/7 · max hold 10D"),
        Strategy("Fixed $1,000", "$1,000 per signal · no compounding · Trade only", fixed_amount=1000.0, compound=False),
        Strategy("Trade + Watch", "Trade and Watch signals · compound sizing", triage=("Trade", "Watch")),
        Strategy("No News-heavy", "Trade only · exclude catalyst-heavy setups without price structure", exclude_news_heavy=True),
        Strategy("Volume + Compression only", "Trade/Watch only when volume and compression confirm", triage=("Trade", "Watch"), volume_compression_only=True),
        Strategy("Profit Lock 10/5", "Trade only · lock +5% after +10% MFE", profit_trigger_pct=10.0, profit_lock_pct=5.0),
        Strategy("Hold 10D", "Trade only · no stop/profit lock · fixed 10D hold", stop_pct=None, profit_trigger_pct=None, profit_lock_pct=None, trend_fail=False, max_hold_days=10),
        Strategy("Position Cap 25%", "Default exit rules · explicit 25% symbol cap", max_symbol_pct=25.0),
    ]


def signal_allowed(item: Dict[str, Any], strategy: Strategy) -> bool:
    triage = clean_triage(item.get("triage"))
    if triage not in strategy.triage:
        return False
    if strategy.exclude_news_heavy and item.get("is_news_heavy"):
        return False
    if strategy.volume_compression_only and not item.get("is_volume_compression"):
        return False
    return True


def simulate_exit(symbol: str, item: Dict[str, Any], hist: pd.DataFrame, strategy: Strategy) -> Optional[Dict[str, Any]]:
    entry_pos = next_pos_after(hist, str(item.get("signal_date")))
    if entry_pos is None or entry_pos >= len(hist):
        return None

    entry_row = hist.iloc[entry_pos]
    entry_date = hist.index[entry_pos].date().isoformat()
    raw_entry = to_float(entry_row.get("Open"))
    if raw_entry is None or raw_entry <= 0:
        return None

    entry_price = raw_entry * (1.0 + BUY_SLIPPAGE)
    stop_price = entry_price * (1.0 - (strategy.stop_pct or 0.0) / 100.0) if strategy.stop_pct is not None else None

    lock_active = False
    lock_price: Optional[float] = None
    current_pos = len(hist) - 1
    planned_max_pos = entry_pos + max(1, strategy.max_hold_days)
    max_pos = min(current_pos, planned_max_pos)

    exit_pos: Optional[int] = None
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None

    high_max = entry_price
    low_min = entry_price
    peak_day = 0

    for pos in range(entry_pos, max_pos + 1):
        row = hist.iloc[pos]
        days = pos - entry_pos
        op = to_float(row.get("Open")) or to_float(row.get("Close")) or entry_price
        hi = to_float(row.get("High")) or op
        lo = to_float(row.get("Low")) or op
        cl = to_float(row.get("Close")) or op
        sma20 = to_float(row.get("SMA20"))

        if hi > high_max:
            high_max = hi
            peak_day = days
        if lo < low_min:
            low_min = lo

        # If both stop and profit trigger happen intraday, we choose the conservative stop outcome first.
        if stop_price is not None and lo <= stop_price:
            exit_pos = pos
            exit_reason = "stop_loss"
            base_exit = op if op < stop_price else stop_price
            exit_price = base_exit * (1.0 - SELL_SLIPPAGE)
            break

        if strategy.profit_trigger_pct is not None and not lock_active:
            trigger = entry_price * (1.0 + strategy.profit_trigger_pct / 100.0)
            if hi >= trigger:
                lock_active = True
                lock_price = entry_price * (1.0 + (strategy.profit_lock_pct or 0.0) / 100.0)

        if lock_active and lock_price is not None and lo <= lock_price:
            exit_pos = pos
            exit_reason = "profit_lock"
            base_exit = op if op < lock_price else lock_price
            exit_price = base_exit * (1.0 - SELL_SLIPPAGE)
            break

        if strategy.trend_fail and days >= 2 and sma20 is not None and cl < sma20:
            exit_pos = pos
            exit_reason = "trend_fail"
            exit_price = cl * (1.0 - SELL_SLIPPAGE)
            break

    if exit_pos is None:
        if planned_max_pos <= current_pos:
            exit_pos = max_pos
            exit_reason = "max_hold"
            exit_price = (to_float(hist.iloc[exit_pos].get("Close")) or entry_price) * (1.0 - SELL_SLIPPAGE)
        else:
            # Not enough future bars to complete the planned hold. Keep the position open.
            exit_pos = None
            exit_reason = "open"
            exit_price = None

    current_price = to_float(hist.iloc[current_pos].get("Close")) or entry_price
    current_date = hist.index[current_pos].date().isoformat()

    if exit_pos is not None:
        exit_date = hist.index[exit_pos].date().isoformat()
        hold_days = max(0, exit_pos - entry_pos)
        realized_exit_price = exit_price if exit_price is not None else current_price * (1.0 - SELL_SLIPPAGE)
    else:
        exit_date = None
        hold_days = max(0, current_pos - entry_pos)
        realized_exit_price = None

    return {
        "symbol": symbol,
        "signal_date": item.get("signal_date"),
        "entry_date": entry_date,
        "exit_date": exit_date,
        "status": "open" if exit_date is None else "closed",
        "entry_price": entry_price,
        "exit_price": realized_exit_price,
        "current_date": current_date,
        "current_price": current_price,
        "return_pct": pct(realized_exit_price, entry_price) if realized_exit_price is not None else pct(current_price, entry_price),
        "unrealized_return_pct": pct(current_price, entry_price),
        "exit_reason": exit_reason,
        "holding_days": hold_days,
        "mfe_pct": pct(high_max, entry_price),
        "mae_pct": pct(low_min, entry_price),
        "peak_day": peak_day,
        "triage": clean_triage(item.get("triage")),
        "archetype": item.get("archetype") or "Unclassified",
        "theme": item.get("theme") or theme_of(item),
        "rank": item.get("rank"),
        "score_pts": item.get("score_pts"),
        "reason": item.get("reason"),
        "is_news_heavy": bool(item.get("is_news_heavy")),
        "is_volume_compression": bool(item.get("is_volume_compression")),
        "raw_item": item,
    }


# -----------------------------
# Portfolio simulation
# -----------------------------

def mark_position_value(pos: Dict[str, Any], histories: Dict[str, pd.DataFrame], date: str) -> float:
    price = price_on_or_before(histories.get(pos["symbol"], pd.DataFrame()), date, "Close") or to_float(pos.get("entry_price")) or 0.0
    return (to_float(pos.get("shares")) or 0.0) * price


def mark_benchmark_value(lot: Dict[str, Any], bench: pd.DataFrame, date: str) -> float:
    price = price_on_or_before(bench, date, "Close") or to_float(lot.get("entry_price")) or 0.0
    return (to_float(lot.get("shares")) or 0.0) * price


def max_drawdown_from_curve(curve: List[Dict[str, Any]], key: str = "portfolio_equity") -> float:
    peak = None
    max_dd = 0.0
    for row in curve:
        v = to_float(row.get(key))
        if v is None:
            continue
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            dd = (v / peak - 1.0) * 100.0
            max_dd = min(max_dd, dd)
    return max_dd


def summarize_buckets(trades: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        buckets[str(t.get(key) or "Unknown")].append(t)

    out: List[Dict[str, Any]] = []
    for label, rows in buckets.items():
        pnl = sum((to_float(r.get("pl_value")) or 0.0) for r in rows)
        cost = sum((to_float(r.get("cost")) or 0.0) for r in rows)
        out.append({
            "label": label,
            "count": len(rows),
            "total_pl": round(pnl, 2),
            "total_cost": round(cost, 2),
            "avg_return_pct": safe_round(avg([r.get("return_pct") for r in rows]), 2),
            "win_rate": safe_round(win_rate([r.get("return_pct") for r in rows]), 2),
        })
    out.sort(key=lambda x: abs(to_float(x.get("total_pl")) or 0.0), reverse=True)
    return out


def empty_result(strategy: Strategy) -> Dict[str, Any]:
    return {
        "strategy": asdict(strategy),
        "summary": {
            "strategy_name": strategy.name,
            "note": strategy.note,
            "initial_capital": round(INITIAL_CAPITAL, 2),
            "portfolio_equity": round(INITIAL_CAPITAL, 2),
            "cash": round(INITIAL_CAPITAL, 2),
            "market_value": 0.0,
            "external_capital": round(INITIAL_CAPITAL, 2),
            "net_pl_value": 0.0,
            "net_return_pct": 0.0,
            "benchmark_symbol": BENCHMARK_1,
            "benchmark_equity": round(INITIAL_CAPITAL, 2),
            "benchmark_return_pct": 0.0,
            "secondary_benchmark_symbol": BENCHMARK_2,
            "secondary_benchmark_equity": round(INITIAL_CAPITAL, 2),
            "secondary_benchmark_return_pct": 0.0,
            "alpha_pct": 0.0,
            "alpha_value": 0.0,
            "max_drawdown_pct": 0.0,
            "return_drawdown_ratio": None,
            "closed_trades": 0,
            "open_positions": 0,
            "skipped_signals": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "avg_holding_days": None,
            "avg_mfe_pct": None,
            "avg_mae_pct": None,
            "mfe_mae_ratio": None,
            "profit_lock_hit_rate": None,
            "stop_hit_rate": None,
            "exposure_pct": 0.0,
        },
        "equity_curve": [],
        "closed_trades": [],
        "open_positions": [],
        "skipped": [],
        "exit_reason_summary": [],
        "triage_summary": [],
        "archetype_summary": [],
        "theme_summary": [],
    }


def run_strategy(strategy: Strategy, signals: List[Dict[str, Any]], histories: Dict[str, pd.DataFrame], bench1: pd.DataFrame, bench2: pd.DataFrame) -> Dict[str, Any]:
    proposed: List[Dict[str, Any]] = []
    for item in signals:
        if not signal_allowed(item, strategy):
            continue
        sym = item["symbol"]
        hist = histories.get(sym, pd.DataFrame())
        if hist.empty:
            continue
        ex = simulate_exit(sym, item, hist, strategy)
        if ex is not None:
            proposed.append(ex)

    proposed.sort(key=lambda x: (x["entry_date"], x.get("rank") or 999, x["symbol"]))
    if not proposed:
        return empty_result(strategy)

    first_entry = min(p["entry_date"] for p in proposed)
    latest_hist_dates = [df.index[-1].date().isoformat() for df in histories.values() if df is not None and not df.empty]
    last_date = max(latest_hist_dates + [max(p.get("exit_date") or p.get("current_date") or p["entry_date"] for p in proposed)])
    cal = calendar_between(histories, first_entry, last_date)
    if not cal:
        cal = sorted(set([p["entry_date"] for p in proposed] + [p.get("exit_date") for p in proposed if p.get("exit_date")]))

    by_entry: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in proposed:
        by_entry[p["entry_date"]].append(p)

    cash = INITIAL_CAPITAL
    external_capital = INITIAL_CAPITAL
    open_positions: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    bench1_cash = INITIAL_CAPITAL
    bench2_cash = INITIAL_CAPITAL
    bench1_lots: List[Dict[str, Any]] = []
    bench2_lots: List[Dict[str, Any]] = []

    curve: List[Dict[str, Any]] = []

    def mark_portfolio(date: str) -> Tuple[float, float]:
        mv = sum(mark_position_value(pos, histories, date) for pos in open_positions)
        return cash + mv, mv

    def mark_benchmark(date: str, cash_value: float, lots: List[Dict[str, Any]], bench: pd.DataFrame) -> Tuple[float, float]:
        mv = sum(mark_benchmark_value(lot, bench, date) for lot in lots)
        return cash_value + mv, mv

    def exposures(date: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        sym_exp: Dict[str, float] = defaultdict(float)
        theme_exp: Dict[str, float] = defaultdict(float)
        for pos in open_positions:
            val = mark_position_value(pos, histories, date)
            sym_exp[pos["symbol"]] += val
            theme_exp[pos.get("theme", "Other")] += val
        return sym_exp, theme_exp

    for date in cal:
        # 1) Close portfolio positions whose rule exit date has arrived.
        remaining_positions: List[Dict[str, Any]] = []
        for pos in open_positions:
            exit_date = pos.get("exit_date")
            if exit_date and exit_date <= date:
                exit_price = to_float(pos.get("exit_price")) or price_on_or_before(histories.get(pos["symbol"], pd.DataFrame()), date, "Close") or to_float(pos.get("entry_price")) or 0.0
                proceeds = (to_float(pos.get("shares")) or 0.0) * exit_price - COMMISSION
                cash += proceeds
                closed_pos = dict(pos)
                closed_pos["exit_value"] = round(proceeds, 2)
                closed_pos["pl_value"] = round(proceeds - (to_float(pos.get("cost")) or 0.0), 2)
                closed_pos["return_pct"] = safe_round(pct(exit_price, pos.get("entry_price")), 2)
                closed.append(closed_pos)
            else:
                remaining_positions.append(pos)
        open_positions = remaining_positions

        # 2) Close benchmark lots on the same exit dates as their paired portfolio trades.
        def close_benchmark_lots(cash_value: float, lots: List[Dict[str, Any]], bench: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
            remaining: List[Dict[str, Any]] = []
            for lot in lots:
                exit_date = lot.get("exit_date")
                if exit_date and exit_date <= date:
                    px = price_on_or_before(bench, exit_date, "Close") or price_on_or_before(bench, date, "Close") or to_float(lot.get("entry_price")) or 0.0
                    cash_value += (to_float(lot.get("shares")) or 0.0) * px * (1.0 - SELL_SLIPPAGE)
                else:
                    remaining.append(lot)
            return cash_value, remaining

        bench1_cash, bench1_lots = close_benchmark_lots(bench1_cash, bench1_lots, bench1)
        bench2_cash, bench2_lots = close_benchmark_lots(bench2_cash, bench2_lots, bench2)

        # 3) Open new positions for today's entries.
        for p in by_entry.get(date, []):
            equity, _ = mark_portfolio(date)
            if equity <= 0 or cash <= 0:
                skipped.append({**p, "skip_reason": "no_cash_or_equity"})
                continue

            if strategy.fixed_amount is not None:
                desired = strategy.fixed_amount
            else:
                desired = cash * strategy.allocation_pct
            desired = max(0.0, desired)

            sym_exp, theme_exp = exposures(date)
            sym_cap = equity * strategy.max_symbol_pct / 100.0
            theme_cap = equity * strategy.max_theme_pct / 100.0
            sym_room = max(0.0, sym_cap - sym_exp.get(p["symbol"], 0.0))
            theme_room = max(0.0, theme_cap - theme_exp.get(p.get("theme", "Other"), 0.0))
            amount = min(desired, cash, sym_room, theme_room)

            if amount < MIN_TRADE_AMOUNT:
                skipped.append({**p, "skip_reason": "cash_or_cap"})
                continue

            entry_price = to_float(p.get("entry_price")) or 0.0
            if entry_price <= 0:
                skipped.append({**p, "skip_reason": "invalid_entry_price"})
                continue

            shares = max(0.0, (amount - COMMISSION) / entry_price)
            if shares <= 0:
                skipped.append({**p, "skip_reason": "zero_shares"})
                continue

            cost = shares * entry_price + COMMISSION
            cash -= cost

            pos = dict(p)
            pos["shares"] = shares
            pos["cost"] = round(cost, 2)
            pos["buy_type"] = "compound" if strategy.compound else "fixed"
            open_positions.append(pos)

            # Benchmark lots: invest up to the benchmark cash available at the same accepted-entry date.
            # This avoids hidden leverage while keeping timing tied to actual portfolio trades.
            b1_open = price_on_or_after(bench1, date, "Open")
            if b1_open and b1_open > 0 and bench1_cash >= MIN_TRADE_AMOUNT:
                b_amount = min(cost, bench1_cash)
                b_entry = b1_open * (1.0 + BUY_SLIPPAGE)
                bench1_cash -= b_amount
                bench1_lots.append({
                    "entry_date": date,
                    "exit_date": p.get("exit_date"),
                    "entry_price": b_entry,
                    "shares": b_amount / b_entry,
                    "cost": b_amount,
                    "source_symbol": p["symbol"],
                })

            b2_open = price_on_or_after(bench2, date, "Open")
            if b2_open and b2_open > 0 and bench2_cash >= MIN_TRADE_AMOUNT:
                b_amount = min(cost, bench2_cash)
                b_entry = b2_open * (1.0 + BUY_SLIPPAGE)
                bench2_cash -= b_amount
                bench2_lots.append({
                    "entry_date": date,
                    "exit_date": p.get("exit_date"),
                    "entry_price": b_entry,
                    "shares": b_amount / b_entry,
                    "cost": b_amount,
                    "source_symbol": p["symbol"],
                })

        equity, market_value = mark_portfolio(date)
        b1_equity, b1_mv = mark_benchmark(date, bench1_cash, bench1_lots, bench1)
        b2_equity, b2_mv = mark_benchmark(date, bench2_cash, bench2_lots, bench2)
        curve.append({
            "date": date,
            "portfolio_equity": round(equity, 2),
            "cash": round(cash, 2),
            "market_value": round(market_value, 2),
            "external_capital": round(external_capital, 2),
            "benchmark_symbol": BENCHMARK_1,
            "benchmark_equity": round(b1_equity, 2),
            "benchmark_cash": round(bench1_cash, 2),
            "benchmark_market_value": round(b1_mv, 2),
            "secondary_benchmark_symbol": BENCHMARK_2,
            "secondary_benchmark_equity": round(b2_equity, 2),
            "secondary_benchmark_cash": round(bench2_cash, 2),
            "secondary_benchmark_market_value": round(b2_mv, 2),
            "drawdown_pct": None,
            "exposure_pct": safe_round((market_value / equity * 100.0) if equity else None, 2),
        })

    if not curve:
        return empty_result(strategy)

    peak = None
    for row in curve:
        v = to_float(row.get("portfolio_equity"))
        if v is None:
            continue
        if peak is None or v > peak:
            peak = v
        row["drawdown_pct"] = safe_round((v / peak - 1.0) * 100.0 if peak else 0.0, 2)

    final = curve[-1]
    final_date = str(final.get("date"))
    final_equity = to_float(final.get("portfolio_equity")) or INITIAL_CAPITAL
    b1_equity = to_float(final.get("benchmark_equity")) or INITIAL_CAPITAL
    b2_equity = to_float(final.get("secondary_benchmark_equity")) or INITIAL_CAPITAL
    net_return = pct(final_equity, external_capital) or 0.0
    b1_return = pct(b1_equity, external_capital) or 0.0
    b2_return = pct(b2_equity, external_capital) or 0.0
    max_dd = max_drawdown_from_curve(curve)
    ret_dd = net_return / abs(max_dd) if max_dd else None

    open_out: List[Dict[str, Any]] = []
    for pos in open_positions:
        cur_price = price_on_or_before(histories.get(pos["symbol"], pd.DataFrame()), final_date, "Close") or to_float(pos.get("entry_price")) or 0.0
        mv = (to_float(pos.get("shares")) or 0.0) * cur_price
        out = dict(pos)
        out["current_price"] = round(cur_price, 2)
        out["market_value"] = round(mv, 2)
        out["unrealized_pl"] = round(mv - (to_float(out.get("cost")) or 0.0), 2)
        out["unrealized_return_pct"] = safe_round(pct(cur_price, pos.get("entry_price")), 2)
        open_out.append(out)

    closed.sort(key=lambda x: (x.get("exit_date") or "", abs(to_float(x.get("pl_value")) or 0.0)), reverse=True)
    all_trades = closed + open_out

    avg_mfe = avg([t.get("mfe_pct") for t in closed])
    avg_mae = avg([t.get("mae_pct") for t in closed])
    mfe_mae = abs(avg_mfe) / abs(avg_mae) if avg_mfe is not None and avg_mae not in (None, 0) else None

    summary = {
        "strategy_name": strategy.name,
        "note": strategy.note,
        "initial_capital": round(INITIAL_CAPITAL, 2),
        "portfolio_equity": round(final_equity, 2),
        "cash": round(to_float(final.get("cash")) or 0.0, 2),
        "market_value": round(to_float(final.get("market_value")) or 0.0, 2),
        "external_capital": round(external_capital, 2),
        "net_pl_value": round(final_equity - external_capital, 2),
        "net_return_pct": safe_round(net_return, 2),
        "benchmark_symbol": BENCHMARK_1,
        "benchmark_equity": round(b1_equity, 2),
        "benchmark_return_pct": safe_round(b1_return, 2),
        "secondary_benchmark_symbol": BENCHMARK_2,
        "secondary_benchmark_equity": round(b2_equity, 2),
        "secondary_benchmark_return_pct": safe_round(b2_return, 2),
        "alpha_pct": safe_round(net_return - b1_return, 2),
        "alpha_value": round(final_equity - b1_equity, 2),
        "secondary_alpha_pct": safe_round(net_return - b2_return, 2),
        "secondary_alpha_value": round(final_equity - b2_equity, 2),
        "max_drawdown_pct": safe_round(max_dd, 2),
        "return_drawdown_ratio": safe_round(ret_dd, 2),
        "closed_trades": len(closed),
        "open_positions": len(open_out),
        "skipped_signals": len(skipped),
        "win_rate": safe_round(win_rate([t.get("return_pct") for t in closed]), 2),
        "avg_return_pct": safe_round(avg([t.get("return_pct") for t in closed]), 2),
        "avg_holding_days": safe_round(avg([t.get("holding_days") for t in closed]), 2),
        "avg_mfe_pct": safe_round(avg_mfe, 2),
        "avg_mae_pct": safe_round(avg_mae, 2),
        "mfe_mae_ratio": safe_round(mfe_mae, 2),
        "profit_lock_hit_rate": safe_round(sum(1 for t in closed if t.get("exit_reason") == "profit_lock") / len(closed) * 100.0, 2) if closed else None,
        "stop_hit_rate": safe_round(sum(1 for t in closed if t.get("exit_reason") == "stop_loss") / len(closed) * 100.0, 2) if closed else None,
        "exposure_pct": final.get("exposure_pct"),
    }

    return {
        "strategy": asdict(strategy),
        "summary": summary,
        "equity_curve": curve,
        "closed_trades": closed[:MAX_CLOSED_ROWS],
        "open_positions": open_out,
        "skipped": skipped[:100],
        "exit_reason_summary": summarize_buckets(closed, "exit_reason"),
        "triage_summary": summarize_buckets(all_trades, "triage"),
        "archetype_summary": summarize_buckets(all_trades, "archetype"),
        "theme_summary": summarize_buckets(all_trades, "theme"),
    }


# -----------------------------
# Payload
# -----------------------------

def build_strategy_comparison(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        s = r.get("summary", {}) if isinstance(r, dict) else {}
        rows.append({
            "name": s.get("strategy_name"),
            "note": s.get("note"),
            "net_return_pct": s.get("net_return_pct"),
            "benchmark_return_pct": s.get("benchmark_return_pct"),
            "alpha_pct": s.get("alpha_pct"),
            "alpha_value": s.get("alpha_value"),
            "max_drawdown_pct": s.get("max_drawdown_pct"),
            "return_drawdown_ratio": s.get("return_drawdown_ratio"),
            "win_rate": s.get("win_rate"),
            "closed_trades": s.get("closed_trades"),
            "open_positions": s.get("open_positions"),
            "skipped_signals": s.get("skipped_signals"),
            "exposure_pct": s.get("exposure_pct"),
        })
    return rows


def highlights(default: Dict[str, Any], comparison: List[Dict[str, Any]]) -> Dict[str, Any]:
    closed = default.get("closed_trades", []) if isinstance(default, dict) else []
    valid_ret = [x for x in comparison if to_float(x.get("net_return_pct")) is not None]
    valid_dd = [x for x in comparison if to_float(x.get("max_drawdown_pct")) is not None]
    best_return = max(valid_ret, key=lambda x: to_float(x.get("net_return_pct")) or -1e9) if valid_ret else None
    lowest_dd = max(valid_dd, key=lambda x: to_float(x.get("max_drawdown_pct")) or -1e9) if valid_dd else None
    valid_risk = [x for x in comparison if to_float(x.get("return_drawdown_ratio")) is not None]
    best_risk = max(valid_risk, key=lambda x: to_float(x.get("return_drawdown_ratio")) or -1e9) if valid_risk else None
    valid_trades = [x for x in closed if to_float(x.get("return_pct")) is not None]
    best_trade = max(valid_trades, key=lambda x: to_float(x.get("return_pct")) or -1e9) if valid_trades else None
    worst_trade = min(valid_trades, key=lambda x: to_float(x.get("return_pct")) or 1e9) if valid_trades else None

    fixed = next((x for x in comparison if x.get("name") == "Fixed $1,000"), None)
    dsum = default.get("summary", {}) if isinstance(default, dict) else {}
    compounding_impact = None
    if fixed and to_float(fixed.get("net_return_pct")) is not None and to_float(dsum.get("net_return_pct")) is not None:
        fixed_equity = INITIAL_CAPITAL * (1.0 + (to_float(fixed.get("net_return_pct")) or 0.0) / 100.0)
        compounding_impact = {
            "return_delta_pct": safe_round((to_float(dsum.get("net_return_pct")) or 0.0) - (to_float(fixed.get("net_return_pct")) or 0.0), 2),
            "value_delta": safe_round((to_float(dsum.get("portfolio_equity")) or 0.0) - fixed_equity, 2),
        }

    return {
        "best_return_strategy": best_return,
        "lowest_drawdown_strategy": lowest_dd,
        "best_risk_adjusted_strategy": best_risk,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "compounding_impact": compounding_impact,
    }


def main() -> None:
    files = selected_daily_files()
    start = files[0].parent.name
    end = files[-1].parent.name
    signals = load_signals()

    symbols = sorted(set([s["symbol"] for s in signals] + [BENCHMARK_1, BENCHMARK_2]))
    histories = fetch_histories(symbols, start, end)
    bench1 = histories.get(BENCHMARK_1, pd.DataFrame())
    bench2 = histories.get(BENCHMARK_2, pd.DataFrame())

    if bench1.empty:
        log("WARN", f"Primary benchmark history is empty: {BENCHMARK_1}")
    if bench2.empty:
        log("WARN", f"Secondary benchmark history is empty: {BENCHMARK_2}")

    results = []
    for st in strategies():
        log("INFO", f"simulate strategy: {st.name}")
        results.append(run_strategy(st, signals, histories, bench1, bench2))

    default = results[0] if results else empty_result(strategies()[0])
    comparison = build_strategy_comparison(results)

    payload = {
        "version": "daily_simulation_v2_hardened",
        "date": end,
        "range": {"start": start, "end": end, "daily_files": len(files), "signals": len(signals)},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "policy": {
            "initial_capital": INITIAL_CAPITAL,
            "primary_model": "Initial-capital compounding portfolio",
            "entry": "Eligible Daily signals enter at the next trading day's open.",
            "default_exit": "Stop -8%, profit lock 15/7, trend fail below SMA20, or max hold 10 trading days.",
            "benchmark": f"Benchmark lots enter on accepted portfolio entries and exit on the same portfolio exit dates, without hidden leverage.",
            "costs": {
                "buy_slippage_pct": BUY_SLIPPAGE * 100.0,
                "sell_slippage_pct": SELL_SLIPPAGE * 100.0,
                "commission": COMMISSION,
            },
        },
        "summary": default.get("summary", {}),
        "equity_curve": default.get("equity_curve", []),
        "closed_trades": default.get("closed_trades", []),
        "open_positions": default.get("open_positions", []),
        "exit_reason_summary": default.get("exit_reason_summary", []),
        "triage_summary": default.get("triage_summary", []),
        "archetype_summary": default.get("archetype_summary", []),
        "theme_summary": default.get("theme_summary", []),
        "strategy_comparison": comparison,
        "strategy_results": results,
        "highlights": highlights(default, comparison),
    }

    ensure_dir(SIM_DIR)
    out = SIM_DIR / f"{end}.json"
    write_json(out, payload)
    write_json(SIM_DIR / "latest.json", payload)
    log("INFO", f"Wrote {out}")
    log("INFO", f"Wrote {SIM_DIR / 'latest.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR", f"FATAL in rebuild_daily_simulation: {exc}")
        sys.exit(1)
