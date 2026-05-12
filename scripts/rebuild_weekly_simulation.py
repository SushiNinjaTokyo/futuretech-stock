#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
BACKTEST_WEEKS = int(os.getenv("WEEKLY_BACKTEST_WEEKS", "24"))
BACKTEST_END_DATE = os.getenv("WEEKLY_BACKTEST_END_DATE", "").strip()
BENCHMARK = os.getenv("WEEKLY_SIM_BENCHMARK", os.getenv("WEEKLY_BACKTEST_BENCHMARK", "SPY")).strip().upper() or "SPY"
MIN_SIGNAL_LEVEL = os.getenv("WEEKLY_SIM_MIN_SIGNAL", "B").strip().upper() or "B"

BUY_SLIPPAGE = float(os.getenv("WEEKLY_SIM_BUY_SLIPPAGE_PCT", "0.15")) / 100.0
SELL_SLIPPAGE = float(os.getenv("WEEKLY_SIM_SELL_SLIPPAGE_PCT", "0.15")) / 100.0
COMMISSION = float(os.getenv("WEEKLY_SIM_COMMISSION", "0"))
BASE_SHARES = float(os.getenv("WEEKLY_SIM_BASE_SHARES", "1"))
DEFAULT_STOP_PCT = float(os.getenv("WEEKLY_SIM_STOP_PCT", "5")) / 100.0
DEFAULT_SCORE_EXIT = int(os.getenv("WEEKLY_SIM_SCORE_EXIT", "700"))
DEFAULT_TIME_EXIT_DAYS = int(os.getenv("WEEKLY_SIM_TIME_EXIT_DAYS", "60"))
LIQUIDITY_WARN_FRACTION = float(os.getenv("WEEKLY_SIM_LIQUIDITY_WARN_FRACTION", "0.001"))
MAX_TRADE_LOG_ROWS = int(os.getenv("WEEKLY_SIM_MAX_TRADE_LOG_ROWS", "300"))

QUALIFIED_SIGNALS = {
    "A": {"A+ Fresh Breakout", "A Leader"},
    "B": {"A+ Fresh Breakout", "A Leader", "B Constructive Setup"},
    "C": {"A+ Fresh Breakout", "A Leader", "B Constructive Setup", "C Early Watch"},
}


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


def ratio(n: Any, d: Any) -> Optional[float]:
    nv = to_float(n)
    dv = to_float(d)
    if nv is None or dv is None or dv == 0:
        return None
    return nv / dv


def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def win_rate(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(1 for v in vals if v > 0) / len(vals)


def sum_float(values: Iterable[Any]) -> float:
    return sum((to_float(v) or 0.0) for v in values)


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


def next_trading_pos_after(df: pd.DataFrame, date_str: str) -> Optional[int]:
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
    eligible = [i for i, ts in enumerate(df.index) if pd.Timestamp(ts).normalize() <= dt]
    return eligible[-1] if eligible else None


def pos_on_or_after(df: pd.DataFrame, date_str: str) -> Optional[int]:
    if df.empty:
        return None
    dt = pd.Timestamp(date_str).normalize()
    for i, ts in enumerate(df.index):
        if pd.Timestamp(ts).normalize() >= dt:
            return i
    return None


def trading_days_between(df: pd.DataFrame, start_pos: int, end_pos: int) -> int:
    return max(0, end_pos - start_pos)


def extract_slice_until(df: pd.DataFrame, as_of_saturday: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp(as_of_saturday).normalize()
    return df[df.index.normalize() <= cutoff].copy()


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


def fetch_histories(symbols: List[str], first_saturday: str) -> Dict[str, pd.DataFrame]:
    start_dt = parse_date(first_saturday) - timedelta(days=760)
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=3)
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_history_window(sym, start, end)
        if df.empty:
            log("WARN", f"no history for {sym}")
        out[sym] = df
    return out


def latest_price_date(histories: Dict[str, pd.DataFrame]) -> str:
    """Return the latest available market date across loaded price histories."""
    dates: List[pd.Timestamp] = []
    for df in histories.values():
        if df is not None and not df.empty:
            try:
                dates.append(pd.Timestamp(df.index[-1]).normalize())
            except Exception:
                pass
    if not dates:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return max(dates).date().isoformat()


def should_append_current_point(equity_curve: List[Dict[str, Any]], point: Dict[str, Any]) -> bool:
    """Append/replace only when the last curve point is not already the current valuation."""
    if not equity_curve:
        return True
    last = equity_curve[-1]
    keys = ["portfolio_equity", "spy_equity", "external_capital", "cash", "market_value"]
    return any((to_float(last.get(k)) or 0.0) != (to_float(point.get(k)) or 0.0) for k in keys) or last.get("date") != point.get("date")


def row_price(df: pd.DataFrame, pos: Optional[int], field: str = "Close") -> Optional[float]:
    if pos is None or pos < 0 or pos >= len(df):
        return None
    if field not in df.columns:
        field = "Close"
    return to_float(df.iloc[pos].get(field))


def price_on_or_before(df: pd.DataFrame, date_str: str, field: str = "Close") -> Optional[float]:
    return row_price(df, pos_on_or_before(df, date_str), field)


def buy_price(open_price: float) -> float:
    return open_price * (1.0 + BUY_SLIPPAGE)


def sell_price(raw_price: float) -> float:
    return raw_price * (1.0 - SELL_SLIPPAGE)


def score_band(score: Any) -> str:
    s = to_float(score)
    if s is None:
        return "Unknown"
    if s >= 850:
        return "850+"
    if s >= 800:
        return "800-849"
    if s >= 750:
        return "750-799"
    if s >= 700:
        return "700-749"
    return "<700"


def sequence_bucket(seq: Any) -> str:
    try:
        s = int(seq)
    except Exception:
        return "Unknown"
    if s <= 1:
        return "1st Buy"
    if s == 2:
        return "2nd Buy"
    return "3rd+ Buy"


@dataclass
class SimPolicy:
    name: str
    label: str
    stop_pct: float = DEFAULT_STOP_PCT
    score_exit: int = DEFAULT_SCORE_EXIT
    time_exit_days: Optional[int] = DEFAULT_TIME_EXIT_DAYS
    reinvest: bool = True
    base_shares: float = BASE_SHARES


DEFAULT_POLICY = SimPolicy(
    name="default_12w",
    label="Default 12W",
    stop_pct=DEFAULT_STOP_PCT,
    score_exit=DEFAULT_SCORE_EXIT,
    time_exit_days=DEFAULT_TIME_EXIT_DAYS,
    reinvest=True,
)

COMPARISON_POLICIES = [
    DEFAULT_POLICY,
    SimPolicy("conservative_8w", "Conservative 8W", stop_pct=0.05, score_exit=750, time_exit_days=40, reinvest=True),
    SimPolicy("trend_follow_16w", "Trend Follow 16W", stop_pct=0.08, score_exit=650, time_exit_days=80, reinvest=True),
    SimPolicy("no_time_exit", "No Time Exit", stop_pct=0.05, score_exit=700, time_exit_days=None, reinvest=True),
    SimPolicy("no_reinvestment", "No Reinvestment", stop_pct=0.05, score_exit=700, time_exit_days=60, reinvest=False),
]


def build_snapshot(as_of: str, candidates: List[Any], histories: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    raw_items: List[Dict[str, Any]] = []
    rs_values: Dict[str, Optional[float]] = {}

    for c in candidates:
        df = histories.get(c.symbol, pd.DataFrame())
        d0 = extract_slice_until(df, as_of)
        if d0.empty or len(d0) < 210:
            continue
        d = weekly.add_indicators(d0).dropna(subset=["Close", "Volume"])
        if len(d) < 210:
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

        last = d.iloc[-1]
        raw_items.append({
            "symbol": c.symbol,
            "name": c.name,
            "source": c.source,
            "theme": c.theme or "Other",
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
            "close": safe_round(last.get("Close"), 4),
            "sma20": safe_round(last.get("SMA20"), 4),
            "sma50": safe_round(last.get("SMA50"), 4),
            "avg_dollar_volume_20d": safe_round(d["DOLLAR_VOL"].tail(20).mean(), 2) if "DOLLAR_VOL" in d else None,
            "snapshot_close_date": d.index[-1].date().isoformat(),
        })

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
        "items_by_symbol": {x["symbol"]: x for x in items},
        "qualified": qualified,
        "summary": {
            "valid_items": len(items),
            "qualified_signals": len(qualified),
            "fresh_breakouts": sum(1 for x in qualified if x["signal"] == "A+ Fresh Breakout"),
            "leaders": sum(1 for x in qualified if x["signal"] == "A Leader"),
            "constructive_setups": sum(1 for x in qualified if x["signal"] == "B Constructive Setup"),
        },
    }


def build_snapshots(saturdays: List[str], candidates: List[Any], histories: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    out = []
    for as_of in saturdays:
        log("INFO", f"Build weekly simulation snapshot as_of={as_of}, candidates={len(candidates)}")
        out.append(build_snapshot(as_of, candidates, histories))
    return out


def spy_value_for_lot(lot: Dict[str, Any], histories: Dict[str, pd.DataFrame], current: bool = False, exit_date: Optional[str] = None) -> Optional[float]:
    spy = histories.get(BENCHMARK, pd.DataFrame())
    if spy.empty:
        return None
    shares = to_float(lot.get("spy_shares")) or 0.0
    if shares <= 0:
        return None
    if current:
        px = row_price(spy, len(spy) - 1, "Close")
    else:
        pos = pos_on_or_after(spy, exit_date or lot.get("exit_date") or lot.get("date"))
        px = row_price(spy, pos, "Close")
        if px is not None:
            px = sell_price(px)
    return shares * px if px is not None else None


def lot_mfe_mae(lot: Dict[str, Any], histories: Dict[str, pd.DataFrame], end_pos: Optional[int] = None) -> Tuple[Optional[float], Optional[float]]:
    sym = lot["symbol"]
    df = histories.get(sym, pd.DataFrame())
    if df.empty:
        return None, None
    start = lot.get("entry_pos")
    if start is None:
        return None, None
    if end_pos is None:
        end_pos = lot.get("exit_pos")
    if end_pos is None:
        end_pos = len(df) - 1
    start = max(0, int(start))
    end_pos = min(len(df) - 1, int(end_pos))
    if end_pos < start:
        return None, None
    window = df.iloc[start:end_pos + 1]
    entry = to_float(lot.get("price"))
    if entry is None or entry <= 0 or window.empty:
        return None, None
    hi = to_float(window["High"].max() if "High" in window else window["Close"].max())
    lo = to_float(window["Low"].min() if "Low" in window else window["Close"].min())
    mfe = pct(hi, entry) if hi is not None else None
    mae = pct(lo, entry) if lo is not None else None
    return safe_round(mfe, 2), safe_round(mae, 2)


def position_stats(lots: List[Dict[str, Any]]) -> Dict[str, float]:
    shares = sum_float(l.get("shares") for l in lots)
    cost = sum_float(l.get("cost") for l in lots)
    avg_cost = cost / shares if shares > 0 else 0.0
    return {"shares": shares, "cost": cost, "avg_cost": avg_cost}


def classify_regime(as_of: str, histories: Dict[str, pd.DataFrame]) -> str:
    spy = histories.get(BENCHMARK, pd.DataFrame())
    d0 = extract_slice_until(spy, as_of)
    if len(d0) < 60:
        return "Unknown"
    d = weekly.add_indicators(d0)
    last = d.iloc[-1]
    close = to_float(last.get("Close"))
    sma50 = to_float(last.get("SMA50"))
    if close is None or sma50 is None:
        return "Unknown"
    return "Risk-on" if close >= sma50 else "Risk-off"


def make_lot(
    symbol: str,
    buy_type: str,
    shares: float,
    price: float,
    cost: float,
    signal_date: str,
    buy_date: str,
    entry_pos: int,
    signal: Dict[str, Any],
    sequence_no: int,
    histories: Dict[str, pd.DataFrame],
    spy_cost: Optional[float] = None,
) -> Dict[str, Any]:
    # Strategy cost and benchmark cost are intentionally separated.
    # Base buys inject the same external capital into both portfolios.
    # Reinvestment buys use each portfolio's own realized cash, so SPY may reinvest
    # a different dollar amount than the strategy if prior performance differs.
    spy = histories.get(BENCHMARK, pd.DataFrame())
    spy_pos = pos_on_or_after(spy, buy_date)
    spy_open = row_price(spy, spy_pos, "Open")
    spy_entry = buy_price(spy_open) if spy_open is not None else None
    spy_cost_value = cost if spy_cost is None else max(0.0, float(spy_cost))
    spy_shares = spy_cost_value / spy_entry if spy_entry and spy_entry > 0 else 0.0
    regime = classify_regime(signal_date, histories)
    return {
        "symbol": symbol,
        "name": signal.get("name"),
        "theme": signal.get("theme") or "Other",
        "buy_type": buy_type,
        "shares": shares,
        "price": price,
        "cost": cost,
        "signal_date": signal_date,
        "buy_date": buy_date,
        "entry_pos": entry_pos,
        "signal": signal.get("signal"),
        "score": signal.get("weekly_score"),
        "score_band": score_band(signal.get("weekly_score")),
        "sequence_no": sequence_no,
        "sequence_bucket": sequence_bucket(sequence_no),
        "regime": regime,
        "spy_entry_price": safe_round(spy_entry, 4),
        "spy_cost": safe_round(spy_cost_value, 4),
        "spy_shares": spy_shares,
    }


def lot_current_value(lot: Dict[str, Any], histories: Dict[str, pd.DataFrame]) -> Optional[float]:
    df = histories.get(lot["symbol"], pd.DataFrame())
    if df.empty:
        return None
    px = row_price(df, len(df) - 1, "Close")
    return (to_float(lot.get("shares")) or 0.0) * px if px is not None else None


def close_lots(
    lots: List[Dict[str, Any]],
    exit_date: str,
    exit_pos: int,
    exit_price_raw: float,
    exit_reason: str,
    histories: Dict[str, pd.DataFrame],
    trade_log: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float, float]:
    closed: List[Dict[str, Any]] = []
    proceeds = 0.0
    spy_proceeds = 0.0
    px = sell_price(exit_price_raw)
    for lot in lots:
        shares = to_float(lot.get("shares")) or 0.0
        cost = to_float(lot.get("cost")) or 0.0
        exit_value = shares * px - COMMISSION
        mfe, mae = lot_mfe_mae(lot, histories, exit_pos)
        ret = pct(exit_value, cost)
        spy_value = spy_value_for_lot(lot, histories, current=False, exit_date=exit_date)
        spy_cost = to_float(lot.get("spy_cost")) or 0.0
        spy_ret = pct(spy_value, spy_cost) if spy_value is not None and spy_cost > 0 else None
        rec = dict(lot)
        rec.update({
            "exit_date": exit_date,
            "exit_price": safe_round(px, 4),
            "exit_reason": exit_reason,
            "exit_pos": exit_pos,
            "exit_value": safe_round(exit_value, 4),
            "realized_pl": safe_round(exit_value - cost, 4),
            "return_pct": safe_round(ret, 2),
            "spy_cost": safe_round(spy_cost, 4),
            "spy_value": safe_round(spy_value, 4),
            "spy_return_pct": safe_round(spy_ret, 2),
            "alpha_pct": safe_round((ret or 0) - (spy_ret or 0), 2) if ret is not None and spy_ret is not None else None,
            "mfe_pct": mfe,
            "mae_pct": mae,
            "holding_days": trading_days_between(histories[lot["symbol"]], int(lot.get("entry_pos", 0)), exit_pos),
        })
        closed.append(rec)
        proceeds += max(0.0, exit_value)
        spy_proceeds += spy_value or 0.0
        trade_log.append({
            "date": exit_date,
            "event": "sell",
            "symbol": lot["symbol"],
            "shares": safe_round(shares, 4),
            "price": safe_round(px, 4),
            "value": safe_round(exit_value, 2),
            "exit_reason": exit_reason,
            "realized_pl": safe_round(exit_value - cost, 2),
        })
    return closed, proceeds, spy_proceeds


def current_lot_record(lot: Dict[str, Any], histories: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    cur_val = lot_current_value(lot, histories)
    cost = to_float(lot.get("cost")) or 0.0
    mfe, mae = lot_mfe_mae(lot, histories)
    spy_val = spy_value_for_lot(lot, histories, current=True)
    spy_cost = to_float(lot.get("spy_cost")) or 0.0
    ret = pct(cur_val, cost) if cur_val is not None else None
    spy_ret = pct(spy_val, spy_cost) if spy_val is not None and spy_cost > 0 else None
    rec = dict(lot)
    rec.update({
        "current_value": safe_round(cur_val, 4),
        "unrealized_pl": safe_round((cur_val or 0.0) - cost, 4) if cur_val is not None else None,
        "return_pct": safe_round(ret, 2),
        "spy_cost": safe_round(spy_cost, 4),
        "spy_value": safe_round(spy_val, 4),
        "spy_return_pct": safe_round(spy_ret, 2),
        "alpha_pct": safe_round((ret or 0) - (spy_ret or 0), 2) if ret is not None and spy_ret is not None else None,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "status": "open",
    })
    return rec


def summarize_group(records: List[Dict[str, Any]], key: str, label: str = "label") -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        k = str(r.get(key) or "Unknown")
        groups.setdefault(k, []).append(r)
    out = []
    for k, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        cost = sum_float(r.get("cost") for r in rows)
        value = sum_float(r.get("exit_value", r.get("current_value")) for r in rows)
        spy_cost = sum_float(r.get("spy_cost") for r in rows)
        spy_value = sum_float(r.get("spy_value") for r in rows)
        strategy_ret = pct(value, cost)
        benchmark_ret = pct(spy_value, spy_cost)
        alpha_value = value - spy_value
        out.append({
            label: k,
            "count": len(rows),
            "cost": safe_round(cost, 2),
            "value": safe_round(value, 2),
            "pl_value": safe_round(value - cost, 2),
            "spy_cost": safe_round(spy_cost, 2),
            "spy_value": safe_round(spy_value, 2),
            "alpha_value": safe_round(alpha_value, 2),
            "return_pct": safe_round(strategy_ret, 2),
            "spy_return_pct": safe_round(benchmark_ret, 2),
            "alpha_pct": safe_round((strategy_ret or 0) - (benchmark_ret or 0), 2) if strategy_ret is not None and benchmark_ret is not None else None,
            "win_rate": safe_round(win_rate([r.get("return_pct") for r in rows]), 4),
            "avg_return_pct": safe_round(avg([r.get("return_pct") for r in rows]), 2),
            "avg_mfe_pct": safe_round(avg([r.get("mfe_pct") for r in rows]), 2),
            "avg_mae_pct": safe_round(avg([r.get("mae_pct") for r in rows]), 2),
        })
    return out


def select_trade_extreme(records: List[Dict[str, Any]], best: bool = True) -> Optional[Dict[str, Any]]:
    valid = [r for r in records if to_float(r.get("return_pct")) is not None]
    if not valid:
        return None
    row = max(valid, key=lambda r: to_float(r.get("return_pct")) or 0.0) if best else min(valid, key=lambda r: to_float(r.get("return_pct")) or 0.0)
    return {
        "symbol": row.get("symbol"),
        "theme": row.get("theme"),
        "buy_type": row.get("buy_type"),
        "entry_date": row.get("buy_date"),
        "exit_date": row.get("exit_date"),
        "status": row.get("status", "closed" if row.get("exit_date") else "open"),
        "return_pct": safe_round(row.get("return_pct"), 2),
        "pl_value": safe_round((to_float(row.get("exit_value", row.get("current_value"))) or 0.0) - (to_float(row.get("cost")) or 0.0), 2),
        "cost": safe_round(row.get("cost"), 2),
        "value": safe_round(row.get("exit_value", row.get("current_value")), 2),
        "mfe_pct": safe_round(row.get("mfe_pct"), 2),
        "mae_pct": safe_round(row.get("mae_pct"), 2),
        "exit_reason": row.get("exit_reason"),
    }


def simulate_policy(policy: SimPolicy, snapshots: List[Dict[str, Any]], histories: Dict[str, pd.DataFrame], details: bool = True) -> Dict[str, Any]:
    open_lots: Dict[str, List[Dict[str, Any]]] = {}
    closed_lots: List[Dict[str, Any]] = []
    trade_log: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []
    liquidity_warnings: List[Dict[str, Any]] = []
    cash = 0.0
    spy_cash = 0.0
    total_new_capital = 0.0
    total_reinvested_capital = 0.0

    for snap in snapshots:
        as_of = snap["as_of"]
        items_by_symbol = snap.get("items_by_symbol", {})
        qualified = snap.get("qualified", [])

        # 1) Stop loss scan from prior check through the current snapshot date.
        for sym in list(open_lots.keys()):
            lots = open_lots.get(sym, [])
            if not lots:
                continue
            df = histories.get(sym, pd.DataFrame())
            if df.empty:
                continue
            stats = position_stats(lots)
            stop_px = stats["avg_cost"] * (1.0 - policy.stop_pct)
            start_pos = min(int(l.get("last_check_pos", l.get("entry_pos", 0))) for l in lots)
            end_pos = pos_on_or_before(df, as_of)
            if end_pos is None:
                continue
            trigger_pos = None
            for p in range(max(0, start_pos + 1), end_pos + 1):
                low = row_price(df, p, "Low")
                if low is not None and low <= stop_px:
                    trigger_pos = p
                    break
            for l in lots:
                l["last_check_pos"] = end_pos
            if trigger_pos is not None:
                exit_date = df.index[trigger_pos].date().isoformat()
                closed, proceeds, spy_proceeds = close_lots(lots, exit_date, trigger_pos, stop_px, "stop_loss", histories, trade_log)
                closed_lots.extend(closed)
                cash += proceeds
                spy_cash += spy_proceeds
                open_lots.pop(sym, None)

        # 2) Weekly score/trend exits at next trading day open.
        for sym in list(open_lots.keys()):
            lots = open_lots.get(sym, [])
            if not lots:
                continue
            df = histories.get(sym, pd.DataFrame())
            entry_pos = next_trading_pos_after(df, as_of)
            if entry_pos is None:
                continue
            entry_open = row_price(df, entry_pos, "Open")
            if entry_open is None:
                continue
            item = items_by_symbol.get(sym)
            exit_reason = None
            if item:
                close = to_float(item.get("close"))
                sma50 = to_float(item.get("sma50"))
                score = to_float(item.get("weekly_score"))
                if close is not None and sma50 is not None and close < sma50:
                    exit_reason = "trend_exit"
                elif score is not None and score < policy.score_exit:
                    exit_reason = "score_exit"
            if exit_reason:
                exit_date = df.index[entry_pos].date().isoformat()
                closed, proceeds, spy_proceeds = close_lots(lots, exit_date, entry_pos, entry_open, exit_reason, histories, trade_log)
                closed_lots.extend(closed)
                cash += proceeds
                spy_cash += spy_proceeds
                open_lots.pop(sym, None)

        # 3) Lot-level time exits at next trading day open.
        if policy.time_exit_days is not None:
            for sym in list(open_lots.keys()):
                lots = open_lots.get(sym, [])
                if not lots:
                    continue
                df = histories.get(sym, pd.DataFrame())
                exit_pos = next_trading_pos_after(df, as_of)
                if exit_pos is None:
                    continue
                exit_open = row_price(df, exit_pos, "Open")
                if exit_open is None:
                    continue
                expired = [l for l in lots if trading_days_between(df, int(l.get("entry_pos", 0)), exit_pos) >= int(policy.time_exit_days or 0)]
                if expired:
                    exit_date = df.index[exit_pos].date().isoformat()
                    closed, proceeds, spy_proceeds = close_lots(expired, exit_date, exit_pos, exit_open, f"time_exit_{int(policy.time_exit_days/5)}w", histories, trade_log)
                    closed_lots.extend(closed)
                    cash += proceeds
                    spy_cash += spy_proceeds
                    remaining_ids = {id(l) for l in expired}
                    remaining = [l for l in lots if id(l) not in remaining_ids]
                    if remaining:
                        open_lots[sym] = remaining
                    else:
                        open_lots.pop(sym, None)

        # 4) Buy current weekly B+ signals.
        valid_signals = []
        for q in qualified:
            sym = q["symbol"]
            df = histories.get(sym, pd.DataFrame())
            pos = next_trading_pos_after(df, as_of)
            op = row_price(df, pos, "Open")
            if pos is not None and op is not None and op > 0:
                valid_signals.append((q, pos, op))
        if valid_signals:
            # Base buy: 1 share per B+ signal, new capital.
            for q, pos, op in valid_signals:
                sym = q["symbol"]
                px = buy_price(op)
                shares = policy.base_shares
                cost = shares * px + COMMISSION
                buy_date = histories[sym].index[pos].date().isoformat()
                seq = len([l for l in open_lots.get(sym, []) if l.get("buy_type") == "base"]) + 1
                lot = make_lot(sym, "base", shares, px, cost, as_of, buy_date, pos, q, seq, histories, spy_cost=cost)
                lot["last_check_pos"] = pos
                open_lots.setdefault(sym, []).append(lot)
                total_new_capital += cost
                trade_log.append({
                    "date": buy_date,
                    "event": "buy",
                    "buy_type": "base",
                    "symbol": sym,
                    "shares": safe_round(shares, 4),
                    "price": safe_round(px, 4),
                    "cost": safe_round(cost, 2),
                    "score": q.get("weekly_score"),
                    "signal": q.get("signal"),
                })
                adv = to_float(q.get("avg_dollar_volume_20d"))
                if adv and cost > adv * LIQUIDITY_WARN_FRACTION:
                    liquidity_warnings.append({"date": buy_date, "symbol": sym, "trade_value": safe_round(cost, 2), "avg_dollar_volume_20d": safe_round(adv, 2), "fraction": safe_round(cost / adv, 6)})

            # Reinvestment buy: deploy available sale proceeds equally across current B+ signals.
            if policy.reinvest and cash > 0:
                allocation = cash / len(valid_signals)
                spy_allocation = spy_cash / len(valid_signals) if spy_cash > 0 else 0.0
                if allocation > 0:
                    deployed = 0.0
                    spy_deployed = 0.0
                    for q, pos, op in valid_signals:
                        sym = q["symbol"]
                        px = buy_price(op)
                        if px <= 0:
                            continue
                        cost = allocation
                        shares = max(0.0, (cost - COMMISSION) / px)
                        if shares <= 0:
                            continue
                        buy_date = histories[sym].index[pos].date().isoformat()
                        seq = len([l for l in open_lots.get(sym, []) if l.get("buy_type") == "base"]) or 1
                        lot = make_lot(sym, "reinvestment", shares, px, cost, as_of, buy_date, pos, q, seq, histories, spy_cost=spy_allocation)
                        lot["last_check_pos"] = pos
                        open_lots.setdefault(sym, []).append(lot)
                        deployed += cost
                        spy_deployed += spy_allocation
                        total_reinvested_capital += cost
                        trade_log.append({
                            "date": buy_date,
                            "event": "buy",
                            "buy_type": "reinvestment",
                            "symbol": sym,
                            "shares": safe_round(shares, 4),
                            "price": safe_round(px, 4),
                            "cost": safe_round(cost, 2),
                            "spy_cost": safe_round(spy_allocation, 2),
                            "score": q.get("weekly_score"),
                            "signal": q.get("signal"),
                        })
                    cash = max(0.0, cash - deployed)
                    spy_cash = max(0.0, spy_cash - spy_deployed)

        # 5) Weekly equity snapshot.
        market_value = 0.0
        for sym, lots in open_lots.items():
            df = histories.get(sym, pd.DataFrame())
            pos = pos_on_or_before(df, as_of)
            px = row_price(df, pos, "Close")
            if px is None:
                continue
            market_value += sum_float(l.get("shares") for l in lots) * px
        portfolio_equity = cash + market_value
        spy_open_value = sum((spy_value_for_lot(l, histories, current=True) or 0.0) for lots in open_lots.values() for l in lots)
        spy_equity = spy_cash + spy_open_value
        net_return = pct(portfolio_equity, total_new_capital) if total_new_capital else None
        spy_return_equity = pct(spy_equity, total_new_capital) if total_new_capital else None
        equity_multiple = ratio(portfolio_equity, total_new_capital) if total_new_capital else None
        peak_multiple = max([e.get("equity_multiple", 0) for e in equity_curve] + ([equity_multiple] if equity_multiple is not None else [0]))
        dd = pct(equity_multiple, peak_multiple) if equity_multiple is not None and peak_multiple else None
        exposure = ratio(market_value, portfolio_equity)
        equity_curve.append({
            "date": as_of,
            "external_capital": safe_round(total_new_capital, 2),
            "new_capital": safe_round(total_new_capital, 2),
            "reinvested_capital": safe_round(total_reinvested_capital, 2),
            "cash": safe_round(cash, 2),
            "market_value": safe_round(market_value, 2),
            "portfolio_equity": safe_round(portfolio_equity, 2),
            "net_return_pct": safe_round(net_return, 2),
            "spy_cash": safe_round(spy_cash, 2),
            "spy_market_value": safe_round(spy_open_value, 2),
            "spy_equity": safe_round(spy_equity, 2),
            "spy_return_pct": safe_round(spy_return_equity, 2),
            "alpha_value": safe_round(portfolio_equity - spy_equity, 2),
            "alpha_pct": safe_round((net_return or 0) - (spy_return_equity or 0), 2) if net_return is not None and spy_return_equity is not None else None,
            "equity_multiple": safe_round(equity_multiple, 6),
            "drawdown_pct": safe_round(dd, 2),
            "exposure_pct": safe_round((exposure or 0) * 100, 2),
        })

    open_lot_records = [current_lot_record(l, histories) for lots in open_lots.values() for l in lots]
    closed_records = closed_lots

    # Aggregated open positions by symbol.
    open_positions = []
    for sym, lots in sorted(open_lots.items()):
        stats = position_stats(lots)
        df = histories.get(sym, pd.DataFrame())
        cur = row_price(df, len(df) - 1, "Close")
        market_value = stats["shares"] * cur if cur is not None else 0.0
        current_score = None
        current_signal = None
        current_theme = lots[-1].get("theme") if lots else "Other"
        if snapshots:
            latest_item = snapshots[-1].get("items_by_symbol", {}).get(sym)
            if latest_item:
                current_score = latest_item.get("weekly_score")
                current_signal = latest_item.get("signal")
                current_theme = latest_item.get("theme") or current_theme
        stop_px = stats["avg_cost"] * (1.0 - policy.stop_pct)
        open_positions.append({
            "symbol": sym,
            "theme": current_theme,
            "shares": safe_round(stats["shares"], 4),
            "avg_cost": safe_round(stats["avg_cost"], 4),
            "total_cost": safe_round(stats["cost"], 2),
            "current_price": safe_round(cur, 4),
            "market_value": safe_round(market_value, 2),
            "unrealized_pl": safe_round(market_value - stats["cost"], 2),
            "unrealized_return_pct": safe_round(pct(market_value, stats["cost"]), 2),
            "current_score": current_score,
            "last_signal": current_signal or lots[-1].get("signal"),
            "stop_price": safe_round(stop_px, 4),
            "distance_to_stop_pct": safe_round(pct(cur, stop_px), 2) if cur is not None and stop_px > 0 else None,
            "lots": len(lots),
        })

    gross_buy_cost = total_new_capital + total_reinvested_capital
    # Turnover metrics are useful, but they are not portfolio equity.
    # Closed proceeds may have been reinvested, so closed proceeds + open value would double count capital.
    turnover_current_value = sum_float(r.get("exit_value") for r in closed_records) + sum_float(r.get("current_value") for r in open_lot_records)
    current_market = sum_float(p.get("market_value") for p in open_positions)
    current_equity = cash + current_market
    spy_open_value = sum((spy_value_for_lot(l, histories, current=True) or 0.0) for lots in open_lots.values() for l in lots)
    spy_equivalent = spy_cash + spy_open_value
    return_on_new_capital = pct(current_equity, total_new_capital) if total_new_capital else None
    net_return = return_on_new_capital
    turnover_return = pct(turnover_current_value, gross_buy_cost) if gross_buy_cost else None
    spy_return = pct(spy_equivalent, total_new_capital) if total_new_capital else None
    alpha = (net_return or 0) - (spy_return or 0) if net_return is not None and spy_return is not None else None

    # Align the final capital curve point with the same current valuation used by top-level KPIs.
    # Earlier curve rows are weekly snapshot valuations; the final row is a current mark-to-market
    # so Portfolio Equity / SPY Equity / Alpha Value cannot drift from the KPI cards.
    current_equity_multiple = ratio(current_equity, total_new_capital) if total_new_capital else None
    prior_peak_multiple = max([to_float(e.get("equity_multiple")) or 0.0 for e in equity_curve] + ([current_equity_multiple] if current_equity_multiple is not None else [0.0]))
    current_dd = pct(current_equity_multiple, prior_peak_multiple) if current_equity_multiple is not None and prior_peak_multiple else None
    current_exposure = ratio(current_market, current_equity)
    current_curve_point = {
        "date": latest_price_date(histories),
        "point_type": "current",
        "external_capital": safe_round(total_new_capital, 2),
        "new_capital": safe_round(total_new_capital, 2),
        "reinvested_capital": safe_round(total_reinvested_capital, 2),
        "cash": safe_round(cash, 2),
        "market_value": safe_round(current_market, 2),
        "portfolio_equity": safe_round(current_equity, 2),
        "net_return_pct": safe_round(net_return, 2),
        "spy_cash": safe_round(spy_cash, 2),
        "spy_market_value": safe_round(spy_open_value, 2),
        "spy_equity": safe_round(spy_equivalent, 2),
        "spy_return_pct": safe_round(spy_return, 2),
        "alpha_value": safe_round(current_equity - spy_equivalent, 2),
        "alpha_pct": safe_round(alpha, 2),
        "equity_multiple": safe_round(current_equity_multiple, 6),
        "drawdown_pct": safe_round(current_dd, 2),
        "exposure_pct": safe_round((current_exposure or 0) * 100, 2),
    }
    if should_append_current_point(equity_curve, current_curve_point):
        if equity_curve and equity_curve[-1].get("date") == current_curve_point.get("date"):
            equity_curve[-1] = current_curve_point
        else:
            equity_curve.append(current_curve_point)

    max_dd = min([e.get("drawdown_pct") for e in equity_curve if e.get("drawdown_pct") is not None] or [0])

    all_lot_records = closed_records + open_lot_records
    closed_win_rate = win_rate([r.get("return_pct") for r in closed_records])
    net_pl_value = current_equity - total_new_capital if total_new_capital else None
    return_drawdown_ratio = None
    if net_return is not None and max_dd is not None and max_dd < 0:
        return_drawdown_ratio = net_return / abs(max_dd)
    best_trade = select_trade_extreme(all_lot_records, best=True)
    worst_trade = select_trade_extreme(all_lot_records, best=False)
    exposure_values = [e.get("exposure_pct") for e in equity_curve if e.get("exposure_pct") is not None]
    largest_position_pct = None
    top5_position_pct = None
    largest_position_symbol = None
    largest_position_value = None
    if current_equity > 0 and open_positions:
        sorted_positions = sorted(open_positions, key=lambda p: to_float(p.get("market_value")) or 0.0, reverse=True)
        vals = [to_float(p.get("market_value")) or 0.0 for p in sorted_positions]
        largest_position_pct = vals[0] / current_equity * 100
        top5_position_pct = sum(vals[:5]) / current_equity * 100
        largest_position_symbol = sorted_positions[0].get("symbol")
        largest_position_value = vals[0]

    summary = {
        "policy_name": policy.name,
        "policy_label": policy.label,
        "benchmark_symbol": BENCHMARK,
        "total_new_capital": safe_round(total_new_capital, 2),
        "reinvested_capital": safe_round(total_reinvested_capital, 2),
        "total_buy_cost": safe_round(gross_buy_cost, 2),
        "turnover_current_value": safe_round(turnover_current_value, 2),
        "gross_current_value": safe_round(turnover_current_value, 2),
        "cash": safe_round(cash, 2),
        "market_value": safe_round(current_market, 2),
        "open_market_value": safe_round(current_market, 2),
        "portfolio_equity": safe_round(current_equity, 2),
        "current_equity": safe_round(current_equity, 2),
        "total_pl": safe_round(current_equity - total_new_capital, 2) if total_new_capital else None,
        "net_pl": safe_round(net_pl_value, 2),
        "net_pl_value": safe_round(net_pl_value, 2),
        "net_return_pct": safe_round(net_return, 2),
        "return_on_new_capital_pct": safe_round(return_on_new_capital, 2),
        "gross_return_pct": safe_round(turnover_return, 2),
        "spy_cash": safe_round(spy_cash, 2),
        "spy_market_value": safe_round(spy_open_value, 2),
        "spy_equity": safe_round(spy_equivalent, 2),
        "spy_equivalent": safe_round(spy_equivalent, 2),
        "spy_return_pct": safe_round(spy_return, 2),
        "alpha_value": safe_round(current_equity - spy_equivalent, 2) if total_new_capital else None,
        "alpha_pct": safe_round(alpha, 2),
        "open_positions": len(open_positions),
        "open_lots": len(open_lot_records),
        "closed_lots": len(closed_records),
        "win_rate": safe_round(closed_win_rate, 4),
        "max_drawdown_pct": safe_round(max_dd, 2),
        "current_drawdown_pct": equity_curve[-1].get("drawdown_pct") if equity_curve else None,
        "return_drawdown_ratio": safe_round(return_drawdown_ratio, 2),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_mfe_pct": safe_round(avg([r.get("mfe_pct") for r in all_lot_records]), 2),
        "avg_mae_pct": safe_round(avg([r.get("mae_pct") for r in all_lot_records]), 2),
        "mfe_mae_ratio": safe_round(abs((avg([r.get("mfe_pct") for r in all_lot_records]) or 0) / (avg([r.get("mae_pct") for r in all_lot_records]) or -1)), 2),
        "average_exposure_pct": safe_round(avg(exposure_values), 2),
        "max_exposure_pct": safe_round(max(exposure_values) if exposure_values else None, 2),
        "current_exposure_pct": safe_round(ratio(current_market, current_equity) * 100 if current_equity else None, 2),
        "largest_position_pct": safe_round(largest_position_pct, 2),
        "largest_position_symbol": largest_position_symbol,
        "largest_position_value": safe_round(largest_position_value, 2),
        "top5_position_pct": safe_round(top5_position_pct, 2),
        "liquidity_warning_count": len(liquidity_warnings),
    }

    comparison_summary = {
        "strategy": policy.label,
        "return_pct": summary["net_return_pct"],
        "return_on_new_capital_pct": summary["return_on_new_capital_pct"],
        "spy_return_pct": summary["spy_return_pct"],
        "alpha_pct": summary["alpha_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "win_rate": summary["win_rate"],
        "closed_lots": summary["closed_lots"],
        "open_lots": summary["open_lots"],
        "avg_mfe_pct": summary["avg_mfe_pct"],
        "avg_mae_pct": summary["avg_mae_pct"],
        "avg_exposure_pct": summary["average_exposure_pct"],
    }

    if not details:
        return comparison_summary

    return {
        "summary": summary,
        "open_positions": sorted(open_positions, key=lambda x: (to_float(x.get("market_value")) or 0), reverse=True),
        "closed_trades": sorted(closed_records, key=lambda x: x.get("exit_date") or "", reverse=True),
        "trade_log": trade_log[-MAX_TRADE_LOG_ROWS:],
        "equity_curve": equity_curve,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "exit_reason_summary": summarize_group(closed_records, "exit_reason", "exit_reason"),
        "buy_type_summary": summarize_group(all_lot_records, "buy_type", "buy_type"),
        "add_on_sequence_summary": summarize_group(all_lot_records, "sequence_bucket", "sequence"),
        "score_band_summary": summarize_group(all_lot_records, "score_band", "score_band"),
        "signal_type_summary": summarize_group(all_lot_records, "signal", "signal"),
        "theme_summary": summarize_group(all_lot_records, "theme", "theme"),
        "regime_summary": summarize_group(all_lot_records, "regime", "regime"),
        "exposure_summary": {
            "average_exposure_pct": summary["average_exposure_pct"],
            "max_exposure_pct": summary["max_exposure_pct"],
            "current_exposure_pct": summary["current_exposure_pct"],
            "largest_position_pct": summary["largest_position_pct"],
            "largest_position_symbol": summary["largest_position_symbol"],
            "largest_position_value": summary["largest_position_value"],
            "top5_position_pct": summary["top5_position_pct"],
            "cash_ratio_pct": safe_round(ratio(cash, current_equity) * 100 if current_equity else None, 2),
        },
        "liquidity_warnings": liquidity_warnings[-100:],
        "comparison_summary": comparison_summary,
    }


def main() -> None:
    saturdays = get_backtest_saturdays(BACKTEST_WEEKS, BACKTEST_END_DATE or None)
    log("INFO", f"Weekly simulation weeks={BACKTEST_WEEKS}, snapshots={saturdays}, benchmark={BENCHMARK}")
    candidates = [c for c in weekly.load_candidates() if not c.exclude]
    if not candidates:
        raise SystemExit("No weekly candidates found. Add rows to data/weekly_candidates.csv")
    symbols = sorted({c.symbol for c in candidates} | {BENCHMARK})
    histories = fetch_histories(symbols, saturdays[0])
    snapshots = build_snapshots(saturdays, candidates, histories)

    default_result = simulate_policy(DEFAULT_POLICY, snapshots, histories, details=True)
    strategy_comparison = []
    for policy in COMPARISON_POLICIES:
        if policy.name == DEFAULT_POLICY.name:
            strategy_comparison.append(default_result["comparison_summary"])
        else:
            strategy_comparison.append(simulate_policy(policy, snapshots, histories, details=False))

    default_cmp = next((r for r in strategy_comparison if r.get("strategy") == DEFAULT_POLICY.label), strategy_comparison[0] if strategy_comparison else {})
    no_reinvest_cmp = next((r for r in strategy_comparison if str(r.get("strategy", "")).lower().startswith("no reinvestment")), None)
    best_return_cmp = max(strategy_comparison, key=lambda r: to_float(r.get("return_pct")) if to_float(r.get("return_pct")) is not None else -999999) if strategy_comparison else None
    lowest_dd_cmp = max(strategy_comparison, key=lambda r: to_float(r.get("max_drawdown_pct")) if to_float(r.get("max_drawdown_pct")) is not None else -999999) if strategy_comparison else None
    best_risk_adjusted_cmp = max(
        strategy_comparison,
        key=lambda r: ((to_float(r.get("return_pct")) or 0.0) / abs(to_float(r.get("max_drawdown_pct")) or -1.0)) if (to_float(r.get("max_drawdown_pct")) or 0.0) < 0 else -999999,
    ) if strategy_comparison else None
    reinvestment_impact = None
    if no_reinvest_cmp:
        reinvestment_impact = {
            "return_delta_pct": safe_round((to_float(default_cmp.get("return_pct")) or 0.0) - (to_float(no_reinvest_cmp.get("return_pct")) or 0.0), 2),
            "alpha_delta_pct": safe_round((to_float(default_cmp.get("alpha_pct")) or 0.0) - (to_float(no_reinvest_cmp.get("alpha_pct")) or 0.0), 2),
            "default_return_pct": safe_round(default_cmp.get("return_pct"), 2),
            "no_reinvestment_return_pct": safe_round(no_reinvest_cmp.get("return_pct"), 2),
        }

    strategy_highlights = {
        "best_return": best_return_cmp,
        "lowest_drawdown": lowest_dd_cmp,
        "best_risk_adjusted": best_risk_adjusted_cmp,
        "reinvestment_impact": reinvestment_impact,
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "policy": {
            "name": "Weekly B+ Add-on Reinvestment Simulation",
            "entry": "Buy every B-or-better weekly signal at the next trading day open.",
            "base_buy": f"{BASE_SHARES:g} share per weekly B+ signal, treated as new capital.",
            "reinvestment": "Available realized proceeds are equally reinvested into the current week's B+ signals; fractional shares are allowed.",
            "exit_priority": ["stop_loss", "trend_exit", "score_exit", "time_exit"],
            "stop_loss": f"Average cost -{DEFAULT_STOP_PCT * 100:.1f}% triggers symbol-level exit.",
            "trend_exit": "Close < SMA50 triggers symbol-level exit at next open.",
            "score_exit": f"Weekly score < {DEFAULT_SCORE_EXIT} triggers symbol-level exit at next open.",
            "time_exit": f"Default lot-level time exit: {DEFAULT_TIME_EXIT_DAYS} trading days / about {DEFAULT_TIME_EXIT_DAYS // 5} weeks.",
            "benchmark": f"{BENCHMARK} equivalent uses the same buy dates, buy amounts, and exit dates as the simulated lots.",
            "cost_policy": {
                "commission_per_trade": COMMISSION,
                "buy_slippage_pct": BUY_SLIPPAGE * 100,
                "sell_slippage_pct": SELL_SLIPPAGE * 100,
            },
            "data_policy": {
                "signal_data_cutoff": "Saturday snapshot based on prior market close data.",
                "entry_price": "Next trading day open with buy slippage.",
                "exit_price": "Stop price or next trading day open with sell slippage.",
            },
        },
        "snapshots": [{"as_of": s["as_of"], "summary": s["summary"]} for s in snapshots],
        "summary": default_result["summary"],
        "open_positions": default_result["open_positions"],
        "closed_trades": default_result["closed_trades"],
        "trade_log": default_result["trade_log"],
        "equity_curve": default_result["equity_curve"],
        "chart_data": {
            "equity_curve": default_result["equity_curve"],
            "strategy_comparison": strategy_comparison,
            "exit_reason_summary": default_result["exit_reason_summary"],
            "theme_summary": default_result["theme_summary"],
        },
        "exit_reason_summary": default_result["exit_reason_summary"],
        "buy_type_summary": default_result["buy_type_summary"],
        "add_on_sequence_summary": default_result["add_on_sequence_summary"],
        "score_band_summary": default_result["score_band_summary"],
        "signal_type_summary": default_result["signal_type_summary"],
        "theme_summary": default_result["theme_summary"],
        "regime_summary": default_result["regime_summary"],
        "exposure_summary": default_result["exposure_summary"],
        "liquidity_warnings": default_result["liquidity_warnings"],
        "best_trade": default_result.get("best_trade"),
        "worst_trade": default_result.get("worst_trade"),
        "strategy_comparison": strategy_comparison,
        "strategy_highlights": strategy_highlights,
    }

    out_dir = OUT_DIR / "data" / "weekly" / "simulation"
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
        log("ERROR", f"FATAL in rebuild_weekly_simulation: {e}")
        sys.exit(1)
