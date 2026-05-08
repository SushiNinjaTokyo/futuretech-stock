#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
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
CANDIDATES_CSV = Path(os.getenv("WEEKLY_CANDIDATES_CSV", str(ROOT / "data" / "weekly_candidates.csv")))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", str(ROOT / "data" / "universe.csv")))
REPORT_DATE = os.getenv("REPORT_DATE")
BENCHMARK = os.getenv("WEEKLY_BENCHMARK", "SPY").strip().upper() or "SPY"
MAX_SYMBOLS = int(os.getenv("WEEKLY_MAX_SYMBOLS", "250"))
ENABLE_FUNDAMENTALS = os.getenv("WEEKLY_ENABLE_FUNDAMENTALS", "false").strip().lower() in {"1", "true", "yes", "on"}


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def resolve_report_date() -> str:
    if REPORT_DATE:
        return REPORT_DATE
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from et_market_date import ET, resolve_report_date as resolve  # type: ignore
        return resolve(datetime.now(ET), int(os.getenv("MARKET_CUTOFF_HOUR_ET", "20"))).isoformat()
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


def pct(cur: Any, prev: Any) -> Optional[float]:
    c = to_float(cur)
    p = to_float(prev)
    if c is None or p is None or p == 0:
        return None
    return (c / p - 1.0) * 100.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_round(x: Any, ndigits: int = 2) -> Optional[float]:
    v = to_float(x)
    return round(v, ndigits) if v is not None else None


@dataclass
class Candidate:
    symbol: str
    name: str = ""
    source: str = "manual"
    theme: str = ""
    trigger: str = "watch"
    note: str = ""
    priority: str = ""
    exclude: bool = False


def load_candidates() -> List[Candidate]:
    rows = read_csv_rows(CANDIDATES_CSV)
    if not rows:
        log("WARN", f"{CANDIDATES_CSV} not found or empty. Falling back to universe.csv.")
        rows = read_csv_rows(UNIVERSE_CSV)
        for r in rows:
            r.setdefault("source", "core")
            r.setdefault("trigger", "universe")
            r.setdefault("note", "fallback from universe")

    dedup: Dict[str, Candidate] = {}
    for r in rows[:MAX_SYMBOLS]:
        sym = (r.get("symbol") or r.get("ticker") or "").strip().upper()
        if not sym:
            continue
        exclude = str(r.get("exclude", "false")).strip().lower() in {"1", "true", "yes", "y"}
        cand = Candidate(
            symbol=sym,
            name=r.get("name", ""),
            source=r.get("source", "manual") or "manual",
            theme=r.get("theme", ""),
            trigger=r.get("trigger", "watch") or "watch",
            note=r.get("note", ""),
            priority=r.get("priority", ""),
            exclude=exclude,
        )
        if sym in dedup:
            old = dedup[sym]
            old.source = " + ".join(sorted(set([x for x in (old.source + " + " + cand.source).split(" + ") if x])))
            old.trigger = " + ".join(sorted(set([x for x in (old.trigger + " + " + cand.trigger).split(" + ") if x])))
            old.theme = old.theme or cand.theme
            old.note = old.note or cand.note
            old.exclude = old.exclude and cand.exclude
        else:
            dedup[sym] = cand
    return list(dedup.values())


def extract_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return either field->ticker or ticker->field depending on version/group_by.
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, level=-1, axis=1)
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, level=0, axis=1)
        else:
            df.columns = [c[0] for c in df.columns]
    wanted = {}
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            wanted[col] = pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame(wanted)
    if "Close" not in out or "Volume" not in out:
        return pd.DataFrame()
    out.index = pd.to_datetime(out.index)
    return out.dropna(subset=["Close", "Volume"])


def fetch_history(symbol: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Add yfinance to requirements.txt.")
    try:
        raw = yf.download(symbol, period="18mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        return extract_ohlcv(raw, symbol)
    except Exception as e:
        log("WARN", f"history fetch failed: {symbol}: {e}")
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["Close"]
    high = d.get("High", close)
    low = d.get("Low", close)
    volume = d["Volume"]

    for n in [10, 20, 50, 150, 200]:
        d[f"SMA{n}"] = close.rolling(n).mean()
        d[f"VOL{n}"] = volume.rolling(n).mean()

    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    for n in [10, 20, 60]:
        d[f"ATR{n}"] = tr.rolling(n).mean()
        d[f"ATR{n}_PCT"] = d[f"ATR{n}"] / close

    d["HH20_PREV"] = high.shift(1).rolling(20).max()
    d["HH55_PREV"] = high.shift(1).rolling(55).max()
    d["LL252"] = low.rolling(252).min()
    d["HH252"] = high.rolling(252).max()
    d["RET"] = close.pct_change()
    d["DOLLAR_VOL"] = close * volume
    return d


def value_at(s: pd.Series, idx_from_end: int) -> Optional[float]:
    try:
        if len(s.dropna()) == 0:
            return None
        return to_float(s.iloc[idx_from_end])
    except Exception:
        return None


def score_trend(d: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
    r = d.iloc[-1]
    close = to_float(r.get("Close")) or 0.0
    sma50 = to_float(r.get("SMA50"))
    sma150 = to_float(r.get("SMA150"))
    sma200 = to_float(r.get("SMA200"))
    hh252 = to_float(r.get("HH252"))
    ll252 = to_float(r.get("LL252"))
    sma200_20 = value_at(d["SMA200"], -21) if len(d) >= 221 else None
    sma200_60 = value_at(d["SMA200"], -61) if len(d) >= 261 else None

    weekly = d.resample("W-FRI").last().dropna(subset=["Close"])
    w10 = weekly["Close"].rolling(10).mean().iloc[-1] if len(weekly) >= 10 else None
    w30 = weekly["Close"].rolling(30).mean().iloc[-1] if len(weekly) >= 30 else None

    checks = {
        "close_gt_sma50": close > (sma50 or math.inf),
        "close_gt_sma150": close > (sma150 or math.inf),
        "close_gt_sma200": close > (sma200 or math.inf),
        "sma50_gt_sma150": (sma50 or 0) > (sma150 or math.inf),
        "sma150_gt_sma200": (sma150 or 0) > (sma200 or math.inf),
        "sma200_up_20d": sma200 is not None and sma200_20 is not None and sma200 > sma200_20,
        "sma200_up_60d": sma200 is not None and sma200_60 is not None and sma200 > sma200_60,
        "above_52w_low_30pct": ll252 is not None and ll252 > 0 and close >= ll252 * 1.30,
        "within_25pct_52w_high": hh252 is not None and hh252 > 0 and close >= hh252 * 0.75,
        "w10_gt_w30": w10 is not None and w30 is not None and w10 > w30,
    }
    weights = {
        "close_gt_sma50": 25,
        "close_gt_sma150": 25,
        "close_gt_sma200": 25,
        "sma50_gt_sma150": 25,
        "sma150_gt_sma200": 25,
        "sma200_up_20d": 25,
        "sma200_up_60d": 20,
        "above_52w_low_30pct": 25,
        "within_25pct_52w_high": 25,
        "w10_gt_w30": 20,
    }
    score = sum(weights[k] for k, ok in checks.items() if ok)
    return int(score), {
        "checks": checks,
        "close": safe_round(close),
        "sma20": safe_round(r.get("SMA20")),
        "sma50": safe_round(sma50),
        "sma150": safe_round(sma150),
        "sma200": safe_round(sma200),
        "high_52w": safe_round(hh252),
        "low_52w": safe_round(ll252),
        "gap_to_52w_high_pct": safe_round(pct(close, hh252), 2) if hh252 else None,
    }


def score_breakout(d: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
    last = d.iloc[-1]
    close = to_float(last.get("Close")) or 0.0
    high = d["High"] if "High" in d else d["Close"]
    vol = d["Volume"]
    avg20 = d["VOL20"]

    is_20 = close >= (to_float(last.get("HH20_PREV")) or math.inf)
    is_55 = close >= (to_float(last.get("HH55_PREV")) or math.inf)

    breakout_pos = None
    breakout_type = None
    for i in range(len(d) - 1, max(-1, len(d) - 12), -1):
        row = d.iloc[i]
        c = to_float(row.get("Close")) or 0.0
        hh55 = to_float(row.get("HH55_PREV"))
        hh20 = to_float(row.get("HH20_PREV"))
        if hh55 is not None and c >= hh55:
            breakout_pos = i
            breakout_type = "55D High"
            break
        if hh20 is not None and c >= hh20 and breakout_pos is None:
            breakout_pos = i
            breakout_type = "20D High"
            break

    days_since = None
    breakout_price = None
    breakout_date = None
    breakout_volume_ratio = None
    maintains = False

    if breakout_pos is not None:
        bo = d.iloc[breakout_pos]
        days_since = len(d) - 1 - breakout_pos
        breakout_date = d.index[breakout_pos].date().isoformat()
        if breakout_type == "55D High":
            breakout_price = to_float(bo.get("HH55_PREV"))
        else:
            breakout_price = to_float(bo.get("HH20_PREV"))
        if breakout_price:
            maintains = close >= breakout_price
        avgv = to_float(bo.get("VOL20"))
        bov = to_float(bo.get("Volume"))
        if avgv and avgv > 0 and bov is not None:
            breakout_volume_ratio = bov / avgv

    score = 0
    if is_20:
        score += 20
    if is_55:
        score += 30
    if days_since is not None:
        if days_since <= 2:
            score += 35
        elif days_since <= 5:
            score += 30
        elif days_since <= 10:
            score += 12
    if breakout_volume_ratio is not None:
        if breakout_volume_ratio >= 2.0:
            score += 30
        elif breakout_volume_ratio >= 1.5:
            score += 24
        elif breakout_volume_ratio >= 1.2:
            score += 12
    from_bo_pct = pct(close, breakout_price) if breakout_price else None
    if from_bo_pct is not None:
        if 0 <= from_bo_pct <= 8:
            score += 25
        elif 8 < from_bo_pct <= 15:
            score += 12
        elif from_bo_pct > 25:
            score -= 15
    if maintains:
        score += 20

    return int(clamp(score, 0, 160)), {
        "breakout_type": breakout_type or ("55D High" if is_55 else "20D High" if is_20 else "None"),
        "breakout_date": breakout_date,
        "days_since_breakout": days_since,
        "breakout_price": safe_round(breakout_price),
        "price_from_breakout_pct": safe_round(from_bo_pct, 2),
        "breakout_volume_ratio": safe_round(breakout_volume_ratio, 2),
        "is_20d_high": bool(is_20),
        "is_55d_high": bool(is_55),
        "maintains_breakout": bool(maintains),
    }


def score_setup(d: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
    r = d.iloc[-1]
    close = to_float(r.get("Close")) or 0.0
    atr10 = to_float(r.get("ATR10_PCT"))
    atr20 = to_float(r.get("ATR20_PCT"))
    atr60 = to_float(r.get("ATR60_PCT"))
    vol10 = to_float(r.get("VOL10"))
    vol50 = to_float(r.get("VOL50"))
    sma20 = to_float(r.get("SMA20"))
    sma50 = to_float(r.get("SMA50"))

    range20 = (d["High"].tail(20).max() / d["Low"].tail(20).min() - 1) if len(d) >= 20 and d["Low"].tail(20).min() > 0 else None
    range60 = (d["High"].tail(60).max() / d["Low"].tail(60).min() - 1) if len(d) >= 60 and d["Low"].tail(60).min() > 0 else None

    score = 0
    if atr20 is not None and atr60 is not None and atr20 < atr60:
        score += 30
    if atr10 is not None and atr20 is not None and atr10 < atr20:
        score += 25
    if range20 is not None and range60 is not None and range20 < range60 * 0.75:
        score += 25
    if vol10 is not None and vol50 is not None and vol10 < vol50:
        score += 25
    if sma20 and abs(close / sma20 - 1) <= 0.08:
        score += 12
    if sma50 and abs(close / sma50 - 1) <= 0.12:
        score += 8
    if len(d) >= 30:
        # A simple proxy for smaller latest pullback versus previous pullback.
        c = d["Close"].tail(30)
        recent_dd = c.tail(10).max() / c.tail(10).min() - 1 if c.tail(10).min() > 0 else None
        prior_dd = c.head(20).max() / c.head(20).min() - 1 if c.head(20).min() > 0 else None
        if recent_dd is not None and prior_dd is not None and recent_dd < prior_dd:
            score += 25

    vcp_like = bool(
        atr10 is not None and atr20 is not None and atr60 is not None and atr10 < atr20 < atr60 and vol10 is not None and vol50 is not None and vol10 < vol50 and close > (sma50 or math.inf)
    )
    return int(clamp(score, 0, 150)), {
        "atr_10_pct": safe_round((atr10 or 0) * 100 if atr10 is not None else None, 2),
        "atr_20_pct": safe_round((atr20 or 0) * 100 if atr20 is not None else None, 2),
        "atr_60_pct": safe_round((atr60 or 0) * 100 if atr60 is not None else None, 2),
        "range_20d_pct": safe_round((range20 or 0) * 100 if range20 is not None else None, 2),
        "range_60d_pct": safe_round((range60 or 0) * 100 if range60 is not None else None, 2),
        "vcp_like": vcp_like,
    }


def score_volume(d: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
    last20 = d.tail(20).copy()
    last = d.iloc[-1]
    close = d["Close"]
    volume = d["Volume"]
    up_mask = last20["Close"] > last20["Close"].shift(1)
    total_vol = float(last20["Volume"].sum()) if len(last20) else 0.0
    up_vol_ratio = float(last20.loc[up_mask, "Volume"].sum()) / total_vol if total_vol > 0 else None
    vol50 = to_float(last.get("VOL50"))
    avg_dollar20 = to_float((d["DOLLAR_VOL"].tail(20)).mean())

    dist = 0
    avg20 = d["VOL20"]
    for i in range(max(1, len(d) - 20), len(d)):
        if close.iloc[i] < close.iloc[i - 1] * 0.98 and volume.iloc[i] > (avg20.iloc[i] or 0):
            dist += 1

    weekly = d.resample("W-FRI").agg({"Open": "first", "Close": "last", "Volume": "sum"}).dropna()
    weekly_up_vol = False
    if len(weekly) >= 2:
        weekly_up_vol = bool(weekly["Close"].iloc[-1] > weekly["Open"].iloc[-1] and weekly["Volume"].iloc[-1] > weekly["Volume"].iloc[-2])

    score = 0
    if up_vol_ratio is not None:
        if up_vol_ratio >= 0.62:
            score += 55
        elif up_vol_ratio >= 0.55:
            score += 42
        elif up_vol_ratio >= 0.50:
            score += 25
        else:
            score += 10
    if avg_dollar20 is not None:
        if avg_dollar20 >= 50_000_000:
            score += 20
        elif avg_dollar20 >= 20_000_000:
            score += 16
        elif avg_dollar20 >= 5_000_000:
            score += 8
    if dist <= 1:
        score += 20
    elif dist == 2:
        score += 10
    if weekly_up_vol:
        score += 10
    # Recent activity without becoming a daily-only anomaly.
    vr = to_float(last.get("VOL10")) / to_float(last.get("VOL50")) if to_float(last.get("VOL10")) and to_float(last.get("VOL50")) else None
    if vr is not None:
        if 0.8 <= vr <= 1.8:
            score += 25
        elif 1.8 < vr <= 2.5:
            score += 15
        elif vr < 0.8:
            score += 8

    return int(clamp(score, 0, 130)), {
        "up_volume_ratio": safe_round((up_vol_ratio or 0) * 100 if up_vol_ratio is not None else None, 1),
        "distribution_days_20d": dist,
        "avg_dollar_volume_20d": safe_round(avg_dollar20, 0),
        "vol10_to_vol50": safe_round(vr, 2),
        "weekly_up_volume": weekly_up_vol,
    }


def get_info(symbol: str) -> Dict[str, Any]:
    if not ENABLE_FUNDAMENTALS or yf is None:
        return {}
    try:
        return dict(yf.Ticker(symbol).get_info() or {})
    except Exception as e:
        log("WARN", f"fundamental fetch failed: {symbol}: {e}")
        return {}


def score_fundamental(info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    if not info:
        return 50, {"confidence": "Low", "available_fields": 0, "revenue_growth": None, "earnings_growth": None, "profit_margin": None, "gross_margin": None}
    fields = {
        "revenue_growth": to_float(info.get("revenueGrowth")),
        "earnings_growth": to_float(info.get("earningsGrowth")),
        "profit_margin": to_float(info.get("profitMargins")),
        "gross_margin": to_float(info.get("grossMargins")),
        "operating_margin": to_float(info.get("operatingMargins")),
        "return_on_equity": to_float(info.get("returnOnEquity")),
    }
    avail = sum(v is not None for v in fields.values())
    score = 0
    max_score = 0
    if fields["revenue_growth"] is not None:
        max_score += 30
        rg = fields["revenue_growth"]
        score += 30 if rg >= 0.30 else 22 if rg >= 0.20 else 14 if rg >= 0.10 else 5 if rg >= 0 else 0
    if fields["earnings_growth"] is not None:
        max_score += 25
        eg = fields["earnings_growth"]
        score += 25 if eg >= 0.30 else 18 if eg >= 0.20 else 10 if eg >= 0 else 0
    if fields["gross_margin"] is not None:
        max_score += 15
        gm = fields["gross_margin"]
        score += 15 if gm >= 0.60 else 10 if gm >= 0.40 else 6 if gm >= 0.25 else 2
    if fields["operating_margin"] is not None:
        max_score += 15
        om = fields["operating_margin"]
        score += 15 if om >= 0.20 else 10 if om >= 0.10 else 5 if om >= 0 else 0
    if fields["return_on_equity"] is not None:
        max_score += 15
        roe = fields["return_on_equity"]
        score += 15 if roe >= 0.20 else 10 if roe >= 0.10 else 5 if roe >= 0 else 0
    normalized = round(score / max_score * 100) if max_score > 0 else 50
    confidence = "High" if avail >= 4 else "Medium" if avail >= 2 else "Low"
    return int(clamp(normalized, 0, 100)), {
        "confidence": confidence,
        "available_fields": avail,
        **{k: safe_round(v * 100, 2) if v is not None else None for k, v in fields.items()},
    }


def score_risk(d: pd.DataFrame, breakout: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    r = d.iloc[-1]
    close = to_float(r.get("Close")) or 0.0
    sma20 = to_float(r.get("SMA20"))
    avg_dollar20 = to_float(d["DOLLAR_VOL"].tail(20).mean())
    atr20 = to_float(r.get("ATR20_PCT"))
    from_bo = to_float(breakout.get("price_from_breakout_pct"))
    ret_1m = pct(close, d["Close"].iloc[-21]) if len(d) >= 21 else None
    sma20_gap = pct(close, sma20) if sma20 else None

    score = 0
    if avg_dollar20 is not None:
        score += 15 if avg_dollar20 >= 20_000_000 else 8 if avg_dollar20 >= 5_000_000 else 0
    if close >= 5:
        score += 10
    elif close >= 3:
        score += 5
    if sma20_gap is not None and sma20_gap <= 15:
        score += 15
    elif sma20_gap is not None and sma20_gap <= 25:
        score += 6
    if atr20 is not None:
        score += 10 if atr20 <= 0.08 else 5 if atr20 <= 0.12 else 0
    score += 10  # earnings calendar is not available in stable form; keep neutral-positive.

    extended = bool(
        (sma20_gap is not None and sma20_gap > 15)
        or (from_bo is not None and from_bo > 15)
        or (ret_1m is not None and ret_1m > 40)
    )
    hard_exclusion = bool(close < 3 or (avg_dollar20 is not None and avg_dollar20 < 5_000_000) or close < (to_float(r.get("SMA200")) or math.inf) or (to_float(r.get("HH252")) and close < (to_float(r.get("HH252")) or 0) * 0.60))
    return int(clamp(score, 0, 60)), {
        "extended": extended,
        "hard_exclusion": hard_exclusion,
        "sma20_gap_pct": safe_round(sma20_gap, 2),
        "return_1m_pct": safe_round(ret_1m, 2),
    }


def calc_returns(d: pd.DataFrame) -> Dict[str, Optional[float]]:
    close = d["Close"]
    cur = close.iloc[-1]
    return {
        "return_1w": safe_round(pct(cur, close.iloc[-6]) if len(close) >= 6 else None, 2),
        "return_1m": safe_round(pct(cur, close.iloc[-21]) if len(close) >= 21 else None, 2),
        "return_3m": safe_round(pct(cur, close.iloc[-64]) if len(close) >= 64 else None, 2),
        "return_6m": safe_round(pct(cur, close.iloc[-127]) if len(close) >= 127 else None, 2),
        "return_12m": safe_round(pct(cur, close.iloc[-253]) if len(close) >= 253 else None, 2),
    }


def rs_composite(returns: Dict[str, Optional[float]]) -> Optional[float]:
    vals = []
    weights = [("return_1m", 0.20), ("return_3m", 0.35), ("return_6m", 0.35), ("return_12m", 0.10)]
    for k, w in weights:
        v = returns.get(k)
        if v is not None:
            vals.append((v, w))
    if not vals:
        return None
    total_w = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / total_w


def percentile_scores(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    valid = {k: v for k, v in values.items() if v is not None and math.isfinite(v)}
    if not valid:
        return {k: 0.0 for k in values}
    sorted_vals = sorted(valid.items(), key=lambda kv: kv[1])
    n = len(sorted_vals)
    out = {k: 0.0 for k in values}
    for i, (k, _v) in enumerate(sorted_vals):
        out[k] = 100.0 if n == 1 else i / (n - 1) * 100.0
    return out


def map_rs_score(percentile: float) -> int:
    if percentile >= 95:
        return 180
    if percentile >= 90:
        return 165
    if percentile >= 85:
        return 150
    if percentile >= 80:
        return 135
    if percentile >= 70:
        return 110
    if percentile >= 60:
        return 80
    return int(clamp(40 + percentile * 0.5, 0, 75))


def make_signal(total: int, trend: int, rs: int, breakout: int, setup: int, volume: int, risk: Dict[str, Any]) -> str:
    if risk.get("hard_exclusion") or trend < 140:
        return "E Avoid"
    if risk.get("extended") and total >= 700:
        return "D Extended"
    if total >= 820 and trend >= 185 and rs >= 145 and breakout >= 120 and volume >= 90:
        return "A+ Fresh Breakout"
    if total >= 780 and trend >= 185 and rs >= 150:
        return "A Leader"
    if total >= 700 and trend >= 165 and setup >= 110:
        return "B Constructive Setup"
    if total >= 620 and trend >= 140 and rs >= 100:
        return "C Early Watch"
    return "C Early Watch" if total >= 560 else "E Avoid"


def make_comment(item: Dict[str, Any]) -> str:
    sig = item["signal"]
    parts: List[str] = []
    if sig == "A+ Fresh Breakout":
        parts.append("Stage 2条件とRSが強く、出来高を伴う初週ブレイク候補。")
    elif sig == "A Leader":
        parts.append("初動ではないが、Trend Templateと相対強度が高いリーダー株候補。")
    elif sig == "B Constructive Setup":
        parts.append("ブレイク前後の形が良く、VCP的な収縮・セットアップを監視。")
    elif sig == "D Extended":
        parts.append("銘柄は強いが、SMA20またはブレイク価格からの乖離が大きく追いかけ注意。")
    elif sig == "E Avoid":
        parts.append("ミネルヴィニ型の中期候補としてはトレンド・流動性・位置のいずれかが不足。")
    else:
        parts.append("形はでき始めているが、主力候補には追加確認が必要。")

    bo = item.get("breakout") or {}
    if bo.get("days_since_breakout") is not None:
        parts.append(f"ブレイクから{bo.get('days_since_breakout')}営業日、BO乖離は{bo.get('price_from_breakout_pct')}%。")
    if item.get("setup", {}).get("vcp_like"):
        parts.append("ATRと出来高が収縮しており、VCP-like。")
    return "".join(parts)


def main() -> None:
    date = resolve_report_date()
    candidates = [c for c in load_candidates() if not c.exclude]
    if not candidates:
        raise SystemExit("No weekly candidates found. Add rows to data/weekly_candidates.csv")

    log("INFO", f"Weekly screening date={date}, candidates={len(candidates)}, benchmark={BENCHMARK}")

    raw_items: List[Dict[str, Any]] = []
    rs_values: Dict[str, Optional[float]] = {}

    for c in candidates:
        df = fetch_history(c.symbol)
        if df.empty or len(df) < 210:
            log("WARN", f"skip {c.symbol}: insufficient history rows={len(df)}")
            continue
        d = add_indicators(df).dropna(subset=["Close", "Volume"])
        if len(d) < 210:
            log("WARN", f"skip {c.symbol}: insufficient indicator rows={len(d)}")
            continue

        trend_score, trend = score_trend(d)
        breakout_score, breakout = score_breakout(d)
        setup_score, setup = score_setup(d)
        volume_score, volume = score_volume(d)
        fund_score, fund = score_fundamental(get_info(c.symbol))
        risk_score, risk = score_risk(d, breakout)
        returns = calc_returns(d)
        rs_comp = rs_composite(returns)
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
        raise SystemExit("No valid weekly items after history fetch.")

    percentiles = percentile_scores(rs_values)
    for item in raw_items:
        sym = item["symbol"]
        rs_pct = percentiles.get(sym, 0.0)
        rs_score = map_rs_score(rs_pct)
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
        item["signal"] = make_signal(total, item["trend_score"], item["rs_score"], item["breakout_score"], item["setup_score"], item["volume_score"], item["risk"])
        item["comment"] = make_comment(item)

    signal_order = {"A+ Fresh Breakout": 0, "A Leader": 1, "B Constructive Setup": 2, "C Early Watch": 3, "D Extended": 4, "E Avoid": 5}
    items = sorted(raw_items, key=lambda x: (signal_order.get(x["signal"], 9), -x["weekly_score"], -x.get("rs_percentile", 0)))
    for i, item in enumerate(items, 1):
        item["rank"] = i

    payload = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "methodology": {
            "name": "Minervini-inspired Weekly Screening",
            "total_points": 1000,
            "weights": {
                "trend_template_stage2": 220,
                "relative_strength": 180,
                "breakout_freshness": 160,
                "vcp_setup_quality": 150,
                "volume_accumulation": 130,
                "fundamental_momentum": 100,
                "risk_extension_liquidity": 60,
            },
            "benchmark": BENCHMARK,
            "fundamentals_enabled": ENABLE_FUNDAMENTALS,
        },
        "summary": {
            "total_candidates": len(candidates),
            "valid_items": len(items),
            "fresh_breakouts": sum(1 for x in items if x["signal"] == "A+ Fresh Breakout"),
            "leaders": sum(1 for x in items if x["signal"] == "A Leader"),
            "constructive_setups": sum(1 for x in items if x["signal"] == "B Constructive Setup"),
        },
        "items": items,
    }

    date_path = OUT_DIR / "data" / "weekly" / f"{date}.json"
    latest_path = OUT_DIR / "data" / "weekly" / "latest.json"
    write_json(date_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote {date_path}")
    log("INFO", f"Wrote {latest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in build_weekly_screening: {e}")
        sys.exit(1)
