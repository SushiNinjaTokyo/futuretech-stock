#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", str(ROOT / "data" / "universe.csv")))
REPORT_DATE = os.getenv("REPORT_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d")).strip()
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").strip().lower()
SLEEP_SECONDS = float(os.getenv("DAILY_V2_SLEEP_SECONDS", "0.8"))
MAX_SYMBOLS = int(os.getenv("DAILY_V2_MAX_SYMBOLS", "0") or "0")
MATURE_SYMBOLS = {
    s.strip().upper()
    for s in os.getenv(
        "DAILY_V2_MATURE_SYMBOLS",
        "AAPL,MSFT,NVDA,META,GOOGL,GOOG,AMZN,AVGO,TSLA,BRK-B,LLY,JPM,V,MA,UNH,XOM,COST,NFLX,ORCL,CRM,ADBE,AMD,INTC,CSCO",
    ).split(",")
    if s.strip()
}

INDEX_SYMBOLS = {"spy": "SPY", "qqq": "QQQ"}


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
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
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    v = to_float(x)
    if v is None:
        return lo
    return max(lo, min(hi, v))


def safe_round(x: Any, n: int = 2) -> Optional[float]:
    v = to_float(x)
    return round(v, n) if v is not None else None


def pct(cur: Any, prev: Any) -> Optional[float]:
    c = to_float(cur)
    p = to_float(prev)
    if c is None or p is None or p == 0:
        return None
    return (c / p - 1.0) * 100.0


def load_universe(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Universe CSV not found: {path}")
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    sym_col = cols.get("symbol", df.columns[0])
    name_col = cols.get("name")
    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip().upper()
        if not sym:
            continue
        out.append({"symbol": sym, "name": str(row.get(name_col, "")).strip() if name_col else ""})
    if MAX_SYMBOLS > 0:
        out = out[:MAX_SYMBOLS]
    return out


def normalize_ohlcv(raw: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if symbol and symbol in df.columns.get_level_values(0):
            df = df[symbol]
        elif symbol and symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, level=-1, axis=1)
        else:
            # yfinance sometimes returns field first, ticker second; select first ticker.
            if "Close" in df.columns.get_level_values(0):
                tickers = [x for x in df.columns.get_level_values(1).unique()]
                if tickers:
                    df = df.xs(tickers[0], level=1, axis=1)
            else:
                df = df.droplevel(0, axis=1)
    cols = {str(c).lower(): c for c in df.columns}
    out = pd.DataFrame(index=pd.to_datetime(df.index))
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        src = cols.get(c.lower())
        out[c] = pd.to_numeric(df[src], errors="coerce") if src is not None else np.nan
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0)
    return out


def fetch_history(symbol: str, as_of: str, lookback_days: int = 560) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    end_dt = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(days=lookback_days)
    for attempt in range(3):
        try:
            raw = yf.download(
                symbol,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = normalize_ohlcv(raw, symbol)
            if len(df) >= 60:
                return df
            raise ValueError(f"insufficient rows: {len(df)}")
        except Exception as e:
            log("WARN", f"{symbol}: fetch failed attempt={attempt+1}: {e}")
            time.sleep(1.5 + attempt * 1.0)
    return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_index()
    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    v = d["Volume"]
    d["sma10"] = c.rolling(10).mean()
    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma150"] = c.rolling(150).mean()
    d["high5"] = h.rolling(5).max()
    d["high10"] = h.rolling(10).max()
    d["high20"] = h.rolling(20).max()
    d["high50"] = h.rolling(50).max()
    d["low10"] = l.rolling(10).min()
    d["low20"] = l.rolling(20).min()
    d["vol20"] = v.rolling(20).mean()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr14"] / c * 100.0
    d["ret1"] = c.pct_change(1) * 100.0
    d["ret5"] = c.pct_change(5) * 100.0
    d["ret20"] = c.pct_change(20) * 100.0
    d["ret60"] = c.pct_change(60) * 100.0
    d["rvol20"] = v / (d["vol20"] + 1e-9)
    d["range_pos"] = (c - l) / ((h - l).replace(0, np.nan))
    d["close_pos20"] = (c - d["low20"]) / ((d["high20"] - d["low20"]).replace(0, np.nan))
    d["bb_width20"] = (c.rolling(20).std() * 4.0) / c * 100.0
    d["bb_width_pct63"] = d["bb_width20"].rolling(63).rank(pct=True)
    d["atr_pct_rank63"] = d["atr_pct"].rolling(63).rank(pct=True)
    d["extension_sma20"] = (c / d["sma20"] - 1.0) * 100.0
    d["extension_sma50"] = (c / d["sma50"] - 1.0) * 100.0
    d["dollar_volume"] = c * v
    d["avg_dollar_volume20"] = d["dollar_volume"].rolling(20).mean()
    return d


def extract_component_map(kind: str, report_date: str) -> Dict[str, float]:
    candidates = [
        OUT_DIR / "data" / report_date / f"{kind}.json",
        OUT_DIR / "data" / kind / "latest.json",
    ]
    out: Dict[str, float] = {}
    for path in candidates:
        j = read_json(path)
        if not j:
            continue
        items = j.get("items", j) if isinstance(j, dict) else j
        if not isinstance(items, list):
            continue
        for row in items:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol", "")).upper()
            if not sym:
                continue
            raw = row.get("score_0_1", row.get("score01", row.get("score", row.get("value"))))
            out[sym] = clamp(raw)
        if out:
            break
    return out


def score_symbol(symbol: str, name: str, df: pd.DataFrame, index_map: Dict[str, pd.DataFrame], maps: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    d = add_indicators(df)
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) >= 2 else last
    close = float(last["Close"])
    open_ = float(last["Open"])
    high = float(last["High"])
    low = float(last["Low"])
    volume = float(last["Volume"])

    range_pos = clamp(last.get("range_pos"), 0, 1)
    rvol20 = max(0.0, to_float(last.get("rvol20")) or 0.0)
    dollar_vol = max(0.0, to_float(last.get("dollar_volume")) or 0.0)
    avg_dollar_vol20 = max(0.0, to_float(last.get("avg_dollar_volume20")) or 0.0)

    price_up = 1.0 if close > float(prev.get("Close", close)) else 0.0
    dollar_liq_score = clamp(math.log10(max(avg_dollar_vol20, 1.0)) / 9.0)
    volume_shock = clamp(0.45 * (1 - math.exp(-rvol20 / 2.2)) + 0.25 * range_pos + 0.20 * price_up + 0.10 * dollar_liq_score)

    bb_rank = clamp(1.0 - (to_float(last.get("bb_width_pct63")) or 0.5))
    atr_rank = clamp(1.0 - (to_float(last.get("atr_pct_rank63")) or 0.5))
    comp_external = clamp(maps.get("compression", {}).get(symbol, 0.0))
    breakout_today = 1.0 if close >= (to_float(prev.get("high20")) or high) else 0.0
    compression_release = clamp(0.35 * bb_rank + 0.25 * atr_rank + 0.25 * comp_external + 0.15 * breakout_today)

    above20 = 1.0 if close >= (to_float(last.get("sma20")) or close * 10) else 0.0
    above50 = 1.0 if close >= (to_float(last.get("sma50")) or close * 10) else 0.0
    high20_break = 1.0 if close >= (to_float(prev.get("high20")) or high) else 0.0
    high50_break = 1.0 if close >= (to_float(prev.get("high50")) or high) else 0.0
    close_pos20 = clamp(last.get("close_pos20"), 0, 1)
    trend_external = clamp(maps.get("trends", {}).get(symbol, 0.0))
    setup_quality = clamp(0.20 * above20 + 0.20 * above50 + 0.20 * high20_break + 0.15 * high50_break + 0.15 * close_pos20 + 0.10 * trend_external)

    spy = add_indicators(index_map.get("spy", pd.DataFrame())) if not index_map.get("spy", pd.DataFrame()).empty else pd.DataFrame()
    qqq = add_indicators(index_map.get("qqq", pd.DataFrame())) if not index_map.get("qqq", pd.DataFrame()).empty else pd.DataFrame()
    spy_last = spy.iloc[-1] if not spy.empty else {}
    qqq_last = qqq.iloc[-1] if not qqq.empty else {}
    ret5 = to_float(last.get("ret5")) or 0.0
    ret20 = to_float(last.get("ret20")) or 0.0
    ret60 = to_float(last.get("ret60")) or 0.0
    qret5 = to_float(qqq_last.get("ret5")) or 0.0
    qret20 = to_float(qqq_last.get("ret20")) or 0.0
    qret60 = to_float(qqq_last.get("ret60")) or 0.0
    sret20 = to_float(spy_last.get("ret20")) or 0.0
    rs5 = ret5 - qret5
    rs20 = ret20 - qret20
    rs60 = ret60 - qret60
    relative_strength = clamp(0.30 * clamp((rs5 + 5) / 15) + 0.40 * clamp((rs20 + 8) / 24) + 0.20 * clamp((rs60 + 10) / 40) + 0.10 * clamp((ret20 - sret20 + 8) / 24))

    ext20 = to_float(last.get("extension_sma20")) or 0.0
    ext50 = to_float(last.get("extension_sma50")) or 0.0
    ret1 = to_float(last.get("ret1")) or 0.0
    atr_pct = to_float(last.get("atr_pct")) or 0.0
    not_extended = 1.0 - clamp(max(ext20 - 8.0, 0) / 18.0 + max(ret1 - 10.0, 0) / 16.0 + max((to_float(last.get("ret5")) or 0) - 22.0, 0) / 30.0)
    stop_distance = min(8.0, max(3.0, atr_pct * 1.6))
    rr_quality = clamp((12.0 - stop_distance) / 10.0)
    entry_timing = clamp(0.45 * not_extended + 0.25 * range_pos + 0.15 * above20 + 0.15 * rr_quality)

    news_external = clamp(maps.get("news", {}).get(symbol, 0.0))
    catalyst = clamp(0.25 * news_external + 0.25 * news_external * volume_shock + 0.25 * news_external * setup_quality + 0.25 * max(trend_external, comp_external))

    spy_above20 = 1.0 if spy_last is not None and to_float(spy_last.get("Close")) and to_float(spy_last.get("sma20")) and float(spy_last.get("Close")) >= float(spy_last.get("sma20")) else 0.0
    qqq_above20 = 1.0 if qqq_last is not None and to_float(qqq_last.get("Close")) and to_float(qqq_last.get("sma20")) and float(qqq_last.get("Close")) >= float(qqq_last.get("sma20")) else 0.0
    spy_above50 = 1.0 if spy_last is not None and to_float(spy_last.get("Close")) and to_float(spy_last.get("sma50")) and float(spy_last.get("Close")) >= float(spy_last.get("sma50")) else 0.0
    qqq_above50 = 1.0 if qqq_last is not None and to_float(qqq_last.get("Close")) and to_float(qqq_last.get("sma50")) and float(qqq_last.get("Close")) >= float(qqq_last.get("sma50")) else 0.0
    if spy_above20 and qqq_above20:
        regime = "Risk-on"
        regime_score = 1.0
    elif spy_above50 and qqq_above50:
        regime = "Neutral"
        regime_score = 0.6
    else:
        regime = "Risk-off"
        regime_score = 0.15

    mature_penalty = 0.0
    if symbol in MATURE_SYMBOLS:
        mature_penalty += 0.09
    if ret60 > 80:
        mature_penalty += 0.07
    extension_penalty = clamp(max(ret1 - 15, 0) / 35 + max(ext20 - 18, 0) / 35 + max(ext50 - 35, 0) / 45)
    weak_close_penalty = 0.0
    if range_pos < 0.45 and rvol20 >= 1.4:
        weak_close_penalty = 0.10
    if close < open_ and rvol20 >= 1.5:
        weak_close_penalty += 0.08
    news_hype_penalty = 0.0
    if news_external >= 0.6 and setup_quality < 0.45 and compression_release < 0.45:
        news_hype_penalty = 0.10
    liquidity_penalty = 0.08 if avg_dollar_vol20 < 1_000_000 else 0.0
    penalty = clamp(mature_penalty + 0.55 * extension_penalty + weak_close_penalty + news_hype_penalty + liquidity_penalty, 0, 0.35)

    raw = (
        0.22 * volume_shock +
        0.20 * compression_release +
        0.18 * setup_quality +
        0.15 * relative_strength +
        0.15 * entry_timing +
        0.07 * catalyst +
        0.03 * regime_score
    )
    final01 = clamp(raw - penalty)
    score_pts = int(round(final01 * 1000))

    if score_pts >= 750 and volume_shock >= 0.55 and (compression_release >= 0.45 or setup_quality >= 0.55) and penalty < 0.18 and regime != "Risk-off":
        triage = "Trade"
    elif score_pts >= 650 and volume_shock >= 0.45:
        triage = "Watch"
    else:
        triage = "Ignore"

    archetype_parts: List[str] = []
    if volume_shock >= 0.62:
        archetype_parts.append("Volume")
    if compression_release >= 0.55:
        archetype_parts.append("Compression")
    if setup_quality >= 0.62:
        archetype_parts.append("Breakout")
    if catalyst >= 0.55:
        archetype_parts.append("Catalyst")
    if not archetype_parts:
        archetype_parts.append("Mixed")
    archetype = " + ".join(archetype_parts[:3])

    reason = []
    if volume_shock >= 0.62:
        reason.append(f"RVOL {rvol20:.2f}x")
    if compression_release >= 0.55:
        reason.append("compression release")
    if setup_quality >= 0.62:
        reason.append("breakout setup")
    if relative_strength >= 0.65:
        reason.append("relative strength")
    if penalty >= 0.18:
        reason.append("penalty: extended/mature")
    if not reason:
        reason.append("mixed signal")

    return {
        "symbol": symbol,
        "name": name,
        "as_of": REPORT_DATE,
        "final_score_0_1": round(final01, 6),
        "score_pts": score_pts,
        "triage": triage,
        "archetype": archetype,
        "regime": regime,
        "reason": "; ".join(reason),
        "price": round(close, 2),
        "volume": int(volume),
        "score_components": {
            "volume_anomaly": round(volume_shock, 6),
            "compression_release": round(compression_release, 6),
            "trends_breakout": round(setup_quality, 6),
            "news": round(catalyst, 6),
        },
        "score_weights": {
            "volume_anomaly": 0.22,
            "compression_release": 0.20,
            "trends_breakout": 0.18,
            "relative_strength": 0.15,
            "entry_timing": 0.15,
            "news": 0.07,
            "market_regime": 0.03,
        },
        "v2_components": {
            "volume_liquidity_shock": round(volume_shock, 6),
            "compression_release": round(compression_release, 6),
            "breakout_setup_quality": round(setup_quality, 6),
            "relative_strength": round(relative_strength, 6),
            "entry_timing": round(entry_timing, 6),
            "catalyst_confirmation": round(catalyst, 6),
            "market_regime": round(regime_score, 6),
            "penalty": round(penalty, 6),
        },
        "metrics": {
            "ret_1d_pct": safe_round(ret1, 2),
            "ret_5d_pct": safe_round(ret5, 2),
            "ret_20d_pct": safe_round(ret20, 2),
            "ret_60d_pct": safe_round(ret60, 2),
            "rvol20": safe_round(rvol20, 2),
            "range_pos": safe_round(range_pos, 3),
            "close_pos20": safe_round(close_pos20, 3),
            "atr_pct": safe_round(atr_pct, 2),
            "extension_sma20_pct": safe_round(ext20, 2),
            "extension_sma50_pct": safe_round(ext50, 2),
            "dollar_volume": safe_round(dollar_vol, 0),
            "avg_dollar_volume20": safe_round(avg_dollar_vol20, 0),
            "rs_5d_vs_qqq": safe_round(rs5, 2),
            "rs_20d_vs_qqq": safe_round(rs20, 2),
            "rs_60d_vs_qqq": safe_round(rs60, 2),
            "mature_penalty": safe_round(mature_penalty * 100, 2),
            "extension_penalty": safe_round(extension_penalty * 100, 2),
            "weak_close_penalty": safe_round(weak_close_penalty * 100, 2),
            "news_hype_penalty": safe_round(news_hype_penalty * 100, 2),
        },
        "legacy": {
            "score_pts": score_pts,
            "volume_anomaly": round(volume_shock, 6),
        },
    }


def build_market_snapshot(index_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    out = {}
    for key, sym in INDEX_SYMBOLS.items():
        df = index_map.get(key, pd.DataFrame())
        if df.empty:
            out[key] = {"symbol": sym, "regime": "Unknown"}
            continue
        d = add_indicators(df)
        last = d.iloc[-1]
        c = to_float(last.get("Close"))
        sma20 = to_float(last.get("sma20"))
        sma50 = to_float(last.get("sma50"))
        regime = "Risk-on" if c and sma20 and c >= sma20 else ("Neutral" if c and sma50 and c >= sma50 else "Risk-off")
        out[key] = {
            "symbol": sym,
            "close": safe_round(c, 2),
            "ret_1d_pct": safe_round(last.get("ret1"), 2),
            "ret_5d_pct": safe_round(last.get("ret5"), 2),
            "ret_20d_pct": safe_round(last.get("ret20"), 2),
            "above_sma20": bool(c and sma20 and c >= sma20),
            "above_sma50": bool(c and sma50 and c >= sma50),
            "regime": regime,
        }
    return out


def update_manifest(date: str, payload: Dict[str, Any]) -> None:
    path = OUT_DIR / "data" / "daily-v2" / "manifest.json"
    manifest = read_json(path) or {"version": "daily_event_score_v2", "dates": {}}
    manifest.setdefault("dates", {})[date] = {
        "status": "ok",
        "generated_at": payload.get("generated_at"),
        "items": len(payload.get("items", [])),
        "top10": len(payload.get("items", [])[:10]),
        "trade": sum(1 for x in payload.get("items", []) if x.get("triage") == "Trade"),
        "watch": sum(1 for x in payload.get("items", []) if x.get("triage") == "Watch"),
    }
    write_json(path, manifest)


def main() -> None:
    if DATA_PROVIDER != "yfinance":
        log("WARN", f"Daily v2 currently uses yfinance path. DATA_PROVIDER={DATA_PROVIDER}")
    universe = load_universe(UNIVERSE_CSV)
    maps = {
        "compression": extract_component_map("compression", REPORT_DATE),
        "trends": extract_component_map("trends", REPORT_DATE),
        "news": extract_component_map("news", REPORT_DATE),
    }
    index_map = {k: fetch_history(sym, REPORT_DATE) for k, sym in INDEX_SYMBOLS.items()}
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    for idx, u in enumerate(universe, 1):
        sym = u["symbol"]
        log("INFO", f"Daily v2 scoring {idx}/{len(universe)} {sym} as_of={REPORT_DATE}")
        df = fetch_history(sym, REPORT_DATE)
        if df.empty:
            failed.append(sym)
            continue
        try:
            rows.append(score_symbol(sym, u.get("name", ""), df, index_map, maps))
        except Exception as e:
            log("WARN", f"{sym}: scoring failed: {e}")
            failed.append(sym)
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
    rows.sort(key=lambda r: (r.get("triage") == "Trade", r.get("score_pts", 0), r.get("v2_components", {}).get("entry_timing", 0)), reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    payload = {
        "version": "daily_event_score_v2",
        "date": REPORT_DATE,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "methodology": {
            "name": "Daily Event Score",
            "objective": "Rank timing-sensitive event candidates by volume, compression release, setup quality, relative strength, entry risk, and penalty controls.",
            "ui_version_label": "hidden",
            "weights": {
                "volume_liquidity_shock": 220,
                "compression_release": 200,
                "breakout_setup_quality": 180,
                "relative_strength": 150,
                "entry_timing": 150,
                "catalyst_confirmation": 70,
                "market_regime_alignment": 30,
                "penalties_max": -250,
            },
        },
        "market": build_market_snapshot(index_map),
        "summary": {
            "universe": len(universe),
            "scored": len(rows),
            "failed": len(failed),
            "trade": sum(1 for x in rows if x.get("triage") == "Trade"),
            "watch": sum(1 for x in rows if x.get("triage") == "Watch"),
            "ignore": sum(1 for x in rows if x.get("triage") == "Ignore"),
        },
        "items": rows[:25],
        "all_items": rows,
        "failed": failed,
    }
    day_dir = OUT_DIR / "data" / "daily-v2" / REPORT_DATE
    write_json(day_dir / "top10.json", payload)
    write_json(OUT_DIR / "data" / "daily-v2" / "latest.json", payload)
    update_manifest(REPORT_DATE, payload)
    log("INFO", f"Wrote {day_dir / 'top10.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in build_daily_v2: {e}")
        sys.exit(1)
