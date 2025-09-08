#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily aggregation script for futuretech-stock.

- Reads a universe CSV (env: UNIVERSE_CSV, default: data/universe.csv).
- Pulls price/volume (yfinance) and computes:
  - price_delta_1d / 1w / 1m (percentage, trading-day offsets 1 / 5 / 20)
  - simple volume anomaly features (rvol20, z60, pct_rank_90, dollar_vol)
  - volume_anomaly_score (0..1)
- Loads auxiliary component scores saved by other scripts:
  - site/data/dii/latest.json        -> insider_momo (0..1)
  - site/data/trends/latest.json     -> trends_breakout (0..1)
  - site/data/news/latest.json       -> news (0..1)
  The loader is schema-tolerant; if the files/keys are absent, zeros are used.
- Produces per-symbol records with canonical keys to match the front-end:
  - score_components: {volume_anomaly, insider_momo, trends_breakout, news}
  - score_weights:    same keys; taken from env and normalized in the client
  - final_score_0_1 and score_pts (0..1000)
  - detail.vol_anomaly with the raw metrics
- Writes:
  - {OUT_DIR}/data/{YYYY-MM-DD}/top10.json
  - {OUT_DIR}/data/top10/latest.json (symlink-like copy)

Environment:
  OUT_DIR (default: site)
  UNIVERSE_CSV (default: data/universe.csv)
  REPORT_DATE (YYYY-MM-DD; if missing, uses today in UTC)
  DATA_PROVIDER (yfinance|tiingo; yfinance default)
  TIINGO_TOKEN (required if DATA_PROVIDER=tiingo)
  MOCK_MODE (true|false; if true, skips network and emits zeros)

  WEIGHT_VOL_ANOM (default 0.25)
  WEIGHT_DII      (default 0.25) -> mapped to 'insider_momo'
  WEIGHT_TRENDS   (default 0.30)
  WEIGHT_NEWS     (default 0.20)
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# ---------- Small logging helpers ----------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def log_info(msg: str) -> None:
    print(f"{_ts()} [INFO] {msg}", flush=True)

def log_warn(msg: str) -> None:
    print(f"{_ts()} [WARN] {msg}", flush=True)

def log_error(msg: str) -> None:
    print(f"{_ts()} [ERROR] {msg}", file=sys.stderr, flush=True)

# ---------- Env helpers ----------
def env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)

def env_b(name: str, default: bool=False) -> bool:
    v = os.environ.get(name)
    if v is None: 
        return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def env_s(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()

# ---------- IO ----------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_warn(f"Failed to read JSON {path}: {e}")
        return None

# ---------- Universe ----------
def load_universe(csv_path: str) -> List[Dict[str, str]]:
    cols = ["symbol", "name"]
    if not os.path.exists(csv_path):
        log_warn(f"Universe CSV missing: {csv_path} (using empty universe)")
        return []
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        symbol_col = next((c for c in df.columns if c.startswith("symbol")), "symbol")
        name_col   = next((c for c in df.columns if c.startswith("name")), "name")
        out = []
        for _, row in df.iterrows():
            sym = str(row.get(symbol_col, "")).strip().upper()
            nm  = str(row.get(name_col, "")).strip()
            if sym:
                out.append({"symbol": sym, "name": nm})
        return out
    except Exception as e:
        log_error(f"Failed to load universe CSV: {e}")
        return []

# ---------- Market data ----------
def fetch_history_yf(symbol: str, months: int=12) -> Optional[pd.DataFrame]:
    if yf is None:
        log_warn("yfinance is not available")
        return None
    try:
        # Using period instead of start/end to avoid tz issues; 12 months ~ 252 trading days
        df = yf.Ticker(symbol).history(period=f"{months}mo", interval="1d", auto_adjust=False)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df = df.rename(columns=str.title)  # ensure 'Close','Volume' case
            df.index = pd.to_datetime(df.index)
            df = df[["Close","Volume"]].dropna()
            return df
        return None
    except Exception as e:
        log_warn(f"[YF] history failed for {symbol}: {e}")
        return None

def fetch_history_tiingo(symbol: str, token: str, months: int=12) -> Optional[pd.DataFrame]:
    if pdr is None:
        log_warn("pandas_datareader not available; cannot use Tiingo")
        return None
    try:
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.offsets.Day(int(30.5*months))
        df = pdr.get_data_tiingo(symbol, api_key=token, start=start.date(), end=end.date())
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)  # drop 'symbol' level
            cols_map = {
                "close": "Close",
                "volume": "Volume",
            }
            for a,b in cols_map.items():
                if a in df.columns and b not in df.columns: df[b]=df[a]
            df = df[["Close","Volume"]].dropna()
            return df
        return None
    except Exception as e:
        log_warn(f"[Tiingo] history failed for {symbol}: {e}")
        return None

# ---------- Features ----------
def pct_change(series: pd.Series, lag: int) -> Optional[float]:
    if series is None or len(series) <= lag:
        return None
    try:
        cur = float(series.iloc[-1])
        prev = float(series.iloc[-1 - lag])
        if prev == 0 or np.isnan(prev) or np.isnan(cur):
            return None
        return (cur/prev - 1.0) * 100.0
    except Exception:
        return None

def nan_to_none(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        f = float(x)
        if math.isnan(f): return None
        return f
    except Exception:
        return None

def compute_vol_anomaly(df: pd.DataFrame) -> Tuple[Dict[str, Any], float]:
    """
    Returns:
      details: dict with rvol20, z60, pct_rank_90, dollar_vol, eligible
      score : 0..1
    """
    if df is None or len(df) < 30:
        return ({"eligible": False}, 0.0)

    vol = df["Volume"].astype(float).copy()
    close = df["Close"].astype(float).copy()
    dv = close * vol

    rvol20 = float(vol.iloc[-1] / (vol.tail(21).iloc[:-1].mean() + 1e-9))
    # log-volume stabilizes skew
    lv = np.log1p(vol)
    mu = float(lv.tail(61).iloc[:-1].mean())
    sd = float(lv.tail(61).iloc[:-1].std(ddof=0) + 1e-9)
    z60 = float((lv.iloc[-1] - mu) / sd)

    # percentile rank of today's dollar volume within last 90d (excluding today for neutrality)
    last = dv.tail(91)
    ref = last.iloc[:-1]
    if len(ref) == 0:
        pr90 = 0.0
    else:
        pr90 = float((ref <= last.iloc[-1]).mean())  # 0..1
    dollar_vol = float(dv.iloc[-1])

    # Basic eligibility heuristic (won't hide anything; just input to score)
    eligible = bool(close.iloc[-1] >= 2 and (dv.tail(21).iloc[:-1].mean() >= 1_000_000))

    # Convert features to 0..1
    # rvol tends to be 0.5 .. 5 range typically; squash with 1 - exp(-x)
    def squash_rvol(x: float) -> float:
        x = max(0.0, x)
        return 1.0 - math.exp(-x/2.5)

    # z-score: 0 at z=0, ~0.5 at z=1, -> 1 as z grows
    def squash_z(z: float) -> float:
        return 1/(1+math.exp(-z))  # logistic

    s_rvol = squash_rvol(rvol20)
    s_z    = squash_z(z60)
    s_pr   = max(0.0, min(1.0, pr90))

    # Weighted average; emphasize percentile and rvol
    score = (0.45*s_pr + 0.40*s_rvol + 0.15*s_z)
    score = min(1.0, max(0.0, float(score)))

    details = {
        "rvol20": round(rvol20, 3),
        "z60": round(z60, 3),
        "pct_rank_90": round(pr90, 3),
        "dollar_vol": dollar_vol,
        "eligible": eligible,
    }
    return details, score

# ---------- External component loaders (schema-tolerant) ----------
def _dict_get_case_ins(d: Dict[str, Any], *names: str) -> Any:
    """case-insensitive getter across multiple candidate names"""
    lower = {k.lower(): v for k,v in d.items()}
    for name in names:
        if name is None: 
            continue
        v = lower.get(name.lower())
        if v is not None:
            return v
    return None

def load_component_map_latest(base_dir: str, kind: str) -> Dict[str, float]:
    """
    Loads site/data/{kind}/latest.json (if present) and returns {symbol: score_0_1}.
    The function tries multiple schema patterns:
      - { "items": { "AAPL": {"score": 0.7, ...}, ... } }
      - { "items": { "AAPL": {"score_0_1": 0.7, ...}, ... } }
      - { "items": { "AAPL": {"normalized": 0.7, ...}, ... } }
      - { "items": { "AAPL": 0.7, ... } }
      - { "AAPL": {"score": 0.7}, ... }
      - arrays with objects containing symbol and score
    If nothing matches, returns empty dict.
    """
    path = os.path.join(base_dir, "data", kind, "latest.json")
    j = read_json(path)
    if not j:
        return {}
    # candidates
    def extract_score(obj: Any) -> Optional[float]:
        if isinstance(obj, (int,float)):
            try:
                f = float(obj)
                if math.isnan(f): 
                    return None
                # clamp to 0..1 if looks like a fraction; otherwise min-max later
                return f
            except Exception:
                return None
        if isinstance(obj, dict):
            v = _dict_get_case_ins(obj, "score_0_1", "score01", "s01", "score", "normalized", "value")
            try:
                if v is None: 
                    return None
                f = float(v)
                if math.isnan(f):
                    return None
                return f
            except Exception:
                return None
        return None

    items = {}
    payload = j.get("items", j)

    if isinstance(payload, dict):
        for k, v in payload.items():
            sym = str(k).upper()
            s = extract_score(v)
            if s is not None:
                items[sym] = s
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict): 
                continue
            sym = _dict_get_case_ins(row, "symbol", "ticker")
            if not sym: 
                continue
            s = extract_score(row)
            if s is None:
                s = extract_score(_dict_get_case_ins(row, "metrics", "data", "values") or {})
            if s is not None:
                items[str(sym).upper()] = s

    # If values don't look like 0..1 (max > 1.2), min-max scale into 0..1
    vals = [v for v in items.values() if v is not None and not math.isnan(v)]
    if len(vals) >= 2:
        vmin, vmax = min(vals), max(vals)
        if vmax > 1.2 or vmin < 0.0:
            rng = (vmax - vmin) or 1.0
            items = {k: max(0.0, min(1.0, float((v - vmin)/rng))) for k,v in items.items()}
    else:
        # If single value, just clamp
        items = {k: max(0.0, min(1.0, float(v))) for k,v in items.items()}

    return items

# ---------- Main aggregation ----------
@dataclass
class Config:
    out_dir: str
    universe_csv: str
    report_date: str
    provider: str
    tiingo_token: Optional[str]
    mock_mode: bool
    w_vol: float
    w_dii: float
    w_tr: float
    w_news: float

def load_config() -> Config:
    out_dir = env_s("OUT_DIR", "site")
    universe_csv = env_s("UNIVERSE_CSV", os.path.join("data","universe.csv"))
    report_date = env_s("REPORT_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    provider = env_s("DATA_PROVIDER", "yfinance").lower()
    tiingo_token = os.environ.get("TIINGO_TOKEN", "") or None
    mock_mode = env_b("MOCK_MODE", False)

    # weights
    w_vol   = env_f("WEIGHT_VOL_ANOM", 0.25)
    w_dii   = env_f("WEIGHT_DII",      0.25)  # maps to insider_momo
    w_trend = env_f("WEIGHT_TRENDS",   0.30)
    w_news  = env_f("WEIGHT_NEWS",     0.20)

    # Normalize weights to sum to 1 (avoid downstream mismatch)
    ws = np.array([w_vol, w_dii, w_trend, w_news], dtype=float)
    if np.all(np.isfinite(ws)) and ws.sum() > 0:
        ws = ws / ws.sum()
    else:
        ws = np.array([0.25,0.25,0.30,0.20])
    w_vol, w_dii, w_trend, w_news = map(float, ws.tolist())

    return Config(out_dir, universe_csv, report_date, provider, tiingo_token, mock_mode,
                  w_vol, w_dii, w_trend, w_news)

def aggregate():
    cfg = load_config()
    ensure_dir(os.path.join(cfg.out_dir, "data"))
    out_day_dir = os.path.join(cfg.out_dir, "data", cfg.report_date)
    ensure_dir(out_day_dir)

    # Components from other steps (schema tolerant)
    base = cfg.out_dir
    dii_map    = load_component_map_latest(base, "dii")       # -> insider_momo
    trends_map = load_component_map_latest(base, "trends")    # -> trends_breakout
    news_map   = load_component_map_latest(base, "news")      # -> news
    # We intentionally do NOT depend on an outsider "price_vol_anom" file; we compute ourselves.

    # Universe
    universe = load_universe(cfg.universe_csv)
    symbols = [u["symbol"] for u in universe]
    name_map = {u["symbol"]: u.get("name","") for u in universe}

    if cfg.mock_mode:
        log_info(f"[MOCK] Generating zeroed dataset for {len(symbols)} symbols")
    else:
        log_info(f"[MD] provider={cfg.provider} symbols={len(symbols)}")

    # Fetch market data and build rows
    rows: List[Dict[str, Any]] = []
    failed = []

    for sym in symbols:
        nm = name_map.get(sym, "")
        if cfg.mock_mode:
            df = None
        elif cfg.provider == "tiingo" and cfg.tiingo_token:
            df = fetch_history_tiingo(sym, cfg.tiingo_token, months=12)
        else:
            df = fetch_history_yf(sym, months=12)

        # Price deltas default None
        d1=d5=d20=None
        vol_detail=None
        vol_score=0.0

        if df is None or len(df) == 0:
            failed.append(sym)
        else:
