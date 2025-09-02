#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily scorer
- Volume anomaly: per-symbol abnormal volume vs its own history (robust z)
- Add DII model, Google Trends, News coverage to scoring
- Fair across sizes
- Data source: yfinance primary, stooq fallback
- Market date is given by env REPORT_DATE (already resolved to US market date)
- Sleep jitter to avoid API rate limiting
- Output: site/data/<REPORT_DATE>/top10.json and site/data/latest.json
- English labels only (for global audience)
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

import yfinance as yf
from pandas_datareader import data as pdr


# --------------------------
# Config (weights & paths)
# --------------------------

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "data/universe.csv").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()

WEIGHT_VOL_ANOM = float(os.environ.get("WEIGHT_VOL_ANOM", "0.25"))
WEIGHT_DII      = float(os.environ.get("WEIGHT_DII", "0.25"))
WEIGHT_TRENDS   = float(os.environ.get("WEIGHT_TRENDS", "0.30"))
WEIGHT_NEWS     = float(os.environ.get("WEIGHT_NEWS", "0.20"))

# history lookbacks
PRICE_LOOKBACK_DAYS = 120  # for volume baseline & returns
VOL_BASE_DAYS_MIN   = 40   # need minimum to compute anomaly

# data sources sleep
REQ_SLEEP_BASE = 0.3
REQ_SLEEP_JITTER = 0.25

# paths for side-data produced by other scripts
DATA_ROOT        = os.path.join(OUT_DIR, "data")
DATE_DIR         = os.path.join(DATA_ROOT, REPORT_DATE) if REPORT_DATE else os.path.join(DATA_ROOT, "today")
TRENDS_JSON_PATH = os.path.join(DATA_ROOT, "trends", "latest.json")
NEWS_JSON_PATH   = os.path.join(DATA_ROOT, "news", "latest.json")
DII_JSON_PATH    = os.path.join(DATA_ROOT, "dii", "latest.json")


# --------------------------
# Small utils
# --------------------------

def ensure_dirs():
    os.makedirs(DATE_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_ROOT, "latest"), exist_ok=True)

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def soft_cap01(x: float, k: float = 5.0) -> float:
    """map [0, +inf) to [0,1) with smooth saturation"""
    x = max(0.0, float(x))
    return 1.0 - math.exp(-k * x)  # simple smooth cap

def load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def sleep_brief():
    t = REQ_SLEEP_BASE + random.random() * REQ_SLEEP_JITTER
    time.sleep(t)

def parse_universe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize expected columns
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    rename = {}
    if "ticker" in df.columns and "symbol" not in df.columns:
        rename["ticker"] = "symbol"
    if rename:
        df = df.rename(columns=rename)
    if "symbol" not in df.columns:
        raise RuntimeError("universe.csv must include 'symbol' column")
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    return df[["symbol", "name"]].copy()

# ------------------------------------------------
# Robust loaders for trends/news/dii "latest.json"
# ------------------------------------------------

def load_items_map(path: str) -> Dict[str, dict]:
    """
    Accepts formats:
      - {"items": {"NVDA": {...}, "NVIDIA CORPORATION": {...}}}
      - {"items": [ {"symbol":"NVDA", ...}, {"name":"NVIDIA Corporation", ...} ]}
    Returns: {UPPER_KEY: record}
    """
    j = load_json(path, default={}) or {}
    items = j.get("items") or {}
    m: Dict[str, dict] = {}

    def put(key: str, rec: dict):
        k = (key or "").strip().upper()
        if not k or not isinstance(rec, dict):
            return
        if k not in m or len(rec) > len(m[k]):
            m[k] = rec

    if isinstance(items, dict):
        for k, v in items.items():
            if not isinstance(v, dict):
                continue
            sym  = str(v.get("symbol") or v.get("ticker") or "").strip().upper()
            name = str(v.get("name")   or v.get("query")  or "").strip().upper()
            put(k, v)
            if sym:  put(sym, v)
            if name: put(name, v)
    elif isinstance(items, list):
        for v in items:
            if not isinstance(v, dict):
                continue
            sym  = str(v.get("symbol") or v.get("ticker") or "").strip().upper()
            name = str(v.get("name")   or v.get("query")  or "").strip().upper()
            if sym:  put(sym, v)
            if name: put(name, v)
    return m

def extract_score01(obj: dict, default: float = 0.0) -> float:
    """
    Try common keys to get a 0..1 score. If not present, convert from raw metrics.
    """
    if not isinstance(obj, dict):
        return default

    candidates = [
        "score_0_1", "score",
        "coverage_0_1", "coverage_score",
        "breakout_0_1", "breakout_score",
        "norm", "normalized", "z_u", "zscore_0_1",
    ]
    for k in candidates:
        if k in obj:
            try:
                return clamp01(obj[k])
            except Exception:
                pass

    # derive from typical raw fields
    if "z" in obj:
        try:
            z = float(obj.get("z") or 0.0)
            # clamp +/-3σ and map 0→0.5
            return clamp01(0.5 + max(-3.0, min(3.0, z)) / 6.0)
        except Exception:
            pass

    cnt = obj.get("count") or obj.get("recent_count") or obj.get("total_count")
    tot = obj.get("total")
    try:
        if cnt is not None and tot:
            cnt = float(cnt); tot = float(tot)
            if tot > 0:
                return clamp01(cnt / tot)
    except Exception:
        pass

    return default


# --------------------------
# Market data fetch helpers
# --------------------------

def fetch_history_yf(symbols: List[str], end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Download history for multiple symbols via yfinance (primary).
    Returns dict of symbol -> DataFrame (index=Datetime, columns=['Open','High','Low','Close','Adj Close','Volume'])
    """
    out: Dict[str, pd.DataFrame] = {}
    # yfinance can batch, but to be gentle mix small batches + sleep
    batch = 6
    for i in range(0, len(symbols), batch):
        subs = symbols[i:i+batch]
        try:
            df = yf.download(
                tickers=" ".join(subs),
                end=end_date,
                period="6mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True
            )
            # When single symbol, columns not multiindex; normalize
            if isinstance(df.columns, pd.MultiIndex):
                for sym in subs:
                    if sym in df.columns.get_level_values(0):
                        subdf = df[sym].rename_axis("Date").copy()
                        if not subdf.empty:
                            out[sym] = subdf
            else:
                # single symbol case
                out[subs[0]] = df
        except Exception:
            # ignore, will fallback
            pass
        sleep_brief()
    return out

def fetch_history_stooq(symbol: str, end_date: str) -> pd.DataFrame:
    """
    Stooq fallback for a single symbol.
    """
    try:
        # Stooq ignores end parameter; fetch longer range then slice
        df = pdr.DataReader(symbol, data_source="stooq")
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        # rename to yfinance-like columns
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        # keep last ~6 months
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df[["Open","High","Low","Close","Volume"]].copy()
    except Exception:
        return pd.DataFrame()

def get_last_trading_row(h: pd.DataFrame, report_date: str) -> Tuple[pd.Timestamp, pd.Series] | Tuple[None, None]:
    if h is None or h.empty:
        return (None, None)
    h = h.copy()
    h.index = pd.to_datetime(h.index)
    # prefer the report_date row; otherwise previous available row
    target = pd.to_datetime(report_date)
    if target in h.index:
        row = h.loc[target]
        return (target, row)
    else:
        sub = h[h.index <= target]
        if sub.empty:
            return (None, None)
        ts = sub.index.max()
        return (ts, sub.loc[ts])

def robust_vol_anomaly_score(hist: pd.DataFrame, ts: pd.Timestamp) -> Tuple[float, float]:
    """
    Compute per-symbol abnormal volume score (0..1) using robust stats.
    - baseline: 60-day rolling median & MAD
    - score: soft-capped scaled robust z
    Returns (value_raw, score_points) where value_raw is e.g., % above median
    """
    if hist is None or hist.empty or "Volume" not in hist.columns:
        return (0.0, 0.0)

    h = hist.copy()
    h.index = pd.to_datetime(h.index)
    h = h[h.index <= ts]
    if len(h) < VOL_BASE_DAYS_MIN:
        return (0.0, 0.0)

    # last day's volume
    v_today = float(h.loc[ts, "Volume"])

    # robust baseline from trailing 60 business days before ts (exclude ts)
    window = h.iloc[:-1].tail(60)["Volume"]
    if window.empty or window.std() == 0:
        return (0.0, 0.0)

    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))  # median absolute deviation
    if mad <= 0:
        # fallback to iqr
        q25, q75 = np.percentile(window, [25, 75])
        iqr = max(1.0, q75 - q25)
        z = (v_today - med) / (iqr / 1.349)  # approximate to std
    else:
        z = (v_today - med) / (1.4826 * mad)  # MAD to sigma

    # value: relative to median, %
    value_pct_above = 0.0
    if med > 0:
        value_pct_above = max(0.0, (v_today / med - 1.0) * 100.0)

    # normalize z to 0..1 with soft cap and half-wave rectify (only spikes up matter)
    z_pos = max(0.0, float(z))
    score01 = soft_cap01(z_pos / 3.0, k=4.0)  # z≈3 -> near 1.0

    # points in 0..1000 domain for UI consistency
    points = round(score01 * 1000.0)

    return (round(value_pct_above, 2), float(points))


def pct_change_over_n(h: pd.DataFrame, ts: pd.Timestamp, n_days: int) -> float:
    if h is None or h.empty or "Close" not in h.columns:
        return 0.0
    h = h.copy()
    h.index = pd.to_datetime(h.index)
    h = h[h.index <= ts]
    if len(h) < n_days + 1:
        return 0.0
    c_now = float(h["Close"].iloc[-1])
    c_prev = float(h["Close"].iloc[-(n_days+1)])
    if c_prev == 0:
        return 0.0
    return round((c_now / c_prev - 1.0) * 100.0, 2)


# --------------------------
# Main
# --------------------------

def main():
    ensure_dirs()

    uni = parse_universe(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]
    names_map = {str(row.symbol).strip().upper(): str(row.name).strip() for row in uni.itertuples(index=False)}

    # side features
    trends_map = load_items_map(TRENDS_JSON_PATH)
    news_map   = load_items_map(NEWS_JSON_PATH)
    dii_map    = load_items_map(DII_JSON_PATH)

    # batch fetch prices via yfinance (primary)
    yf_hist = fetch_history_yf(symbols, REPORT_DATE)

    results = []
    for sym in symbols:
        name = names_map.get(sym, sym)

        h = yf_hist.get(sym)
        ts, row = get_last_trading_row(h, REPORT_DATE)

        # if miss, fallback to stooq
        if ts is None:
            sleep_brief()
            h2 = fetch_history_stooq(sym, REPORT_DATE)
            ts, row = get_last_trading_row(h2, REPORT_DATE)
            if ts is not None:
                h = h2

        if ts is None or row is None:
            # could not fetch anything — skip scoring but keep placeholder
            vol_value, vol_points = (0.0, 0.0)
            d1 = d5 = d21 = 0.0
        else:
            vol_value, vol_points = robust_vol_anomaly_score(h, ts)
            # 1D, 1W(~5), 1M(~21) % change
            d1  = pct_change_over_n(h, ts, 1)
            d5  = pct_change_over_n(h, ts, 5)
            d21 = pct_change_over_n(h, ts, 21)

        # --- external features with name fallback ---
        name_upper = name.strip().upper()
        t_rec = trends_map.get(sym) or trends_map.get(name_upper) or {}
        n_rec = news_map.get(sym)   or news_map.get(name_upper)   or {}
        d_rec = dii_map.get(sym)    or dii_map.get(name_upper)    or {}

        trends01 = extract_score01(t_rec, 0.0)
        news01   = extract_score01(n_rec, 0.0)
        dii01    = extract_score01(d_rec, 0.0)

        # Component points (0..1000)
        comp_points = {
            "Volume anomaly": vol_points,
            "Trend breakout": round(trends01 * 1000),
            "News coverage":   round(news01   * 1000),
            "DII model":       round(dii01    * 1000),
        }

        # Final weighted score (0..1000)
        final_points = (
            WEIGHT_VOL_ANOM * comp_points["Volume anomaly"] +
            WEIGHT_TRENDS   * comp_points["Trend breakout"] +
            WEIGHT_NEWS     * comp_points["News coverage"] +
            WEIGHT_DII      * comp_points["DII model"]
        )
        final_points = round(final_points)

        results.append({
            "symbol": sym,
            "name": name,
            "rank_points": final_points,
            "returns": {
                "1D": d1,
                "1W": d5,
                "1M": d21,
            },
            "components": [
                {
                    "name": "News coverage",
                    "value": round(news01 * 100),  # show as 0..100 for UI
                    "weight": int(round(WEIGHT_NEWS * 100)),
                    "points": comp_points["News coverage"],
                },
                {
                    "name": "Trend breakout",
                    "value": round(trends01 * 100),
                    "weight": int(round(WEIGHT_TRENDS * 100)),
                    "points": comp_points["Trend breakout"],
                },
                {
                    "name": "Volume anomaly",
                    "value": vol_value,  # % above median volume
                    "weight": int(round(WEIGHT_VOL_ANOM * 100)),
                    "points": comp_points["Volume anomaly"],
                },
                {
                    "name": "DII model",
                    "value": round(dii01 * 100),
                    "weight": int(round(WEIGHT_DII * 100)),
                    "points": comp_points["DII model"],
                },
            ],
            "explain": "Final score = weighted average of Volume anomaly, Trend breakout, News coverage, and DII model. (0–1000 pts)"
        })

        sleep_brief()

    # sort and take top10
    results.sort(key=lambda x: x["rank_points"], reverse=True)
    top10 = results[:10]

    payload = {
        "date": REPORT_DATE,
        "universe_count": len(symbols),
        "count": len(top10),
        "items": top10
    }

    # save
    ensure_dirs()
    out_path = os.path.join(DATE_DIR, "top10.json")
    latest_path = os.path.join(DATA_ROOT, "latest.json")
    save_json(out_path, payload)
    save_json(latest_path, payload)

    print(f"Generated top10 for {REPORT_DATE}: {len(top10)} symbols (universe={len(symbols)})")


if __name__ == "__main__":
    if not REPORT_DATE:
        print("ERROR: REPORT_DATE env is empty. Run et_market_date.py first.", file=sys.stderr)
        sys.exit(2)
    main()
