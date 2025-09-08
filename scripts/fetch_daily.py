#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily aggregation script for futuretech-stock.

Outputs per-symbol records whose keys match the web UI:
- score_components: {volume_anomaly, insider_momo, trends_breakout, news}
- score_weights:    same keys (sum to 1.0)
- final_score_0_1, score_pts (0..1000)
- price_delta_1d/1w/1m (trading-day offsets 1/5/20)
- detail.vol_anomaly {rvol20,z60,pct_rank_90,dollar_vol,eligible}
- chart_url: /charts/{REPORT_DATE}/{SYMBOL}.png

Env (main):
  OUT_DIR=site (default)
  UNIVERSE_CSV=data/universe.csv
  REPORT_DATE=YYYY-MM-DD (default: UTC today)
  DATA_PROVIDER=yfinance|tiingo (default: yfinance)
  TIINGO_TOKEN=... (when DATA_PROVIDER=tiingo)
  MOCK_MODE=true|false (default: false)

  WEIGHT_VOL_ANOM=0.25
  WEIGHT_DII=0.25
  WEIGHT_TRENDS=0.30
  WEIGHT_NEWS=0.20
"""
from __future__ import annotations

import os
import sys
import json
import math
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


# ---------------------- logging ----------------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log_info(msg: str) -> None:
    print(f"{_ts()} [INFO] {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"{_ts()} [WARN] {msg}", flush=True)


def log_error(msg: str) -> None:
    print(f"{_ts()} [ERROR] {msg}", file=sys.stderr, flush=True)


# ---------------------- env helpers ----------------------
def env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def env_b(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_s(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


# ---------------------- IO ----------------------
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


# ---------------------- universe ----------------------
def load_universe(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        log_warn(f"Universe CSV missing: {csv_path} (using empty universe)")
        return []
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        symbol_col = next((c for c in df.columns if c.startswith("symbol")), "symbol")
        name_col = next((c for c in df.columns if c.startswith("name")), "name")
        out: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            sym = str(row.get(symbol_col, "")).strip().upper()
            nm = str(row.get(name_col, "")).strip()
            if sym:
                out.append({"symbol": sym, "name": nm})
        return out
    except Exception as e:
        log_error(f"Failed to load universe CSV: {e}")
        return []


# ---------------------- market data ----------------------
def fetch_history_yf(symbol: str, months: int = 12) -> Optional[pd.DataFrame]:
    if yf is None:
        log_warn("yfinance is not available")
        return None
    try:
        df = yf.Ticker(symbol).history(period=f"{months}mo", interval="1d", auto_adjust=False)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df = df.rename(columns=str.title)
            df.index = pd.to_datetime(df.index)
            return df[["Close", "Volume"]].dropna()
        return None
    except Exception as e:
        log_warn(f"[YF] history failed for {symbol}: {e}")
        return None


def fetch_history_tiingo(symbol: str, token: str, months: int = 12) -> Optional[pd.DataFrame]:
    if pdr is None:
        log_warn("pandas_datareader not available; cannot use Tiingo")
        return None
    try:
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.offsets.Day(int(30.5 * months))
        df = pdr.get_data_tiingo(symbol, api_key=token, start=start.date(), end=end.date())
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            for a, b in {"close": "Close", "volume": "Volume"}.items():
                if a in df.columns and b not in df.columns:
                    df[b] = df[a]
            return df[["Close", "Volume"]].dropna()
        return None
    except Exception as e:
        log_warn(f"[Tiingo] history failed for {symbol}: {e}")
        return None


# ---------------------- features ----------------------
def pct_change(series: pd.Series, lag: int) -> Optional[float]:
    if series is None or len(series) <= lag:
        return None
    try:
        cur = float(series.iloc[-1])
        prev = float(series.iloc[-1 - lag])
        if prev == 0 or np.isnan(prev) or np.isnan(cur):
            return None
        return (cur / prev - 1.0) * 100.0
    except Exception:
        return None


def nan_to_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f):
            return None
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
        return {"eligible": False}, 0.0

    vol = df["Volume"].astype(float).copy()
    close = df["Close"].astype(float).copy()
    dv = close * vol

    rvol20 = float(vol.iloc[-1] / (vol.tail(21).iloc[:-1].mean() + 1e-9))

    lv = np.log1p(vol)
    mu = float(lv.tail(61).iloc[:-1].mean())
    sd = float(lv.tail(61).iloc[:-1].std(ddof=0) + 1e-9)
    z60 = float((lv.iloc[-1] - mu) / sd)

    last = dv.tail(91)
    ref = last.iloc[:-1]
    if len(ref) == 0:
        pr90 = 0.0
    else:
        pr90 = float((ref <= last.iloc[-1]).mean())
    dollar_vol = float(dv.iloc[-1])

    eligible = bool(close.iloc[-1] >= 2 and (dv.tail(21).iloc[:-1].mean() >= 1_000_000))

    def squash_rvol(x: float) -> float:
        x = max(0.0, x)
        return 1.0 - math.exp(-x / 2.5)

    def squash_z(z: float) -> float:
        return 1 / (1 + math.exp(-z))

    s_rvol = squash_rvol(rvol20)
    s_z = squash_z(z60)
    s_pr = max(0.0, min(1.0, pr90))

    score = (0.45 * s_pr + 0.40 * s_rvol + 0.15 * s_z)
    score = min(1.0, max(0.0, float(score)))

    details = {
        "rvol20": round(rvol20, 3),
        "z60": round(z60, 3),
        "pct_rank_90": round(pr90, 3),
        "dollar_vol": dollar_vol,
        "eligible": eligible,
    }
    return details, score


# ---------------------- external components ----------------------
def _dict_get_case_ins(d: Dict[str, Any], *names: str) -> Any:
    lower = {k.lower(): v for k, v in d.items()}
    for name in names:
        if name is None:
            continue
        v = lower.get(name.lower())
        if v is not None:
            return v
    return None


def load_component_map_latest(base_dir: str, kind: str) -> Dict[str, float]:
    """
    Loads site/data/{kind}/latest.json (if present) -> {symbol: score_0_1}.
    Accepts various schemas and min-max scales to 0..1 if needed.
    """
    path = os.path.join(base_dir, "data", kind, "latest.json")
    j = read_json(path)
    if not j:
        return {}

    def extract_score(obj: Any) -> Optional[float]:
        if isinstance(obj, (int, float)):
            try:
                f = float(obj)
                if math.isnan(f):
                    return None
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

    items: Dict[str, float] = {}
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

    vals = [v for v in items.values() if v is not None and not math.isnan(v)]
    if len(vals) >= 2:
        vmin, vmax = min(vals), max(vals)
        if vmax > 1.2 or vmin < 0.0:
            rng = (vmax - vmin) or 1.0
            items = {k: max(0.0, min(1.0, float((v - vmin) / rng))) for k, v in items.items()}
    else:
        # single value or empty -> clamp
        items = {k: max(0.0, min(1.0, float(v))) for k, v in items.items()}

    return items


# ---------------------- main aggregation ----------------------
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
    universe_csv = env_s("UNIVERSE_CSV", os.path.join("data", "universe.csv"))
    report_date = env_s("REPORT_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    provider = env_s("DATA_PROVIDER", "yfinance").lower()
    tiingo_token = os.environ.get("TIINGO_TOKEN", "") or None
    mock_mode = env_b("MOCK_MODE", False)

    w_vol = env_f("WEIGHT_VOL_ANOM", 0.25)
    w_dii = env_f("WEIGHT_DII", 0.25)  # insider_momo
    w_trend = env_f("WEIGHT_TRENDS", 0.30)
    w_news = env_f("WEIGHT_NEWS", 0.20)

    ws = np.array([w_vol, w_dii, w_trend, w_news], dtype=float)
    if np.all(np.isfinite(ws)) and ws.sum() > 0:
        ws = ws / ws.sum()
    else:
        ws = np.array([0.25, 0.25, 0.30, 0.20])
    w_vol, w_dii, w_trend, w_news = map(float, ws.tolist())

    return Config(out_dir, universe_csv, report_date, provider, tiingo_token, mock_mode, w_vol, w_dii, w_trend, w_news)


def aggregate() -> None:
    cfg = load_config()
    ensure_dir(os.path.join(cfg.out_dir, "data"))
    out_day_dir = os.path.join(cfg.out_dir, "data", cfg.report_date)
    ensure_dir(out_day_dir)

    base = cfg.out_dir
    dii_map = load_component_map_latest(base, "dii")
    trends_map = load_component_map_latest(base, "trends")
    news_map = load_component_map_latest(base, "news")

    universe = load_universe(cfg.universe_csv)
    symbols = [u["symbol"] for u in universe]
    name_map = {u["symbol"]: u.get("name", "") for u in universe}

    if cfg.mock_mode:
        log_info(f"[MOCK] Generating zeroed dataset for {len(symbols)} symbols")
    else:
        log_info(f"[MD] provider={cfg.provider} symbols={len(symbols)}")

    rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for sym in symbols:
        nm = name_map.get(sym, "")

        if cfg.mock_mode:
            df = None
        elif cfg.provider == "tiingo" and cfg.tiingo_token:
            df = fetch_history_tiingo(sym, cfg.tiingo_token, months=12)
        else:
            df = fetch_history_yf(sym, months=12)

        d1 = d5 = d20 = None
        vol_detail = None
        vol_score = 0.0

        if df is None or len(df) == 0:
            failed.append(sym)
        else:
            df = df.sort_index()
            d1 = nan_to_none(pct_change(df["Close"], 1))
            d5 = nan_to_none(pct_change(df["Close"], 5))
            d20 = nan_to_none(pct_change(df["Close"], 20))
            vol_detail, vol_score = compute_vol_anomaly(df)

        insider = float(dii_map.get(sym, 0.0))
        trends = float(trends_map.get(sym, 0.0))
        news = float(news_map.get(sym, 0.0))

        score_components = {
            "volume_anomaly": vol_score,
            "insider_momo": insider,
            "trends_breakout": trends,
            "news": news,
        }
        score_weights = {
            "volume_anomaly": cfg.w_vol,
            "insider_momo": cfg.w_dii,
            "trends_breakout": cfg.w_tr,
            "news": cfg.w_news,
        }
        final_score = (
            score_components["volume_anomaly"] * score_weights["volume_anomaly"]
            + score_components["insider_momo"] * score_weights["insider_momo"]
            + score_components["trends_breakout"] * score_weights["trends_breakout"]
            + score_components["news"] * score_weights["news"]
        )
        final_score = max(0.0, min(1.0, float(final_score)))
        pts = int(round(final_score * 1000))

        rows.append(
            {
                "symbol": sym,
                "name": nm,
                "final_score_0_1": round(final_score, 6),
                "score_pts": pts,
                "score_components": score_components,
                "score_weights": score_weights,
                "price_delta_1d": d1,
                "price_delta_1w": d5,
                "price_delta_1m": d20,
                "detail": {"vol_anomaly": vol_detail or {}},
                "chart_url": f"/charts/{cfg.report_date}/{sym}.png",
            }
        )

    rows.sort(key=lambda r: (r.get("score_pts", 0), r.get("final_score_0_1", 0.0)), reverse=True)
    top10 = rows[:10]
    for i, r in enumerate(top10, start=1):
        r["rank"] = i

    log_info(f"Generated top10 for {cfg.report_date}: {len(top10)} symbols (universe={len(symbols)})")
    if failed:
        log_warn(f"{len(failed)} failed downloads: {failed}")

    out_path = os.path.join(out_day_dir, "top10.json")
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"date": cfg.report_date, "items": top10}, f, ensure_ascii=False, indent=2)

    latest_dir = os.path.join(cfg.out_dir, "data", "top10")
    ensure_dir(latest_dir)
    latest_path = os.path.join(latest_dir, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"date": cfg.report_date, "items": top10}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        aggregate()
    except Exception as e:
        log_error(f"FATAL in fetch_daily: {e}")
        sys.exit(1)
