#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def env_s(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def env_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=sanitize),
        encoding="utf-8",
    )


def load_universe(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        log("ERROR", f"Universe CSV missing: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol", list(df.columns)[0])
    name_col = cols.get("name")

    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip().upper()
        if not sym:
            continue
        nm = str(row.get(name_col, "")).strip() if name_col else ""
        out.append({"symbol": sym, "name": nm})
    return out


def first_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")

    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        s = x.iloc[:, 0]
        if isinstance(s, pd.DataFrame):
            if s.shape[1] == 0:
                return pd.Series(dtype=float)
            s = s.iloc[:, 0]
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce")

    return pd.Series(dtype=float)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Open, High, Low, Close, Volume を1次元列に統一
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = pd.DataFrame(index=pd.to_datetime(df.index))

    target_names = ["open", "high", "low", "close", "volume"]

    if isinstance(df.columns, pd.MultiIndex):
        chosen: Dict[str, Any] = {}
        for col in df.columns:
            parts = [str(c).strip().lower() for c in col if c is not None]
            for t in target_names:
                if t in parts and t not in chosen:
                    chosen[t] = col

        for t in target_names:
            if t in chosen:
                out[t.capitalize()] = first_series(df.loc[:, chosen[t]]).to_numpy()
            else:
                out[t.capitalize()] = np.nan
    else:
        colmap = {str(c).strip().lower(): c for c in df.columns}
        for t in target_names:
            src = colmap.get(t)
            if src is not None:
                ser = df[[src]] if isinstance(df[src], pd.DataFrame) else df[src]
                out[t.capitalize()] = first_series(ser).to_numpy()
            else:
                out[t.capitalize()] = np.nan

    out = out[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return out.sort_index()


def fetch_history(symbol: str, provider: str, token: Optional[str], months: int = 6) -> Optional[pd.DataFrame]:
    for attempt in range(3):
        try:
            if provider == "tiingo" and token and pdr is not None:
                end = pd.Timestamp.utcnow().normalize()
                start = end - pd.DateOffset(months=months)
                raw = pdr.get_data_tiingo(
                    symbol,
                    api_key=token,
                    start=start.date(),
                    end=end.date(),
                )

                if isinstance(raw.index, pd.MultiIndex):
                    raw = raw.reset_index(level=0, drop=True)

                rename_map = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
                for src, dst in rename_map.items():
                    if src in raw.columns:
                        raw[dst] = raw[src]

            else:
                if yf is None:
                    return None
                raw = yf.download(
                    symbol,
                    period=f"{months}mo",
                    interval="1d",
                    progress=False,
                    threads=False,
                    auto_adjust=False,
                )

            if raw is None or len(raw) < 80:
                raise ValueError("empty or too short dataframe")

            df = normalize_ohlcv(raw)
            if len(df) < 80:
                raise ValueError("not enough clean rows")

            return df

        except Exception as e:
            log("WARN", f"{symbol}: fetch_history attempt {attempt + 1} failed: {e}")
            time.sleep(1.2 + attempt * 0.7)

    return None


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def atr(df: pd.DataFrame, window: int) -> float:
    high = first_series(df["High"]).dropna()
    low = first_series(df["Low"]).dropna()
    close = first_series(df["Close"]).dropna()

    n = min(len(high), len(low), len(close))
    if n < window + 1:
        return 0.0

    high = high.iloc[-n:]
    low = low.iloc[-n:]
    close = close.iloc[-n:]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr = tr.dropna()
    if len(tr) < window:
        return 0.0

    return float(tr.tail(window).mean())


def positive_percentile(values: List[float], x: float) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0

    if max(vals) <= 0 or x <= 0:
        return 0.0

    pos = sorted(v for v in vals if v > 0)
    if not pos:
        return 0.0

    if len(pos) == 1:
        return 1.0 if x >= pos[0] else 0.0

    lt = sum(1 for v in pos if v < x)
    eq = sum(1 for v in pos if v == x)
    rank = (lt + 0.5 * max(0, eq - 1)) / (len(pos) - 1)
    return clamp(rank)


def compute_compression_release(df: pd.DataFrame) -> Tuple[Dict[str, Any], float]:
    """
    compression release:
    - 直近10日が60日対比で縮んでいるか
    - ボラが圧縮しているか
    - ATRが圧縮しているか
    - そのうえで、今日上方向に動き始めているか
    """
    if df is None or len(df) < 80:
        return {"eligible": False}, 0.0

    o = first_series(df["Open"]).dropna()
    h = first_series(df["High"]).dropna()
    l = first_series(df["Low"]).dropna()
    c = first_series(df["Close"]).dropna()
    v = first_series(df["Volume"]).dropna()

    n = min(len(o), len(h), len(l), len(c), len(v))
    if n < 80:
        return {"eligible": False}, 0.0

    o = o.iloc[-n:]
    h = h.iloc[-n:]
    l = l.iloc[-n:]
    c = c.iloc[-n:]
    v = v.iloc[-n:]

    c_last = float(c.iloc[-1])
    c_prev = float(c.iloc[-2])

    h10 = h.tail(10)
    l10 = l.tail(10)
    h60 = h.tail(60)
    l60 = l.tail(60)

    range10 = float(h10.max() - l10.min())
    range60 = float(h60.max() - l60.min())
    range_ratio_10_60 = 1.0 if range60 <= 0 else range10 / range60

    ret = c.pct_change().dropna()
    vol10 = float(ret.tail(10).std(ddof=0)) if len(ret) >= 10 else 0.0
    vol60 = float(ret.tail(60).std(ddof=0)) if len(ret) >= 60 else 0.0
    vol_ratio_10_60 = 1.0 if vol60 <= 0 else vol10 / vol60

    atr10 = atr(pd.DataFrame({"High": h, "Low": l, "Close": c}), 10)
    atr60 = atr(pd.DataFrame({"High": h, "Low": l, "Close": c}), 60)
    atr_ratio_10_60 = 1.0 if atr60 <= 0 else atr10 / atr60

    # 圧縮度
    # 小さいほど良いので 1 - ratio 系へ変換
    s_range = clamp(1.0 - (range_ratio_10_60 / 0.65))
    s_vol = clamp(1.0 - (vol_ratio_10_60 / 0.75))
    s_atr = clamp(1.0 - (atr_ratio_10_60 / 0.80))
    compression_score = clamp(0.40 * s_range + 0.35 * s_vol + 0.25 * s_atr)

    # 立ち上がり
    high10 = float(h10.max())
    low10 = float(l10.min())
    close_pos_10d = 0.5 if high10 <= low10 else (c_last - low10) / (high10 - low10)
    s_closepos = clamp((close_pos_10d - 0.55) / 0.45)

    thrust_atr = 0.0 if atr10 <= 0 else max(0.0, (c_last - c_prev) / atr10)
    s_thrust = clamp(thrust_atr / 1.0)

    sma10 = float(c.tail(10).mean())
    sma20 = float(c.tail(20).mean())
    s_ma = (0.6 if c_last > sma10 else 0.0) + (0.4 if c_last > sma20 else 0.0)

    avg_vol20_ex_today = float(v.tail(21).iloc[:-1].mean()) if len(v) >= 21 else float(v.mean())
    rvol20 = c_last * 0.0
    if avg_vol20_ex_today > 0:
        rvol20 = float(v.iloc[-1] / avg_vol20_ex_today)
    s_rvol = clamp(math.log1p(max(0.0, rvol20)) / math.log1p(3.0))

    release_score = clamp(
        0.35 * s_closepos
        + 0.30 * s_thrust
        + 0.20 * s_rvol
        + 0.15 * s_ma
    )

    # 圧縮だけでも、上げだけでもダメ。両方必要
    raw_score = math.sqrt(max(0.0, compression_score * release_score))
    raw_score = clamp(raw_score)

    dollar_vol20 = float((c.tail(20) * v.tail(20)).mean())
    eligible = bool(c_last >= 1.0 and dollar_vol20 >= 100_000)

    detail = {
        "eligible": eligible,
        "range_ratio_10_60": round(range_ratio_10_60, 4),
        "vol_ratio_10_60": round(vol_ratio_10_60, 4),
        "atr_ratio_10_60": round(atr_ratio_10_60, 4),
        "compression_score": round(compression_score, 6),
        "close_pos_10d": round(close_pos_10d, 4),
        "thrust_atr": round(thrust_atr, 4),
        "rvol20": round(rvol20, 4),
        "release_score": round(release_score, 6),
        "dollar_vol20": round(dollar_vol20, 2),
        "price_1d_pct": round((c_last / c_prev - 1.0) * 100.0, 2),
    }

    if not eligible:
        return detail, 0.0

    return detail, round(raw_score, 8)


def main() -> None:
    out_dir = Path(env_s("OUT_DIR", "site"))
    universe_csv = Path(env_s("UNIVERSE_CSV", os.path.join("data", "universe.csv")))
    report_date = env_s("REPORT_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    provider = env_s("DATA_PROVIDER", "yfinance").lower()
    token = os.getenv("TIINGO_TOKEN") or None

    universe = load_universe(universe_csv)
    if not universe:
        raise SystemExit("Universe is empty")

    raw_items: List[Dict[str, Any]] = []
    raw_scores: List[float] = []

    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")
        log("INFO", f"Processing {sym}")

        df = fetch_history(sym, provider, token, months=6)
        if df is None:
            detail = {"eligible": False}
            raw_score = 0.0
        else:
            detail, raw_score = compute_compression_release(df)

        raw_items.append({
            "symbol": sym,
            "name": nm,
            "compression_raw_score": round(raw_score, 8),
            **detail,
        })
        raw_scores.append(raw_score)

    items: List[Dict[str, Any]] = []
    for row in raw_items:
        score01 = positive_percentile(raw_scores, float(row["compression_raw_score"]))
        items.append({
            **row,
            "score_0_1": round(score01, 6),
        })

    payload = {
        "date": report_date,
        "items": items,
    }

    day_path = out_dir / "data" / report_date / "compression.json"
    latest_path = out_dir / "data" / "compression" / "latest.json"

    write_json(day_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote Compression: {day_path} ({len(items)} items)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in fetch_compression: {e}")
        sys.exit(1)
