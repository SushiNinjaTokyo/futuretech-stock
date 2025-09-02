#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily scorer / aggregator
- 入力: yfinance（日足）、Trends / News / DII の latest.json
- 出力: site/data/<date>/top10.json と site/data/latest.json
- 仕様：
  * テンプレートが期待するキー名に完全一致（score_pts, deltas など）
  * どの外部データが欠けても動作（重みは存在する要素で正規化）
"""

from __future__ import annotations
import os, sys, json, time, math, random
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr

# ---- env / config
OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "data/universe.csv").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()

WEIGHT_VOL_ANOM = float(os.environ.get("WEIGHT_VOL_ANOM", "0.25"))
WEIGHT_DII      = float(os.environ.get("WEIGHT_DII", "0.25"))
WEIGHT_TRENDS   = float(os.environ.get("WEIGHT_TRENDS", "0.30"))
WEIGHT_NEWS     = float(os.environ.get("WEIGHT_NEWS", "0.20"))

PRICE_LOOKBACK_DAYS = 180
VOL_BASE_DAYS_MIN   = 40

REQ_SLEEP_BASE = 0.25
REQ_SLEEP_JITTER = 0.2

DATA_ROOT        = os.path.join(OUT_DIR, "data")
DATE_DIR         = os.path.join(DATA_ROOT, REPORT_DATE) if REPORT_DATE else os.path.join(DATA_ROOT, "today")
TRENDS_JSON_PATH = os.path.join(DATA_ROOT, "trends", "latest.json")
NEWS_JSON_PATH   = os.path.join(DATA_ROOT, "news", "latest.json")
DII_JSON_PATH    = os.path.join(DATA_ROOT, "dii", "latest.json")

def ensure_dirs():
    os.makedirs(DATE_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_ROOT, "latest"), exist_ok=True)

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def soft_cap01(x: float, k: float = 5.0) -> float:
    x = max(0.0, float(x))
    return 1.0 - math.exp(-k * x)

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
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker":"symbol"})
        else:
            raise RuntimeError("universe.csv must include 'symbol'")
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["name"]   = df["name"].astype(str).str.strip()
    return df[["symbol","name"]].copy()

def items_map_from(payload: Any) -> Dict[str, dict]:
    """
    {"items":[{...}]} / {"items":{sym:{...}}} のどちらでもOKに。
    返り値: {SYMBOL_UPPER: record}
    """
    m: Dict[str, dict] = {}
    j = payload or {}
    items = j.get("items") if isinstance(j, dict) else None
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict): 
                continue
            k = str(it.get("symbol") or it.get("ticker") or it.get("name") or "").strip().upper()
            if k:
                m[k] = it
    elif isinstance(items, dict):
        for k, it in items.items():
            if isinstance(it, dict):
                m[str(k).strip().upper()] = it
    return m

# --------------- market data ---------------

def fetch_history_yf(symbols: List[str], end_date: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    batch = 6
    for i in range(0, len(symbols), batch):
        subs = symbols[i:i+batch]
        try:
            df = yf.download(
                tickers=" ".join(subs),
                end=end_date,
                period="9mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                for sym in subs:
                    if sym in df.columns.get_level_values(0):
                        subdf = df[sym].rename_axis("Date").copy()
                        if not subdf.empty:
                            out[sym] = subdf
            else:
                out[subs[0]] = df
        except Exception:
            pass
        sleep_brief()
    return out

def fetch_history_stooq(symbol: str, end_date: str) -> pd.DataFrame:
    try:
        df = pdr.DataReader(symbol, data_source="stooq")
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df[["Open","High","Low","Close","Volume"]].copy()
    except Exception:
        return pd.DataFrame()

def last_row(h: pd.DataFrame, report_date: str):
    if h is None or h.empty:
        return (None, None)
    h = h.copy()
    h.index = pd.to_datetime(h.index)
    target = pd.to_datetime(report_date)
    if target in h.index:
        return (target, h.loc[target])
    sub = h[h.index <= target]
    if sub.empty:
        return (None, None)
    ts = sub.index.max()
    return (ts, sub.loc[ts])

def rvol_score(hist: pd.DataFrame, ts) -> Tuple[float, float]:
    """出来高ロバスト異常度 (値[%], スコアpts[0..1000])"""
    if hist is None or hist.empty or "Volume" not in hist.columns:
        return (0.0, 0.0)
    h = hist.copy()
    h.index = pd.to_datetime(h.index)
    h = h[h.index <= ts]
    if len(h) < VOL_BASE_DAYS_MIN:
        return (0.0, 0.0)
    v_today = float(h.iloc[-1]["Volume"])
    window = h.iloc[:-1].tail(60)["Volume"]
    if window.empty:
        return (0.0, 0.0)
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))
    if mad <= 0:
        q25, q75 = np.percentile(window, [25, 75])
        iqr = max(1.0, q75 - q25)
        z = (v_today - med) / (iqr / 1.349)
    else:
        z = (v_today - med) / (1.4826 * mad)
    pct_above = 0.0 if med <= 0 else max(0.0, (v_today / med - 1.0) * 100.0)
    z_pos = max(0.0, float(z))
    score01 = soft_cap01(z_pos / 3.0, k=4.0)
    return (round(pct_above, 2), round(score01 * 1000.0))

def pct_change_n(h: pd.DataFrame, ts, n: int) -> float:
    if h is None or h.empty or "Close" not in h.columns:
        return 0.0
    h = h.copy()
    h.index = pd.to_datetime(h.index)
    h = h[h.index <= ts]
    if len(h) < n + 1:
        return 0.0
    c_now = float(h["Close"].iloc[-1])
    c_prev = float(h["Close"].iloc[-(n+1)])
    return 0.0 if c_prev == 0 else round((c_now / c_prev - 1.0) * 100.0, 2)

# --------------- main ---------------

def main():
    if not REPORT_DATE:
        print("ERROR: REPORT_DATE env is empty. Run et_market_date.py first.", file=sys.stderr)
        sys.exit(2)

    ensure_dirs()

    uni = parse_universe(UNIVERSE_CSV)
    symbols = [str(s).upper() for s in uni["symbol"].tolist()]
    names_map = {str(r.symbol).upper(): str(r.name) for r in uni.itertuples(index=False)}

    trends_map = items_map_from(load_json(TRENDS_JSON_PATH, {}))
    news_map   = items_map_from(load_json(NEWS_JSON_PATH, {}))
    dii_map    = items_map_from(load_json(DII_JSON_PATH, {}))

    yf_hist = fetch_history_yf(symbols, REPORT_DATE)

    results: List[Dict[str, Any]] = []

    for sym in symbols:
        name = names_map.get(sym, sym)
        h = yf_hist.get(sym)
        ts, row = last_row(h, REPORT_DATE)
        if ts is None:
            h2 = fetch_history_stooq(sym, REPORT_DATE)
            ts, row = last_row(h2, REPORT_DATE)
            if ts is not None:
                h = h2

        # price deltas
        if ts is None:
            d1 = d5 = d20 = 0.0
            vol_value, vol_pts = (0.0, 0.0)
        else:
            vol_value, vol_pts = rvol_score(h, ts)
            d1  = pct_change_n(h, ts, 1)
            d5  = pct_change_n(h, ts, 5)
            d20 = pct_change_n(h, ts, 21)

        # features
        t_rec = trends_map.get(sym, {})
        n_rec = news_map.get(sym, {})
        d_rec = dii_map.get(sym, {})

        trends01 = clamp01(t_rec.get("score_0_1", t_rec.get("breakout_0_1", 0.0)))
        news01   = clamp01(n_rec.get("score_0_1", 0.0))
        dii01    = clamp01(d_rec.get("score_0_1", 0.0))

        # weights re-normalize for present components
        comps01 = {
            "volume_anomaly": vol_pts / 1000.0,
            "trends_breakout": trends01,
            "news": news01,
            "dii": dii01,
        }
        weights = {
            "volume_anomaly": WEIGHT_VOL_ANOM,
            "trends_breakout": WEIGHT_TRENDS,
            "news": WEIGHT_NEWS,
            "dii": WEIGHT_DII,
        }
        # 欠損実装: 存在しない/0を無視して再正規化（ただし出来高は常に有）
        present = {k: v for k, v in comps01.items()}
        wsum = sum(max(0.0, weights.get(k, 0.0)) for k in present.keys())
        wsum = wsum if wsum > 0 else 1.0
        wnorm = {k: max(0.0, weights.get(k, 0.0)) / wsum for k in present.keys()}

        final01 = sum(wnorm[k] * present[k] for k in present.keys())
        final_pts = int(round(final01 * 1000.0))

        # 詳細（出来高）
        detail_vol = {
            "rvol20": None,  # 将来用
            "z60": None,
            "pct_rank_90": None,
            "dollar_vol": None,
            "eligible": True
        }

        # ペイロード（テンプレ互換）
        results.append({
            "symbol": sym,
            "name": name,
            "score_pts": final_pts,
            "final_score_0_1": final01,
            "deltas": {"d1": d1, "d5": d5, "d20": d20},
            "trends_breakout": trends01,
            "vol_anomaly_score": comps01["volume_anomaly"],
            "news_score": news01,
            "news_recent_count": int(n_rec.get("recent_count", 0) or 0),
            "dii_score": dii01,
            "dii_components": d_rec.get("components", {}),
            "score_components": comps01,
            "score_weights": wnorm,
            "detail": {"vol_anomaly": detail_vol},
        })

        sleep_brief()

    results.sort(key=lambda x: x["score_pts"], reverse=True)
    top10 = results[:10]

    payload = {"date": REPORT_DATE, "universe_count": len(symbols), "count": len(top10), "items": top10}

    ensure_dirs()
    out_path = os.path.join(DATE_DIR, "top10.json")
    latest_path = os.path.join(DATA_ROOT, "latest.json")
    save_json(out_path, payload)
    save_json(latest_path, payload)

    print(f"Generated top10 for {REPORT_DATE}: {len(top10)} symbols (universe={len(symbols)})")

if __name__ == "__main__":
    main()
