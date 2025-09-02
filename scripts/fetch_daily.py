#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute daily ranking and persist site/data/<date>/top10.json
Inputs:
- site/data/<date>/dii.json
- site/data/<date>/trends.json
- site/data/<date>/news.json
Env:
- OUT_DIR (default: site)
- UNIVERSE_CSV (default: data/universe.csv)
- REPORT_DATE (YYYY-MM-DD)  … produced by et_market_date.py
- DATA_PROVIDER (yfinance|tiingo) default yfinance
- WEIGHT_VOL_ANOM, WEIGHT_TRENDS, WEIGHT_NEWS, WEIGHT_DII (float 0..1)
- MOCK_MODE=false
"""

from __future__ import annotations
import os, sys, json, time, math, dataclasses, logging, pathlib, statistics as stats
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)sZ [%(levelname)s] %(message)s",
)
logging.Formatter.converter = time.gmtime

def t0():
    return time.perf_counter()

def lap(since, label):
    dt = time.perf_counter() - since
    logging.info("[TIME] %s: %.3fs", label, dt)

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_json(path: pathlib.Path, default=None):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default

def ensure_dir(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def pct(x):
    return float(x) * 100.0

def safe_div(a, b, default=0.0):
    try:
        return float(a) / float(b) if float(b) != 0 else default
    except Exception:
        return default

def clip01(x):
    return max(0.0, min(1.0, float(x)))

def zscore(vs: List[float], v: float, ddof=0) -> float:
    if not vs:
        return 0.0
    mu = stats.fmean(vs)
    sd = (stats.pstdev(vs) if ddof == 0 else stats.stdev(vs)) or 1e-9
    return (v - mu) / sd

# ──────────────────────────────────────────────────────────────────────────────
# config
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = pathlib.Path(os.getenv("OUT_DIR", "site"))
DATA_DIR = OUT_DIR / "data"
REPORT_DATE = os.getenv("REPORT_DATE")
UNIVERSE_CSV = pathlib.Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))

WEIGHT_VOL_ANOM = float(os.getenv("WEIGHT_VOL_ANOM", "0.25"))
WEIGHT_TRENDS   = float(os.getenv("WEIGHT_TRENDS", "0.30"))
WEIGHT_NEWS     = float(os.getenv("WEIGHT_NEWS", "0.20"))
WEIGHT_DII      = float(os.getenv("WEIGHT_DII", "0.25"))

WEIGHTS = {
    "volume_anomaly": WEIGHT_VOL_ANOM,
    "trends_breakout": WEIGHT_TRENDS,
    "news": WEIGHT_NEWS,
    "dii": WEIGHT_DII,
}
w_sum = sum(WEIGHTS.values())
if w_sum <= 0:
    logging.warning("Weights sum to 0, fallback to equal weights.")
    for k in WEIGHTS: WEIGHTS[k] = 0.25
else:
    for k in WEIGHTS:
        WEIGHTS[k] = WEIGHTS[k] / w_sum

POINTS_SCALE = 1000  # 0..1000 表示

# ──────────────────────────────────────────────────────────────────────────────
# load inputs
# ──────────────────────────────────────────────────────────────────────────────
def load_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(f"Universe CSV not found: {UNIVERSE_CSV}")
    df = pd.read_csv(UNIVERSE_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns:
        raise ValueError("universe.csv must contain 'symbol' column")
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    return df[["symbol", "name"]].dropna().reset_index(drop=True)

def data_path(kind: str, date: str) -> pathlib.Path:
    return DATA_DIR / date / f"{kind}.json"

# ──────────────────────────────────────────────────────────────────────────────
# components
# ──────────────────────────────────────────────────────────────────────────────
def comp_news(symbol: str, news_j: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """最近7日ニュース件数を log 正規化。"""
    items = (news_j or {}).get(symbol, [])
    cnt = len(items)
    # 0 -> 0, 1 -> 0.5, 3 -> ~0.79, 5 -> ~0.86, 10 -> ~0.91
    val = clip01(math.log1p(cnt) / math.log(11))  # 正規化 [0,1]
    return val, {"recent_count": cnt}

def comp_trends(symbol: str, trends_j: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    直近 2 週間の平均とその前 6 週間の平均でブレイクアウト度。
    trends.json は { "SOUN": { "series": [...(日別 0-100)] } } を想定
    """
    s = (trends_j or {}).get(symbol, {})
    series = s.get("series", [])
    if not series:
        return 0.0, {"breakout": 0.0}
    # 後半14, 前半42（合計56日 ≒ today 2m）
    tail = series[-14:]
    head = series[-56:-14] if len(series) >= 56 else series[:-14]
    m_tail = stats.fmean(tail) if tail else 0.0
    m_head = stats.fmean(head) if head else 0.0
    raw = safe_div((m_tail - m_head), (m_head + 1e-6))  # 相対増分
    # -100%～+∞ を [-1, +1] にクリップ後 0..1 に写像
    val = clip01((max(-1.0, min(1.0, raw))) * 0.5 + 0.5)
    return val, {"breakout_rel": raw, "m_tail": m_tail, "m_head": m_head}

def comp_dii(symbol: str, dii_j: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    dii.json は { "SOUN": { "score_0_1": 0.62, "recent_weeks": 4, ... } } を想定。
    なければ 0。
    """
    s = (dii_j or {}).get(symbol, {})
    if "score_0_1" in s:
        return clip01(s["score_0_1"]), {**s}
    # 互換: fast_volume フォールバック
    # recent_rvol: 直近RVOL20平均を0..1に圧縮
    rvol = s.get("recent_rvol20")
    if rvol is not None:
        # rvol=1→0.5, rvol=2→~0.73, rvol=3→~0.82
        val = clip01(1 - math.exp(-max(0.0, float(rvol))) )
        return val, {"recent_rvol20": rvol}
    return 0.0, {}

def comp_volume_anomaly(symbol: str, hist: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    60日ヒストリカルから RVOL20, zscore(RVOL20), DollarVol パーセンタイルなど。
    """
    if hist.empty:
        return 0.0, {}
    # price & volume
    hist = hist.dropna(subset=["Close", "Volume"]).copy()
    hist["dollar_vol"] = hist["Close"] * hist["Volume"]
    hist["vol_ma20"] = hist["Volume"].rolling(20).mean()
    hist["rvol20"] = hist["Volume"] / hist["vol_ma20"]
    hist["rv_ma60"] = hist["rvol20"].rolling(60).mean()

    # 今日（最新行）
    cur = hist.iloc[-1]
    rvol20 = float(cur.get("rvol20", np.nan))
    dv = float(cur.get("dollar_vol", np.nan))

    # z-score (過去60の rvol20 分布に対して)
    z = zscore([x for x in hist["rvol20"].dropna().tolist()[-60:]], rvol20 if not math.isnan(rvol20) else 0.0)

    # DollarVol 90日分位
    window = hist["dollar_vol"].dropna().tolist()[-90:]
    pr = 0.0
    if window:
        rank = sum(1 for v in window if v <= dv)
        pr = rank / len(window)  # 0..1

    # 正規化: rvol を 0..1 に圧縮 (1→0.5, 2→~0.73, 3→~0.82)
    if math.isnan(rvol20) or math.isinf(rvol20):
        rvol_val = 0.0
    else:
        rvol_val = clip01(1 - math.exp(-max(0.0, rvol20)))

    # 2 指標を合成（z は 0→0.5、+3σ→~0.95 程度になるよう圧縮）
    z_val = clip01(1 / (1 + math.exp(-0.75 * z)))  # logistic

    val = clip01(0.7 * rvol_val + 0.3 * z_val)
    return val, {
        "rvol20": rvol20 if not math.isnan(rvol20) else None,
        "z60": z,
        "pct_rank_90": pr,
        "dollar_vol": dv if not math.isnan(dv) else None,
        "eligible": bool(pr >= 0.2),  # 最低流動性
    }

# ──────────────────────────────────────────────────────────────────────────────
# prices / returns
# ──────────────────────────────────────────────────────────────────────────────
def load_prices(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    end = pd.Timestamp.utcnow().floor("D") + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=150)
    logging.info("[PX] Downloading prices %s..%s for %d syms", start.date(), end.date(), len(symbols))
    data = yf.download(symbols, start=start.date(), end=end.date(), group_by="ticker", auto_adjust=True, progress=False, threads=True)
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = None
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[sym].reset_index().rename(columns=str)
            else:
                # 単一銘柄ケース
                df = data.reset_index().rename(columns=str)
        except Exception:
            df = None
        if df is None or df.empty:
            logging.warning("[PX] empty for %s", sym)
            out[sym] = pd.DataFrame(columns=["Date","Open","High","Low","Close","Adj Close","Volume"])
        else:
            df = df.rename(columns={"Adj Close":"AdjClose"})
            out[sym] = df
    return out

def compute_returns(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"1D": None, "1W": None, "1M": None}
    p = df["Close"].astype(float)
    r1  = (p.iloc[-1] / p.iloc[-2] - 1.0) if len(p) >= 2 else None
    r5  = (p.iloc[-1] / p.iloc[-6] - 1.0) if len(p) >= 6 else None
    r20 = (p.iloc[-1] / p.iloc[-21] - 1.0) if len(p) >= 21 else None
    return {"1D": (r1*100 if r1 is not None else None),
            "1W": (r5*100 if r5 is not None else None),
            "1M": (r20*100 if r20 is not None else None)}

# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if not REPORT_DATE:
        raise RuntimeError("REPORT_DATE missing")
    since = t0()

    uni = load_universe()
    symbols = uni["symbol"].tolist()
    lap(since, "load_universe")

    dpath = data_path("dii", REPORT_DATE)
    tpath = data_path("trends", REPORT_DATE)
    npath = data_path("news", REPORT_DATE)

    dii_j    = read_json(dpath, default={})
    trends_j = read_json(tpath, default={})
    news_j   = read_json(npath, default={})
    logging.info("[IN] dii:%s trends:%s news:%s", dpath, tpath, npath)
    lap(since, "read_inputs")

    px = load_prices(symbols)
    lap(since, "download_prices")

    rows = []
    for _, r in uni.iterrows():
        sym = str(r["symbol"]).upper()
        name = str(r.get("name") or sym)

        # components
        vol_val, vol_detail = comp_volume_anomaly(sym, px.get(sym, pd.DataFrame()))
        tr_val,  tr_detail  = comp_trends(sym, trends_j)
        nw_val,  nw_detail  = comp_news(sym, news_j)
        di_val,  di_detail  = comp_dii(sym, dii_j)

        comps = {
            "volume_anomaly": vol_val,
            "trends_breakout": tr_val,
            "news": nw_val,
            "dii": di_val,
        }

        # weighted
        final01 = 0.0
        pts = 0
        for k, v in comps.items():
            w = WEIGHTS.get(k, 0.0)
            final01 += w * v
        pts = int(round(final01 * POINTS_SCALE))

        rets = compute_returns(px.get(sym, pd.DataFrame()))

        row = {
            "symbol": sym,
            "name": name,
            "final_score_0_1": final01,
            "score_pts": pts,
            "score_components": comps,
            "score_weights": WEIGHTS,
            "vol_anomaly_score": vol_val,
            "trends_breakout": tr_val,
            "news_score": nw_val,
            "news_recent_count": nw_detail.get("recent_count", 0),
            "dii_score": di_val,
            "dii_components": di_detail,
            "returns": rets,
            "deltas": { "d1": rets["1D"], "d5": rets["1W"], "d20": rets["1M"] },
            "detail": {
                "vol_anomaly": vol_detail
            }
        }

        logging.debug(
            "[ROW] %s comps=%s weights=%s final01=%.4f pts=%d rets=%s",
            sym, json.dumps(comps, ensure_ascii=False),
            json.dumps(WEIGHTS), final01, pts, rets
        )
        rows.append(row)

    # sort and top10
    rows.sort(key=lambda x: x["score_pts"], reverse=True)
    top10 = rows[:10]

    outdir = DATA_DIR / REPORT_DATE
    outdir.mkdir(parents=True, exist_ok=True)
    top10_path = outdir / "top10.json"
    latest_path = DATA_DIR / "top10" / "latest.json"
    ensure_dir(latest_path)

    with top10_path.open("w", encoding="utf-8") as f:
        json.dump(top10, f, ensure_ascii=False, indent=2)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(top10, f, ensure_ascii=False, indent=2)

    logging.info("Generated top10 for %s: %d symbols (universe=%d)", REPORT_DATE, len(top10), len(rows))
    lap(since, "all_done")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("FATAL in fetch_daily: %s", e)
        sys.exit(1)
