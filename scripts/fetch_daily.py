#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, json, math, pathlib, random, datetime, time, io, csv
from zoneinfo import ZoneInfo
import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- market date resolver (import from scripts/ or CWD) ----
try:
    from scripts.et_market_date import get_effective_market_date  # type: ignore
except Exception:
    try:
        from et_market_date import get_effective_market_date  # type: ignore
    except Exception:
        def get_effective_market_date():
            now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
            d = now_et.date()
            if now_et.hour < 18:
                d -= datetime.timedelta(days=1)
            while d.weekday() >= 5:
                d -= datetime.timedelta(days=1)
            return d

# ---------------- Config ----------------
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = (os.getenv("REPORT_DATE") and datetime.date.fromisoformat(os.getenv("REPORT_DATE"))) or get_effective_market_date()
DATE_S = DATE.isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

YFI_SLEEP = float(os.getenv("YFI_SLEEP", "0.4"))
YFI_JITTER = float(os.getenv("YFI_JITTER", "0.25"))

# Final score weights（>0 の要素のみ正規化して合算）
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.35"))
W_DII    = float(os.getenv("WEIGHT_DII",      "0.00"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",   "0.40"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS",     "0.25"))

# DII 内部の表示用スコア
W_DII_VOL    = float(os.getenv("DII_WEIGHT_VOL",    "0.30"))
W_DII_TRENDS = float(os.getenv("DII_WEIGHT_TRENDS", "0.45"))
W_DII_NEWS   = float(os.getenv("DII_WEIGHT_NEWS",   "0.25"))

TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",    f"{OUT_DIR}/data/news/latest.json")
DII_JSON     = os.getenv("DII_JSON",     f"{OUT_DIR}/data/dii/latest.json")

# ---------------- Utils ----------------
def pct_change(c0: float|None, c_prev: float|None) -> float|None:
    if c0 is None or c_prev is None or c_prev == 0 or any(map(lambda x: x!=x, [c0, c_prev])):
        return None
    return (c0 - c_prev) / c_prev * 100.0

def get_closes_for_deltas(df: pd.DataFrame, end_s: str):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d = d[d["date"] <= pd.to_datetime(end_s)]
    if d.empty:
        return None, None, None, None
    c0 = float(d["close"].iloc[-1])
    def nth_back(n):
        if len(d) > n:
            return float(d["close"].iloc[-(n+1)])
        return None
    return c0, nth_back(1), nth_back(5), nth_back(20)

def load_json(path, default=None):
    try:
        p = pathlib.Path(path)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default

def clamp01(x):
    try:
        x = float(x)
        if x != x:  # NaN
            return 0.0
        return max(0.0, min(1.0, x))
    except Exception:
        return 0.0

def load_items_map(path: str) -> dict[str, dict]:
    """
    site/data/*/latest.json → {"items": { "NVDA": {...}, ... }} 形式 も
    {"items": [ {...,"symbol":"NVDA"}, ... ]} 形式もサポート。
    返り値は {TICKER_UPPER: rec}。
    """
    j = load_json(path, default={}) or {}
    items = j.get("items") or {}
    m: dict[str, dict] = {}

    if isinstance(items, dict):
        for k, v in items.items():
            if not v: 
                continue
            # 可能ならシンボルを上書き
            sym = str(v.get("symbol") or k).upper().strip()
            m[sym] = v
    elif isinstance(items, list):
        for v in items:
            if not isinstance(v, dict):
                continue
            sym = str(v.get("symbol") or v.get("ticker") or "").upper().strip()
            name = str(v.get("name") or v.get("query") or "").strip()
            key = sym or name.upper()
            if key:
                m[key] = v
    return m

# ---------------- Vendors ----------------
def yfi_eod_range(symbol: str, start: str, end: str) -> pd.DataFrame:
    if MOCK_MODE:
        dates = pd.date_range(start=start, end=end, freq="B")
        base = 100.0 + random.Random(symbol).random()*20
        rows = []
        for d in dates:
            base *= (1.0 + random.uniform(-0.02, 0.02))
            vol = random.randint(100_000, 10_000_000)
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "open": base*0.99, "high": base*1.01, "low": base*0.98,
                         "close": base, "volume": vol})
        return pd.DataFrame(rows)

    start_dt = datetime.date.fromisoformat(start)
    end_dt   = datetime.date.fromisoformat(end)

    # 1) yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        df = t.history(start=(start_dt - datetime.timedelta(days=7)).isoformat(),
                       end=(end_dt + datetime.timedelta(days=2)).isoformat(),
                       interval="1d", auto_adjust=True)
        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = [str(c).strip().lower() for c in df.columns]
            date_col = "date" if "date" in df.columns else "index"
            close_series = df.get("close")
            if close_series is None or close_series.isna().all():
                close_series = df.get("adj close") if df.get("adj close") is not None else df.get("adjclose")
            out = pd.DataFrame({
                "date":   pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
                "open":   pd.to_numeric(df.get("open"), errors="coerce"),
                "high":   pd.to_numeric(df.get("high"), errors="coerce"),
                "low":    pd.to_numeric(df.get("low"),  errors="coerce"),
                "close":  pd.to_numeric(close_series,   errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"), errors="coerce").fillna(0),
            })
            out = out.dropna(subset=["close"])
            out = out[(out["date"] >= start) & (out["date"] <= end)]
            if not out.empty:
                return out[["date","open","high","low","close","volume"]]
    except Exception as e:
        print(f"[WARN] yfinance failed for {symbol}: {e}", file=sys.stderr)

    # 2) Stooq CSV direct
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        r = requests.get(url, timeout=20)
        if r.ok and "Date,Open,High,Low,Close,Volume" in r.text:
            reader = csv.DictReader(io.StringIO(r.text))
            rows = []
            for row in reader:
                rows.append({
                    "date": row["Date"],
                    "open": float(row["Open"] or 0),
                    "high": float(row["High"] or 0),
                    "low": float(row["Low"] or 0),
                    "close": float(row["Close"] or 0),
                    "volume": float(row["Volume"] or 0),
                })
            if rows:
                df = pd.DataFrame(rows)
                df = df[(df["date"] >= start) & (df["date"] <= end)]
                if not df.empty:
                    time.sleep(0.2)
                    return df[["date","open","high","low","close","volume"]]
    except Exception as e:
        print(f"[WARN] stooq csv failed for {symbol}: {e}", file=sys.stderr)

    # 3) pandas-datareader (stooq)
    try:
        import pandas_datareader.data as web
        df = web.DataReader(symbol, "stooq", start=start_dt - datetime.timedelta(days=3), end=end_dt + datetime.timedelta(days=1))
        if df is not None and not df.empty:
            df = df.sort_index()
            out = pd.DataFrame({
                "date":   pd.to_datetime(df.index).strftime("%Y-%m-%d"),
                "open":   pd.to_numeric(df["Open"],  errors="coerce"),
                "high":   pd.to_numeric(df["High"],  errors="coerce"),
                "low":    pd.to_numeric(df["Low"],   errors="coerce"),
                "close":  pd.to_numeric(df["Close"], errors="coerce"),
                "volume": pd.to_numeric(df["Volume"],errors="coerce").fillna(0),
            })
            out = out.dropna(subset=["close"])
            out = out[(out["date"] >= start) & (out["date"] <= end)]
            if not out.empty:
                time.sleep(0.2)
                return out[["date","open","high","low","close","volume"]]
    except Exception as e:
        print(f"[WARN] pandas-datareader stooq failed for {symbol}: {e}", file=sys.stderr)

    return pd.DataFrame(columns=["date","open","high","low","close","volume"])

# ---------------- Volume anomaly ----------------
def compute_volume_anomaly(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"score": 0.0, "eligible": False}
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce")
    if len(d) < 45:
        return {"score": 0.0, "eligible": False}

    v_today = float(d["volume"].iloc[-1])
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1])
    rvol20 = v_today / v_sma20 if v_sma20 > 0 else 0.0

    rvol_series = d["volume"] / d["volume"].rolling(20).mean()
    rvol_60 = rvol_series.tail(60).dropna()
    if len(rvol_60) < 20:
        z60 = 0.0
    else:
        mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
        z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
    z60_u = max(0.0, min(1.0, z60 / 3.0))

    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    dv90 = d["dollar_vol"].tail(90).dropna().tolist()
    if dv90:
        arr = sorted(dv90)
        import bisect
        k = bisect.bisect_right(arr, last_dv)
        pr90 = k / len(arr)
    else:
        pr90 = 0.0

    score = 0.6 * z60_u + 0.25 * max(0.0, min(1.0, rvol20 / 5.0)) + 0.15 * pr90
    eligible = (v_sma20 > 0) and math.isfinite(score)
    return {
        "score": float(score),
        "rvol20": float(rvol20),
        "z60": float(z60),
        "pct_rank_90": float(pr90),
        "dollar_vol": float(last_dv),
        "eligible": bool(eligible),
    }

# ---------------- Charts ----------------
def save_chart_png_weekly_3m(symbol, df, out_dir, date_iso):
    if df is None or df.empty: return
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    w = pd.DataFrame({
        "close": d["close"].resample("W-FRI").last(),
        "volume": d["volume"].resample("W-FRI").sum(),
    }).dropna()
    w = w.tail(13)
    plt.figure(figsize=(9, 4.6), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.3)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    out = pathlib.Path(out_dir)/"charts"/date_iso/f"{symbol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()

def ensure_dirs(date_iso):
    (pathlib.Path(OUT_DIR)/"data"/date_iso).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"charts"/date_iso).mkdir(parents=True, exist_ok=True)

# ---------------- Main ----------------
def main():
    ensure_dirs(DATE_S)

    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist() if str(s).strip()]

    # external features
    trends_map = load_items_map(TRENDS_JSON)   # {SYM: {..., score_0_1}}
    news_map   = load_items_map(NEWS_JSON)     # {SYM: {..., score_0_1, recent_count}}
    dii_map    = load_items_map(DII_JSON)      # {SYM: {..., score_0_1}} or {}

    # DIIが空なら最終スコアから自動除外
    global W_DII
    if not dii_map:
        W_DII = 0.0

    end_s   = DATE_S
    start_s = (DATE - datetime.timedelta(days=200)).isoformat()

    records = []
    recent_map = {}

    for sym in symbols:
        try:
            df = yfi_eod_range(sym, start_s, end_s)
        except Exception as e:
            print(f"[WARN] OHLCV failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        recent_map[sym] = df

        d1 = d5 = d20 = None
        if df is not None and not df.empty:
            c0, c_1, c_5, c_20 = get_closes_for_deltas(df, end_s)
            d1  = pct_change(c0, c_1)
            d5  = pct_change(c0, c_5)
            d20 = pct_change(c0, c_20)

        # volume anomaly
        if df is None or df.empty:
            vol_detail = None
            vol_score = 0.0
        else:
            vol_detail = compute_volume_anomaly(df)
            vol_score = float(vol_detail.get("score", 0.0))

        # trends/news/dii 0..1
        td = trends_map.get(sym) or {}
        trends_breakout = clamp01(td.get("score_0_1") or td.get("score"))

        nw = news_map.get(sym) or {}
        news_score = clamp01(nw.get("score_0_1"))
        news_recent = int(nw.get("recent_count") or nw.get("total_count") or 0)

        di = dii_map.get(sym) or {}
        dii_external = clamp01(di.get("score_0_1"))

        # Final score（重み>0のみ正規化）
        comps = {
            "volume_anomaly": vol_score,
            "trends_breakout": trends_breakout,
            "news": news_score,
        }
        if W_DII > 0:
            comps["dii"] = dii_external

        raw_w = {
            "volume_anomaly": W_VOL,
            "trends_breakout": W_TRENDS,
            "news": W_NEWS,
            "dii": W_DII,
        }
        present = [k for k in comps.keys() if raw_w.get(k,0) > 0]
        wsum = sum(raw_w[k] for k in present) or 1.0
        norm_w = {k: raw_w[k]/wsum for k in present}
        final_0_1 = sum(comps[k] * norm_w.get(k,0.0) for k in comps.keys())
        score_pts = int(round(final_0_1 * 1000))

        # DII内部の表示用（最終スコアとは独立）
        dii_comps = {"volume_anomaly": vol_score, "trends": trends_breakout, "news": news_score}
        dii_wsum = max(1e-9, W_DII_VOL + W_DII_TRENDS + W_DII_NEWS)
        dii_score_internal = (
            W_DII_VOL * dii_comps["volume_anomaly"] +
            W_DII_TRENDS * dii_comps["trends"] +
            W_DII_NEWS * dii_comps["news"]
        ) / dii_wsum

        rec = {
            "symbol": sym,
            "name": uni.loc[uni["symbol"]==sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"]==sym, "theme"].values[0] if "theme" in uni.columns else "",
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,

            "vol_anomaly_score": vol_score,
            "trends_breakout": trends_breakout,
            "news_score": news_score,
            "news_recent_count": news_recent,

            "dii_external": dii_external,
            "dii_score": dii_score_internal,

            "score_components": comps,
            "score_weights": raw_w,
            "detail": {"vol_anomaly": vol_detail, "dii": {"score": dii_score_internal, "components": dii_comps}},

            "deltas": {"d1": d1, "d5": d5, "d20": d20},
        }
        records.append(rec)

        time.sleep(YFI_SLEEP + random.random()*YFI_JITTER)

    # rank & output
    records.sort(key=lambda r: (-r["score_pts"], r["symbol"]))
    for i, r in enumerate(records, 1):
        r["rank"] = i

    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE_S
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    # charts
    for r in top10:
        df = recent_map.get(r["symbol"])
        try:
            if df is not None and not df.empty:
                save_chart_png_weekly_3m(r["symbol"], df, OUT_DIR, DATE_S)
            else:
                print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    print(f"Generated top10 for {DATE_S}: {len(top10)} symbols (universe={len(records)})")

if __name__ == "__main__":
    main()
