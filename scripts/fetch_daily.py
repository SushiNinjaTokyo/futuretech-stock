#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (robust, with DII and enhanced volume anomaly)
- OHLCV: yfinance.Ticker.history → Stooq CSV → pandas-datareader(stooq)
- Volume anomaly: RVOL20 + zscore(60d) + POT(heavy tail) bonus
- DII: site/data/dii/latest.json の score_0_1 を採用（フォールバックも実装）
- Trends / News: latest.json を採用
- 重みは環境変数、存在キーのみ正規化して 1000点換算
- 価格差分（1D/1W/1M）は ref(=REPORT_DATEの前営業日) 終値に対し 1/5/20本前
- 出力: site/data/<DATE>/top10.json と charts
"""

import os, sys, json, math, pathlib, random, datetime, time, io, csv
from zoneinfo import ZoneInfo
import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- time helpers ----
def prev_us_business_day(d: datetime.date) -> datetime.date:
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d

def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    if now_et.hour < 18:
        d -= datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d

# ---- config ----
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV","data/universe.csv")
OUT_DIR      = os.getenv("OUT_DIR","site")
DATE         = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE    = os.getenv("MOCK_MODE","false").lower()=="true"

YFI_SLEEP = float(os.getenv("YFI_SLEEP","0.4"))

# weights
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM","0.25"))
W_DII    = float(os.getenv("WEIGHT_DII","0.25"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS","0.30"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS","0.20"))

# inputs
DII_JSON     = os.getenv("DII_JSON",    f"{OUT_DIR}/data/dii/latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON", f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",   f"{OUT_DIR}/data/news/latest.json")

# ---- providers ----
def yfi_eod_range(symbol, start, end):
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

    # 1) yfinance (Ticker.history)
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

            # ❌ Seriesに対する or は禁止（曖昧真偽エラーの原因）
            if "close" in df.columns:
                close_series = df["close"]
            elif "adj close" in df.columns:
                close_series = df["adj close"]
            elif "adjclose" in df.columns:
                close_series = df["adjclose"]
            else:
                raise KeyError("No close/adj close column in yfinance frame")

            out = pd.DataFrame({
                "date":   pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
                "open":   pd.to_numeric(df.get("open"), errors="coerce"),
                "high":   pd.to_numeric(df.get("high"), errors="coerce"),
                "low":    pd.to_numeric(df.get("low"), errors="coerce"),
                "close":  pd.to_numeric(close_series, errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"), errors="coerce").fillna(0),
            }).dropna(subset=["close"])
            out = out[(out["date"] >= start) & (out["date"] <= end)]
            if not out.empty:
                time.sleep(YFI_SLEEP)
                return out[["date","open","high","low","close","volume"]]
    except Exception as e:
        print(f"[WARN] yfinance failed for {symbol}: {e}", file=sys.stderr)

    # 2) Stooq CSV
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        r = requests.get(url, timeout=20)
        if r.ok and "Date,Open,High,Low,Close,Volume" in r.text:
            reader = csv.DictReader(io.StringIO(r.text))
            rows=[]
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

    # 3) pandas-datareader stooq
    try:
        import pandas_datareader.data as web
        df = web.DataReader(symbol, "stooq",
                            start=start_dt - datetime.timedelta(days=7),
                            end=end_dt + datetime.timedelta(days=2))
        if df is not None and not df.empty:
            df = df.sort_index().reset_index().rename(columns={
                "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
            })
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if not df.empty:
                time.sleep(0.2)
                return df[["date","open","high","low","close","volume"]]
    except Exception as e:
        print(f"[WARN] pandas-datareader failed for {symbol}: {e}", file=sys.stderr)

    print(f"[ERROR] {symbol}: no OHLCV from all sources", file=sys.stderr)
    return pd.DataFrame()

# ---- metrics ----
def compute_volume_anomaly(df: pd.DataFrame):
    """強化版 異常出来高スコア 0..1"""
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return {"score": 0.0, "eligible": False}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce")
    if len(d) < 60:
        return {"score": 0.0, "eligible": False}

    v_today = float(d["volume"].iloc[-1])
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1])
    rvol20  = v_today / v_sma20 if v_sma20 > 0 else 0.0

    rvol_series = d["volume"] / d["volume"].rolling(20).mean()
    rvol_60 = rvol_series.tail(60).dropna()
    if len(rvol_60) < 20:
        z60 = 0.0
        tail_bonus = 0.0
    else:
        mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
        z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
        # POT風: 上位10%閾値超過度合いで 0..0.3 を加点
        q90 = float(rvol_60.quantile(0.9))
        tail_bonus = max(0.0, min(0.3, (rvol20 - q90) / max(1e-9, q90))) if q90 > 0 else 0.0

    z60_u = max(0.0, min(1.0, z60 / 3.0))
    base = 0.55 * max(0.0, min(1.0, rvol20 / 5.0)) + 0.45 * z60_u
    vol_score = max(0.0, min(1.0, base + tail_bonus))

    # Dollar volume rank（参考）
    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    pct_rank_90 = float((tail <= last_dv).mean()) if len(tail) >= 5 else 0.0

    return {
        "score": vol_score,
        "eligible": True,
        "rvol20": rvol20,
        "z60": z60,
        "pct_rank_90": pct_rank_90,
        "dollar_vol": last_dv,
    }

def get_closes_for_deltas(df: pd.DataFrame, ref_date_iso: str):
    if df is None or df.empty: return None, None, None, None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    ref_dt = pd.to_datetime(ref_date_iso)
    idx = d.index[d["date"] <= ref_dt]
    if len(idx) == 0:
        return None, None, None, None
    i_base = int(idx[-1])

    def safe_close(i):
        if i < 0 or i >= len(d): return None
        v = d.loc[i, "close"]
        try:
            v = float(v)
            return None if math.isnan(v) else v
        except Exception:
            return None

    c0   = safe_close(i_base)
    c_1  = safe_close(i_base-1)
    c_5  = safe_close(i_base-5)
    c_20 = safe_close(i_base-20)
    return c0, c_1, c_5, c_20

def pct_change(a, b):
    if a is None or b is None or b == 0: return None
    return (a/b - 1.0) * 100.0

# ---- io helpers ----
def load_json_safe(path, default):
    try:
        p = pathlib.Path(path)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default

def load_items_upper(path):
    """
    JSON の形が:
      - {"items": { "NVDA": {...}, ... }}
      - {"data": [ {"symbol":"NVDA", ...}, ... ]}
    などでも吸収して {SYMBOL_UPPER: dict} にして返す。
    """
    j = load_json_safe(path, {})
    items = j.get("items")
    if items is None:
        data = j.get("data")
        if isinstance(data, list):
            items = {}
            for row in data:
                sym = (row.get("symbol") or row.get("ticker") or "").strip().upper()
                if sym:
                    items[sym] = row
        else:
            items = {}
    # dict の場合もキーを大文字へ
    if isinstance(items, dict):
        return {str(k).strip().upper(): v for k, v in items.items()}
    return {}

def ensure_dirs(date_iso):
    (pathlib.Path(OUT_DIR)/"data"/date_iso).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"charts"/date_iso).mkdir(parents=True, exist_ok=True)

# ---- charts ----
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

# ---- main ----
def main():
    report_d = datetime.date.fromisoformat(DATE)
    end_day = prev_us_business_day(report_d)
    start_day = end_day - datetime.timedelta(days=400)
    start = start_day.isoformat()
    end_s = end_day.isoformat()

    ensure_dirs(report_d.isoformat())

    uni = pd.read_csv(UNIVERSE_CSV)
    uni["symbol"] = uni["symbol"].astype(str).str.upper().str.strip()
    symbols = uni["symbol"].tolist()
    name_by_symbol = {row["symbol"]: str(row.get("name","")) for _, row in uni.iterrows()}

    # inputs（大文字キーへ正規化）
    trends_items = load_items_upper(TRENDS_JSON)
    news_items   = load_items_upper(NEWS_JSON)
    dii_items    = load_items_upper(DII_JSON)

    print(f"[INFO] loaded items: trends={len(trends_items)} news={len(news_items)} dii={len(dii_items)}", file=sys.stderr)

    def dii_score_from_obj(o: dict) -> float:
        if not isinstance(o, dict): return 0.0
        # 優先: score_0_1（0..1）
        if "score_0_1" in o and o["score_0_1"] is not None:
            try:
                return max(0.0, min(1.0, float(o["score_0_1"])))
            except Exception:
                pass
        # 代替: score（100刻み等を想定）
        for k in ("score", "score_pct"):
            if k in o and o[k] is not None:
                try:
                    v = float(o[k])
                    if v > 1.00001:  # 0..100 を想定
                        v = v/100.0
                    return max(0.0, min(1.0, v))
                except Exception:
                    continue
        return 0.0

    records=[]
    recent_map={}

    for sym in symbols:
        # OHLCV
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] OHLCV failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        recent_map[sym] = df

        # price deltas
        d1=d5=d20=None
        if df is not None and not df.empty:
            c0,c_1,c_5,c_20 = get_closes_for_deltas(df, end_s)
            d1  = pct_change(c0, c_1)
            d5  = pct_change(c0, c_5)
            d20 = pct_change(c0, c_20)

        # volume anomaly
        if df is None or df.empty:
            vol_detail=None; vol_score=0.0
        else:
            vol_detail = compute_volume_anomaly(df)
            vol_score = float(vol_detail.get("score",0.0))

        # trends/news/dii
        tr = trends_items.get(sym) or {}
        trends_breakout = float(tr.get("score_0_1") or tr.get("breakout_score") or 0.0)

        nw = news_items.get(sym) or {}
        news_score = float(nw.get("score_0_1",0.0))
        news_recent = int(nw.get("recent_count",0))

        di = dii_items.get(sym) or {}
        dii_score = dii_score_from_obj(di)
        # 参考表示用
        dii_ratio = None
        if isinstance(di, dict) and di.get("ats_share_ratio") not in (None, ""):
            try:
                dii_ratio = float(di.get("ats_share_ratio"))
            except Exception:
                dii_ratio = None

        # weights normalize
        comps = {
            "volume_anomaly": vol_score if vol_detail is not None else 0.0,
            "dii": dii_score,
            "trends_breakout": trends_breakout,
            "news": news_score,
        }
        raw_w = {
            "volume_anomaly": W_VOL,
            "dii": W_DII,
            "trends_breakout": W_TRENDS,
            "news": W_NEWS,
        }
        keys = list(comps.keys())
        wsum = sum(max(0.0, raw_w.get(k,0.0)) for k in keys) or 1.0
        norm_w = {k:(max(0.0, raw_w.get(k,0.0))/wsum) for k in keys}

        final_0_1 = sum(comps.get(k,0.0)*norm_w.get(k,0.0) for k in keys)
        score_pts = int(round(final_0_1*1000))

        rec = {
            "symbol": sym,
            "name": name_by_symbol.get(sym, ""),
            "theme": str(uni.loc[uni["symbol"]==sym, "theme"].values[0]) if "theme" in uni.columns else "",
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,

            "vol_anomaly_score": vol_score,
            "trends_breakout": trends_breakout,
            "news_score": news_score,
            "news_recent_count": news_recent,
            "dii_score": dii_score,
            "dii_ratio": dii_ratio,

            "score_components": comps,
            "score_weights": raw_w,

            "detail": {"vol_anomaly": vol_detail, "dii": di},
            "chart_url": f"/charts/{DATE}/{sym}.png",

            "price_delta_1d": d1,
            "price_delta_1w": d5,
            "price_delta_1m": d20,
        }
        records.append(rec)

    records.sort(key=lambda r: r.get("score_pts",0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    for r in top10:
        df = recent_map.get(r["symbol"])
        try:
            if df is not None and not df.empty:
                save_chart_png_weekly_3m(r["symbol"], df, OUT_DIR, DATE)
            else:
                print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    print(f"Generated top10 for {DATE}: {len(top10)} symbols (universe={len(symbols)})")

if __name__ == "__main__":
    main()
