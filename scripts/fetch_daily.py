#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (with price-change chips)
- EOD取得: yfinance → Stooq CSV → pandas-datareader(stooq) の順でフェイルオーバー
- Volume anomaly を算出（0..1）
- Trends / Insider / News の各0..1スコアを取り込み
- "存在するコンポーネントだけ" を重み正規化して 1000点換算
- 週足チャート(3M)生成
- NEW: 価格変化(1d/1w/1m)を計算してJSONに出力

テンプレート(daily.html.j2)が読むキー:
- rank, symbol, name, score_pts, final_score_0_1
- vol_anomaly_score, insider_momo, trends_breakout, news_score, news_recent_count
- detail.vol_anomaly（詳細）
- price_change: { d1_pct, w1_pct, m1_pct }  # ％（例: 1.23 は +1.23%）
- chart_url
"""

import os, sys, json, math, pathlib, random, datetime, time, io, csv
from zoneinfo import ZoneInfo
import requests
import pandas as pd

# charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Time helpers ----------------
def prev_us_business_day(d: datetime.date) -> datetime.date:
    while d.weekday() >= 5:  # Sat/Sun
        d = d - datetime.timedelta(days=1)
    return d

def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

# ---------------- Config ----------------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

YFI_SLEEP = float(os.getenv("YFI_SLEEP", "0.4"))

# Weights
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.20"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.05"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.40"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS",    "0.35"))

FORM4_JSON   = os.getenv("FORM4_JSON",   f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",    f"{OUT_DIR}/data/news/latest.json")

# ---------------- Data providers ----------------
def yfi_eod_range(symbol, start, end):
    """
    安全第一の EOD 取得:
      1) yfinance.Ticker(...).history()
      2) Stooq CSV
      3) pandas-datareader(stooq)
    """
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
        df = t.history(start=(start_dt - datetime.timedelta(days=2)).isoformat(),
                       end=(end_dt + datetime.timedelta(days=1)).isoformat(),
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

    # 3) pandas-datareader(stooq)
    try:
        import pandas_datareader.data as web
        df = web.DataReader(symbol, "stooq",
                            start=start_dt - datetime.timedelta(days=2),
                            end=end_dt + datetime.timedelta(days=1))
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

# ---------------- Metrics ----------------
def compute_volume_anomaly(df: pd.DataFrame):
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
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
    tail = d["dollar_vol"].tail(90).dropna()
    pct_rank_90 = float((tail <= last_dv).mean()) if len(tail) >= 5 else 0.0

    vol_score = 0.6 * max(0.0, min(1.0, rvol20 / 5.0)) + 0.4 * z60_u
    vol_score = max(0.0, min(1.0, vol_score))

    return {
        "score": vol_score,
        "eligible": True,
        "rvol20": rvol20,
        "z60": z60,
        "pct_rank_90": pct_rank_90,
        "dollar_vol": last_dv,
    }

def compute_price_changes(df: pd.DataFrame):
    """
    終値ベースの％変化:
      1d = 前営業日比
      1w = 直近5本前（約1週間）
      1m = 直近21本前（約1か月）
    データ不足時は None。
    """
    if df is None or df.empty or "close" not in df.columns:
        return {"d1_pct": None, "w1_pct": None, "m1_pct": None}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    closes = d["close"].astype(float).values
    n = len(closes)
    res = {"d1_pct": None, "w1_pct": None, "m1_pct": None}

    def pct_change(cur, prev):
        if prev is None or prev == 0 or cur is None:
            return None
        return (cur / prev - 1.0) * 100.0

    cur = closes[-1] if n >= 1 else None
    prev1 = closes[-2] if n >= 2 else None
    prev5 = closes[-6] if n >= 6 else None   # 現在を含めて6本 → 1週間前は -6 の位置
    prev21 = closes[-22] if n >= 22 else None

    res["d1_pct"] = pct_change(cur, prev1)
    res["w1_pct"] = pct_change(cur, prev5)
    res["m1_pct"] = pct_change(cur, prev21)
    return res

# ---------------- IO helpers ----------------
def load_json_safe(path, default):
    try:
        p = pathlib.Path(path)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default

def ensure_dirs(date_iso):
    (pathlib.Path(OUT_DIR)/"data"/date_iso).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"charts"/date_iso).mkdir(parents=True, exist_ok=True)

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

# ---------------- Main ----------------
def main():
    report_d = datetime.date.fromisoformat(DATE)
    end_day = prev_us_business_day(report_d - datetime.timedelta(days=1))
    start_day = end_day - datetime.timedelta(days=180)
    start = start_day.isoformat()
    end_s = end_day.isoformat()

    ensure_dirs(report_d.isoformat())

    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip() for s in uni["symbol"].tolist()]

    trends = load_json_safe(TRENDS_JSON, {"items": {}}).get("items", {})
    form4  = load_json_safe(FORM4_JSON, {"items": {}}).get("items", {})
    news   = load_json_safe(NEWS_JSON,   {"items": {}}).get("items", {})

    records = []
    recent_map = {}

    for sym in symbols:
        # OHLCV
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] OHLCV failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        recent_map[sym] = df

        # Volume anomaly
        if df is None or df.empty:
            vol_detail = None
            vol_score = 0.0
        else:
            vol_detail = compute_volume_anomaly(df)
            vol_score = float(vol_detail.get("score", 0.0))

        # Price changes
        chg = compute_price_changes(df) if df is not None and not df.empty else {"d1_pct": None, "w1_pct": None, "m1_pct": None}

        # Trends / Insider / News
        tr = trends.get(sym) or {}
        trends_breakout = float(tr.get("score_0_1") or tr.get("breakout_score") or 0.0)

        f4 = form4.get(sym) or {}
        insider_momo = float(f4.get("score_30", 0.0))

        nw = news.get(sym) or {}
        news_score = float(nw.get("score_0_1", 0.0))
        news_recent = int(nw.get("recent_count", nw.get("news_recent_count", 0)))

        # weights normalize
        comps = {
            "volume_anomaly": vol_score if vol_detail is not None else 0.0,
            "insider_momo": insider_momo,
            "trends_breakout": trends_breakout,
            "news": news_score,
        }
        raw_w = {
            "volume_anomaly": W_VOL,
            "insider_momo": W_FORM4,
            "trends_breakout": W_TRENDS,
            "news": W_NEWS,
        }
        present_keys = list(comps.keys())
        wsum = sum(max(0.0, raw_w.get(k,0.0)) for k in present_keys) or 1.0
        norm_w = {k: (max(0.0, raw_w.get(k,0.0))/wsum) for k in present_keys}
        final_0_1 = sum((comps.get(k,0.0) * norm_w.get(k,0.0)) for k in present_keys)
        score_pts = int(round(final_0_1 * 1000))

        rec = {
            "symbol": sym,
            "name": uni.loc[uni["symbol"]==sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"]==sym, "theme"].values[0] if "theme" in uni.columns else "",
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,

            # components for badges
            "vol_anomaly_score": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_breakout,
            "news_score": news_score,
            "news_recent_count": news_recent,

            # NEW: price change chips
            "price_change": {
                "d1_pct": chg.get("d1_pct"),
                "w1_pct": chg.get("w1_pct"),
                "m1_pct": chg.get("m1_pct"),
            },

            # breakdown & weights
            "score_components": comps,
            "score_weights": raw_w,

            # details for UI
            "detail": {"vol_anomaly": vol_detail},
            "chart_url": f"/charts/{DATE}/{sym}.png",
        }
        records.append(rec)

    # ranking
    records.sort(key=lambda r: r.get("score_pts", 0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    # output
    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    # charts
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
