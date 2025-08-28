#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (robust)
- OHLCV: yfinance を多段フォールバック + ローカルキャッシュ + スロットリング
- Volume anomaly を 0..1 に算出
- Google Trends (site/data/trends/latest.json) を 0..1 で反映
- Insider momentum (site/data/insider/form4_latest.json) を 0..1 で反映
- News (site/data/news/latest.json) を 0..1 で反映
- 重みは環境変数（存在する項目だけ正規化）→ 1000点に換算
- テンプレートが読む JSON に必須キーを必ず出力
"""

import os, sys, json, math, pathlib, random, datetime, time
from zoneinfo import ZoneInfo
import pandas as pd

# charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Time helpers ----------------
def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # ET 18:00 までは前営業日扱い（安全寄り）
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun を直近平日に寄せる
        d = d - datetime.timedelta(days=1)
    return d

# ---------------- Config ----------------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Weights (raw). Only-present components will be normalized.
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.60"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.20"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.15"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS",    "0.05"))

FORM4_JSON   = os.getenv("FORM4_JSON",   f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",    f"{OUT_DIR}/data/news/latest.json")

# yfinance robustness
YF_RETRIES = int(os.getenv("YF_RETRIES", "3"))
YF_SLEEP   = float(os.getenv("YF_SLEEP", "0.6"))

# cache
CACHE_DIR  = pathlib.Path(OUT_DIR) / "cache" / "ohlcv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cache_path(sym: str) -> pathlib.Path:
    return CACHE_DIR / f"{sym.upper()}.parquet"

def load_cache(sym: str) -> pd.DataFrame:
    p = cache_path(sym)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            # 安全化
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_cache(sym: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    try:
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"])
        df2 = df2.drop_duplicates(subset=["date"]).sort_values("date")
        df2.to_parquet(cache_path(sym), index=False)
    except Exception:
        pass

# ---------------- Data providers ----------------
def yfi_eod_range(symbol, start, end):
    """
    多段フォールバック:
      1) download(start,end)
      2) Ticker().history(start,end)
      3) download(period="9mo")
      4) cache
    常に正規化して [date, open, high, low, close, volume] を返す
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
        df = pd.DataFrame(rows)
        save_cache(symbol, df)
        return df

    import yfinance as yf
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=3)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    def _normalize(tmp: pd.DataFrame) -> pd.DataFrame:
        if tmp is None or tmp.empty: return pd.DataFrame()
        df = tmp.reset_index()
        df.columns = [str(c).strip().lower() for c in df.columns]
        name_map = {"date":"date","open":"open","high":"high","low":"low",
                    "close":"close","adj close":"close","adjclose":"close","volume":"volume"}
        out = {}
        for c in df.columns:
            if c in name_map and name_map[c] not in out:
                out[name_map[c]] = df[c]
        if "date" not in out or "close" not in out or "volume" not in out:
            return pd.DataFrame()
        res = pd.DataFrame({
            "date":   pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d"),
            "open":   pd.to_numeric(out.get("open",   pd.Series(dtype="float64")), errors="coerce"),
            "high":   pd.to_numeric(out.get("high",   pd.Series(dtype="float64")), errors="coerce"),
            "low":    pd.to_numeric(out.get("low",    pd.Series(dtype="float64")), errors="coerce"),
            "close":  pd.to_numeric(out.get("close",  pd.Series(dtype="float64")), errors="coerce"),
            "volume": pd.to_numeric(out.get("volume", pd.Series(dtype="float64")), errors="coerce").fillna(0),
        })
        return res[["date","open","high","low","close","volume"]]

    df = pd.DataFrame()
    src = "empty"
    last_err = None

    # 1) download(start/end) with retries
    for i in range(1, YF_RETRIES+1):
        try:
            tmp = yf.download(
                symbol,
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            df = _normalize(tmp)
            if not df.empty:
                src = "download-range"; break
        except Exception as e:
            last_err = e
        time.sleep(YF_SLEEP * i)

    # 2) history(start/end)
    if df.empty:
        try:
            tkr = yf.Ticker(symbol)
            tmp = tkr.history(
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                interval="1d",
                auto_adjust=True
            )
            df = _normalize(tmp)
            if not df.empty:
                src = "history-range"
        except Exception as e:
            last_err = e

    # 3) download(period="9mo")
    if df.empty:
        for i in range(1, YF_RETRIES+1):
            try:
                tmp = yf.download(
                    symbol, period="9mo", interval="1d",
                    auto_adjust=True, progress=False, threads=False
                )
                df = _normalize(tmp)
                if not df.empty:
                    src = "download-period"; break
            except Exception as e:
                last_err = e
            time.sleep(YF_SLEEP * i)

    # 4) cache
    if df.empty:
        df_cache = load_cache(symbol)
        if not df_cache.empty:
            df = df_cache
            src = "cache"

    # merge & save cache
    if not df.empty:
        old = load_cache(symbol)
        if not old.empty:
            old["date"] = pd.to_datetime(old["date"])
            df["date"]  = pd.to_datetime(df["date"])
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        save_cache(symbol, df)

    # trace
    if src != "cache" and df.empty and last_err:
        print(f"[WARN] yfi empty {symbol} src={src} err={last_err}", file=sys.stderr)
    else:
        print(f"[INFO] yfi {symbol} src={src} rows={len(df)}", file=sys.stderr)

    return df

# ---------------- Metrics ----------------
def compute_volume_anomaly(df: pd.DataFrame):
    """Return dict with 0..1 score and details. If not enough data, score=0."""
    base = {
        "score": 0.0, "eligible": False,
        "rvol20": None, "z60": None, "pct_rank_90": None, "dollar_vol": None
    }
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return base

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce").fillna(0)
    if len(d) < 45:  # 少なくとも約2か月分
        return base

    v_today = float(d["volume"].iloc[-1])

    # RVOL20
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1])
    rvol20 = v_today / v_sma20 if v_sma20 > 0 else 0.0

    # z-score vs past 60d of RVOL
    rvol_series = d["volume"] / d["volume"].rolling(20).mean()
    rvol_60 = rvol_series.tail(60).dropna()
    if len(rvol_60) < 20:
        z60 = 0.0
    else:
        mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
        z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
    z60_u = max(0.0, min(1.0, z60 / 3.0))  # 3σ ≒ 1.0

    # PctRank(90d) by dollar volume（参考値）
    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1]) if not d["dollar_vol"].empty else 0.0
    tail = d["dollar_vol"].tail(90).dropna()
    if len(tail) >= 5:
        pct_rank_90 = float((tail <= last_dv).mean())
    else:
        pct_rank_90 = None

    # 大型株バイアスを避ける → “比率”中心
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

# ---------------- IO helpers ----------------
def load_json_safe(path, default):
    try:
        p = pathlib.Path(path)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default

def ensure_dirs():
    (pathlib.Path(OUT_DIR)/"data"/DATE).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"charts"/DATE).mkdir(parents=True, exist_ok=True)

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
    if w.empty: return
    w = w.tail(13)  # 約3か月

    plt.figure(figsize=(9, 4.6), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.3)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    out = pathlib.Path(out_dir)/"charts"/date_iso/f"{symbol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()

# ---------------- Main ----------------
def pick_trends_score(rec: dict) -> float:
    # fetch_trends.py の保存形式に広く対応
    if not isinstance(rec, dict): return 0.0
    for k in ("score_0_1", "breakout_score", "score", "signal_0_1"):
        v = rec.get(k)
        if isinstance(v, (int,float)):
            try: return float(v)
            except Exception: pass
    return 0.0

def pick_news_score(rec: dict) -> float:
    # fetch_news.py の保存形式に広く対応
    if not isinstance(rec, dict): return 0.0
    for k in ("score_0_1", "signal_0_1", "score", "news_score"):
        v = rec.get(k)
        if isinstance(v, (int,float)):
            try: return float(v)
            except Exception: pass
    return 0.0

def main():
    ensure_dirs()

    # 入力の読み込み
    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]

    trends = load_json_safe(TRENDS_JSON, {"items": {}})
    trends_items = trends.get("items", {})

    form4 = load_json_safe(FORM4_JSON, {"items": {}})
    form4_items = form4.get("items", {})

    news = load_json_safe(NEWS_JSON, {"items": {}})
    news_items = news.get("items", {})

    # 取得期間（約6か月ぶん）
    end = datetime.date.fromisoformat(DATE)
    start = (end - datetime.timedelta(days=180)).isoformat()
    end_s = end.isoformat()

    records = []
    recent_map = {}

    for sym in symbols:
        # 価格/出来高
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] yfi failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()

        recent_map[sym] = df

        vol = compute_volume_anomaly(df)
        vol_score = float(vol.get("score", 0.0))

        # Trends 0..1
        tr = trends_items.get(sym) or {}
        trends_breakout = pick_trends_score(tr)

        # Insider 0..1
        f4 = form4_items.get(sym) or {}
        insider_momo = 0.0
        for k in ("score_30", "insider_momo", "score_90"):
            v = f4.get(k)
            if isinstance(v, (int,float)):
                insider_momo = float(v); break

        # News 0..1
        nw = news_items.get(sym) or {}
        news_score = pick_news_score(nw)

        # --- weights normalization (present-only) ---
        comps = {
            "volume_anomaly": vol_score,
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
        present_keys = [k for k,v in comps.items() if v is not None]
        wsum = sum(max(0.0, raw_w.get(k,0.0)) for k in present_keys) or 1.0
        norm_w = {k: (max(0.0, raw_w.get(k,0.0))/wsum) for k in present_keys}

        final_0_1 = sum( (comps.get(k,0.0) * norm_w.get(k,0.0)) for k in present_keys )
        score_pts = int(round(final_0_1 * 1000))

        name = ""
        theme = ""
        if "name" in uni.columns:
            try: name = str(uni.loc[uni["symbol"]==sym, "name"].values[0])
            except Exception: name = ""
        if "theme" in uni.columns:
            try: theme = str(uni.loc[uni["symbol"]==sym, "theme"].values[0])
            except Exception: theme = ""

        rec = {
            "symbol": sym,
            "name": name,
            "theme": theme,
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,
            "vol_anomaly_score": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_breakout,
            "news_score": news_score,
            "score_components": comps,
            "score_weights": raw_w,  # JS 側で present-only 正規化
            "detail": {"vol_anomaly": vol},
            "chart_url": f"/charts/{DATE}/{sym}.png",
        }
        records.append(rec)

        # polite to Yahoo
        time.sleep(YF_SLEEP)

    # ランキング
    records.sort(key=lambda r: r.get("score_pts", 0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    # JSON 出力
    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    # チャート（存在する df のみ）
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
