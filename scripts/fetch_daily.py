#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (robust volume fetch)
- yfinance → retry & sleep
- fallback to Stooq (pandas-datareader)
- last resort: local CSV cache (site/cache/ohlcv/{SYMBOL}.csv)
- Always merge & persist cache so next run is immune to transient outages
- If samples <45, compute a simplified volume score (still informative)
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
    # 20:00 ET までは前日を “市場日付” 扱い（HTML/データのズレ防止）
    if now_et.hour < 20:
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

# Weights
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.20"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.05"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.40"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS",    "0.35"))

FORM4_JSON   = os.getenv("FORM4_JSON",   f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",    f"{OUT_DIR}/data/news/latest.json")

# Fetch tuning
YF_RETRY = int(os.getenv("YF_RETRY", "3"))
YF_SLEEP = float(os.getenv("YF_SLEEP", "0.5"))  # per-attempt small sleep
FALLBACK_SLEEP = float(os.getenv("FALLBACK_SLEEP", "0.4"))

CACHE_DIR = pathlib.Path(OUT_DIR) / "cache" / "ohlcv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Cache helpers ----------------
def cache_path(sym: str) -> pathlib.Path:
    return CACHE_DIR / f"{sym.upper().strip()}.csv"

def load_cache(sym: str) -> pd.DataFrame:
    p = cache_path(sym)
    if p.exists():
        try:
            df = pd.read_csv(p)
            if not df.empty:
                return normalize_ohlcv_columns(df)
        except Exception:
            pass
    return pd.DataFrame()

def save_cache(sym: str, df: pd.DataFrame):
    if df is None or df.empty: return
    p = cache_path(sym)
    # merge (dedupe by date)
    old = load_cache(sym)
    if old is None or old.empty:
        merged = df.copy()
    else:
        merged = pd.concat([old, df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p, index=False)

# ---------------- Normalization ----------------
def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    # unify columns
    cols = {c.lower().strip(): c for c in d.columns}
    def pick(*names):
        for nm in names:
            if nm in cols: return cols[nm]
        return None
    date_col  = pick("date", "datetime", "index")
    open_col  = pick("open")
    high_col  = pick("high")
    low_col   = pick("low")
    close_col = pick("close", "adj close", "adjclose")
    vol_col   = pick("volume", "vol")

    if date_col is None or close_col is None or vol_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "date":   pd.to_datetime(d[date_col]).dt.strftime("%Y-%m-%d"),
        "open":   pd.to_numeric(d[open_col]  if open_col  else pd.Series(dtype="float64"), errors="coerce"),
        "high":   pd.to_numeric(d[high_col]  if high_col  else pd.Series(dtype="float64"), errors="coerce"),
        "low":    pd.to_numeric(d[low_col]   if low_col   else pd.Series(dtype="float64"), errors="coerce"),
        "close":  pd.to_numeric(d[close_col], errors="coerce"),
        "volume": pd.to_numeric(d[vol_col],   errors="coerce").fillna(0),
    })
    return out[["date","open","high","low","close","volume"]].dropna(subset=["date","close"])

# ---------------- Providers ----------------
def fetch_yfinance(sym: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)
    last_err = None
    for i in range(1, YF_RETRY+1):
        try:
            tmp = yf.download(sym, start=start_dt.isoformat(), end=end_dt.isoformat(),
                              interval="1d", auto_adjust=True, progress=False, threads=False)
            if tmp is not None and not tmp.empty:
                return normalize_ohlcv_columns(tmp.reset_index())
            # fallback to history()
            tkr = yf.Ticker(sym)
            tmp2 = tkr.history(start=start_dt.isoformat(), end=end_dt.isoformat(),
                               interval="1d", auto_adjust=True)
            if tmp2 is not None and not tmp2.empty:
                return normalize_ohlcv_columns(tmp2.reset_index())
        except Exception as e:
            last_err = e
        time.sleep(YF_SLEEP * i)  # incremental backoff
    if last_err:
        print(f"[WARN] yfinance failed for {sym}: {last_err}", file=sys.stderr)
    return pd.DataFrame()

def fetch_stooq(sym: str, start: str, end: str) -> pd.DataFrame:
    # Stooq はコード体系が違う銘柄があるが、主要米株はそのまま通ることが多い
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        print(f"[WARN] pandas-datareader missing: {e}", file=sys.stderr)
        return pd.DataFrame()
    try:
        s = datetime.date.fromisoformat(start)
        e = datetime.date.fromisoformat(end) + datetime.timedelta(days=1)
        df = pdr.DataReader(sym, "stooq", s, e)
        if df is None or df.empty:
            return pd.DataFrame()
        # stooq は降順のことがある
        df = df.sort_index().reset_index().rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        return normalize_ohlcv_columns(df)
    except Exception as e:
        print(f"[WARN] stooq failed for {sym}: {e}", file=sys.stderr)
        return pd.DataFrame()

def fetch_ohlcv(symbol, start, end):
    # MOCK
    if MOCK_MODE:
        dates = pd.date_range(start=start, end=end, freq="B")
        base = 100.0 + random.Random(symbol).random()*20
        rows = []
        for d in dates:
            base *= (1.0 + random.uniform(-0.02, 0.02))
            vol = random.randint(1_000_00, 10_000_000)
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "open": base*0.99, "high": base*1.01, "low": base*0.98,
                         "close": base, "volume": vol})
        df = pd.DataFrame(rows)
        save_cache(symbol, df)
        return df

    # 1) yfinance
    df = fetch_yfinance(symbol, start, end)
    if df is not None and not df.empty:
        save_cache(symbol, df)
        return df

    time.sleep(FALLBACK_SLEEP)

    # 2) Stooq
    df = fetch_stooq(symbol, start, end)
    if df is not None and not df.empty:
        save_cache(symbol, df)
        return df

    time.sleep(FALLBACK_SLEEP)

    # 3) cache (last resort)
    df = load_cache(symbol)
    if df is not None and not df.empty:
        # slice to requested range
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        mask = (d["date"] >= pd.to_datetime(start)) & (d["date"] <= pd.to_datetime(end))
        d = d.loc[mask].sort_values("date")
        if not d.empty:
            print(f"[INFO] {symbol}: using cached OHLCV (offline)", file=sys.stderr)
            return normalize_ohlcv_columns(d)

    print(f"[ERROR] {symbol}: no OHLCV from all sources", file=sys.stderr)
    return pd.DataFrame()

# ---------------- Metrics ----------------
def compute_volume_anomaly(df: pd.DataFrame):
    """
    Return dict with 0..1 score and details.
    If samples <45 but >=10, compute a simplified score (rvol-only) and mark eligible=False.
    """
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return {"score": 0.0, "eligible": False}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)

    n = len(d)
    if n < 10:
        return {"score": 0.0, "eligible": False}

    v_today = float(d["volume"].iloc[-1])
    v_sma20 = float(d["volume"].rolling(20, min_periods=5).mean().iloc[-1])
    rvol20 = v_today / v_sma20 if v_sma20 > 0 else 0.0

    # データ十分なら z60 まで使う
    if n >= 45:
        rvol_series = d["volume"] / d["volume"].rolling(20, min_periods=5).mean()
        rvol_60 = rvol_series.tail(60).dropna()
        if len(rvol_60) < 20:
            z60 = 0.0
        else:
            mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
            z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
        z60_u = max(0.0, min(1.0, z60 / 3.0))
        vol_score = 0.6 * max(0.0, min(1.0, rvol20 / 5.0)) + 0.4 * z60_u
        eligible = True
    else:
        # 簡易: rvol のみで評価（過大評価を避けるため弱めに圧縮）
        vol_score = max(0.0, min(1.0, rvol20 / 6.0))
        z60 = 0.0
        eligible = False

    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    pct_rank_90 = float((tail <= last_dv).mean()) if len(tail) >= 5 else 0.0

    return {
        "score": float(vol_score),
        "eligible": eligible,
        "rvol20": float(rvol20),
        "z60": float(z60),
        "pct_rank_90": float(pct_rank_90),
        "dollar_vol": float(last_dv),
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
    w = w.tail(13)

    plt.figure(figsize=(9, 4.4), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.3)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    out = pathlib.Path(out_dir)/"charts"/date_iso/f"{symbol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()

# ---------------- Main ----------------
def main():
    ensure_dirs()

    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip() for s in uni["symbol"].tolist()]

    trends = load_json_safe(TRENDS_JSON, {"items": {}}).get("items", {})
    form4  = load_json_safe(FORM4_JSON,  {"items": {}}).get("items", {})
    news   = load_json_safe(NEWS_JSON,   {"items": {}}).get("items", {})

    # 安全に長めの期間を要求（欠損でもキャッシュが補う）
    end = datetime.date.fromisoformat(DATE)
    start = (end - datetime.timedelta(days=220)).isoformat()
    end_s = end.isoformat()

    records = []
    recent_map = []

    scratch = []
    for sym in symbols:
        try:
            df = fetch_ohlcv(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] fetch_ohlcv failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()

        vol = compute_volume_anomaly(df)
        vol_score = float(vol.get("score", 0.0))

        tr = trends.get(sym) or {}
        trends_breakout = float(tr.get("score_0_1") or tr.get("breakout_score") or tr.get("raw_breakout") or 0.0)

        f4 = form4.get(sym) or {}
        insider_momo = float(f4.get("score_30", 0.0))

        nw = news.get(sym) or {}
        news_score = float(nw.get("score_0_1") or 0.0)
        news_counts = int(nw.get("count_recent") or 0)

        scratch.append({
            "symbol": sym,
            "name": uni.loc[uni["symbol"]==sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"]==sym, "theme"].values[0] if "theme" in uni.columns else "",
            "vol": vol, "vol_score": vol_score,
            "trends_breakout": trends_breakout,
            "insider_momo": insider_momo,
            "news_score": news_score,
            "news_counts": news_counts,
            "chart_url": f"/charts/{DATE}/{sym}.png",
        })

    # ランク付け（Trends / News）
    def rank_map(values_by_sym, reverse=True):
        syms_sorted = sorted(values_by_sym.items(), key=lambda kv: kv[1], reverse=reverse)
        return {s:i for i,(s,_) in enumerate(syms_sorted, start=1)}

    trends_rank = rank_map({r["symbol"]: r["trends_breakout"] for r in scratch})
    news_rank   = rank_map({r["symbol"]: r["news_score"]      for r in scratch})

    # 合成
    records = []
    for r in scratch:
        comps = {
            "volume_anomaly": r["vol_score"],
            "insider_momo": r["insider_momo"],
            "trends_breakout": r["trends_breakout"],
            "news": r["news_score"],
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

        records.append({
            "symbol": r["symbol"],
            "name": r["name"],
            "theme": r["theme"],
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,
            "vol_anomaly_score": r["vol_score"],
            "insider_momo": r["insider_momo"],
            "trends_breakout": r["trends_breakout"],
            "news_score": r["news_score"],
            "news_count": r["news_counts"],
            "score_components": comps,
            "score_weights": raw_w,
            "detail": {"vol_anomaly": r["vol"]},
            "chart_url": r["chart_url"],
            "trends_rank": trends_rank.get(r["symbol"]),
            "news_rank": news_rank.get(r["symbol"]),
            "trends_top5": (trends_rank.get(r["symbol"], 99) <= 5),
            "news_top5":   (news_rank.get(r["symbol"], 99)   <= 5),
        })

    # ランキング & 出力
    records.sort(key=lambda r: r.get("score_pts", 0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    # チャート（あってもなくてもOK：データが取れた銘柄のみ）
    for r in top10:
        # 直近キャッシュで十分なはずだが、失敗しても続行
        try:
            df_cached = load_cache(r["symbol"])
            if df_cached is not None and not df_cached.empty:
                save_chart_png_weekly_3m(r["symbol"], df_cached, OUT_DIR, DATE)
        except Exception as e:
            print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    print(f"Generated top10 for {DATE}: {len(top10)} symbols (universe={len(symbols)})")

if __name__ == "__main__":
    main()
