#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter
- OHLCV (yfinance) から Volume anomaly を 0..1 に算出
- Google Trends (site/data/{DATE}/trends.json → latest.json) を 0..1 で反映
- Insider momentum (site/data/{DATE}/insider.json → latest.json) を 0..1 で反映
- News (site/data/{DATE}/news.json → latest.json) を 0..1 で反映
- 重みは環境変数で指定し、「存在するコンポーネントのみ」正規化して 1000点化
- テンプレートが読む JSON に必須キーを必ず出力
"""

import os, sys, json, math, pathlib, random, datetime
from zoneinfo import ZoneInfo
import pandas as pd

# charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Time helpers ----------
def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    cutoff_h = int(os.getenv("MARKET_CUTOFF_HOUR_ET", "20"))
    if now_et.hour < cutoff_h:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

# ---------- Config ----------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Weights（生値。存在するキーのみで後段正規化）
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.60"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.20"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.15"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS",    "0.05"))

# Input json paths（date-first→latest fallback）
FORM4_JSON_DATE   = os.getenv("FORM4_JSON_DATE",   f"{OUT_DIR}/data/{DATE}/insider.json")
FORM4_JSON_LATEST = os.getenv("FORM4_JSON",        f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON_DATE  = os.getenv("TRENDS_JSON_DATE",  f"{OUT_DIR}/data/{DATE}/trends.json")
TRENDS_JSON_LATEST= os.getenv("TRENDS_JSON",       f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON_DATE    = os.getenv("NEWS_JSON_DATE",    f"{OUT_DIR}/data/{DATE}/news.json")
NEWS_JSON_LATEST  = os.getenv("NEWS_JSON",         f"{OUT_DIR}/data/news/latest.json")

# ---------- Data providers ----------
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

    import yfinance as yf
    # end は排他的なので +1日、start は余裕を持って
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=3)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    df = None
    for _ in range(2):
        tmp = yf.download(symbol, start=start_dt.isoformat(), end=end_dt.isoformat(),
                          interval="1d", auto_adjust=True, progress=False, threads=False)
        if tmp is not None and not tmp.empty:
            df = tmp; break
    if df is None or df.empty:
        try:
            tkr = yf.Ticker(symbol)
            tmp = tkr.history(start=start_dt.isoformat(), end=end_dt.isoformat(),
                              interval="1d", auto_adjust=True)
            if tmp is not None and not tmp.empty:
                df = tmp
        except Exception:
            pass
        if df is None or df.empty:
            return pd.DataFrame()

    df = df.reset_index()
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

# ---------- Metrics ----------
def compute_volume_anomaly(df: pd.DataFrame):
    """Return dict with 0..1 score and details. If not enough data, score=0."""
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return {"score": 0.0, "eligible": False}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    if len(d) < 45:
        return {"score": 0.0, "eligible": False}

    # 最新⽇（REPORT_DATE 当⽇分が未確定でも "前営業⽇" を採っているので空になりにくい）
    v_today = float(d["volume"].iloc[-1])

    # RVOL20
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1])
    rvol20 = v_today / v_sma20 if v_sma20 > 0 else 0.0

    # RVOL の60⽇Z
    rvol_series = d["volume"] / d["volume"].rolling(20).mean()
    rvol_60 = rvol_series.tail(60).dropna()
    if len(rvol_60) < 20:
        z60 = 0.0
    else:
        mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
        z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
    z60_u = max(0.0, min(1.0, z60 / 3.0))

    # PctRank(90d) by dollar volume（参考値）
    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    if len(tail) >= 5:
        pct_rank_90 = float((tail <= last_dv).mean())
    else:
        pct_rank_90 = 0.0

    # スコア（比率中心）
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

# ---------- IO helpers ----------
def load_json_first(paths, default):
    for p in paths:
        try:
            fp = pathlib.Path(p)
            if fp.exists():
                return json.loads(fp.read_text())
        except Exception:
            pass
    return default

def ensure_dirs():
    (pathlib.Path(OUT_DIR)/"data"/DATE).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"charts"/DATE).mkdir(parents=True, exist_ok=True)

# ---------- Charts ----------
def save_chart_png_weekly_3m(symbol, df, out_dir, date_iso):
    if df is None or df.empty: return
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    w = pd.DataFrame({
        "close": d["close"].resample("W-FRI").last(),
        "volume": d["volume"].resample("W-FRI").sum(),
    }).dropna()
    w = w.tail(13)  # ≒3M
    if w.empty: return
    plt.figure(figsize=(9, 4.6), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.3)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    out = pathlib.Path(out_dir)/"charts"/date_iso/f"{symbol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()

# ---------- Main ----------
def main():
    ensure_dirs()

    # 入力の読み込み
    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]

    # date-first → latest fallback
    trends = load_json_first([TRENDS_JSON_DATE, TRENDS_JSON_LATEST], {"items": {}})
    form4  = load_json_first([FORM4_JSON_DATE, FORM4_JSON_LATEST], {"items": {}})
    news   = load_json_first([NEWS_JSON_DATE,   NEWS_JSON_LATEST],   {"items": {}})

    trends_items = trends.get("items", {})
    form4_items  = form4.get("items", {})
    news_items   = news.get("items", {})

    # 取得期間（180⽇）
    end = datetime.date.fromisoformat(DATE)
    start = (end - datetime.timedelta(days=180)).isoformat()
    end_s = end.isoformat()

    records = []
    recent_map = {}

    for sym in symbols:
        # OHLCV
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] yfi failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        recent_map[sym] = df

        # Volume anomaly
        vol = compute_volume_anomaly(df)
        vol_score = float(vol.get("score", 0.0))

        # Trends 0..1
        tr = (trends_items.get(sym) or {})
        # 互換: score_0_1 優先、次に breakout_score/score 等
        trends_0_1 = (
            tr.get("score_0_1",
                   tr.get("breakout_score",
                          tr.get("score", 0.0)))
        )
        try:
            trends_0_1 = float(trends_0_1 or 0.0)
        except Exception:
            trends_0_1 = 0.0

        # Insider 0..1
        f4 = (form4_items.get(sym) or {})
        insider_momo = f4.get("score_30", f4.get("insider_momo", 0.0))
        try:
            insider_momo = float(insider_momo or 0.0)
        except Exception:
            insider_momo = 0.0

        # News 0..1
        nw = (news_items.get(sym) or {})
        # 互換: score_0_1 優先、なければ news_score_0_1 / sentiment_score_0_1
        news_0_1 = nw.get("score_0_1",
                          nw.get("news_score_0_1",
                                 nw.get("sentiment_score_0_1", 0.0)))
        try:
            news_0_1 = float(news_0_1 or 0.0)
        except Exception:
            news_0_1 = 0.0

        # --- weights normalization (only-present) ---
        comps = {
            "volume_anomaly": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_0_1,
            "news": news_0_1,
        }
        raw_w = {
            "volume_anomaly": W_VOL,
            "insider_momo": W_FORM4,
            "trends_breakout": W_TRENDS,
            "news": W_NEWS,
        }
        present_keys = [k for k, v in comps.items() if v is not None]
        wsum = sum(max(0.0, raw_w.get(k, 0.0)) for k in present_keys) or 1.0
        norm_w = {k: (max(0.0, raw_w.get(k, 0.0)) / wsum) for k in present_keys}

        final_0_1 = sum((float(comps.get(k, 0.0) or 0.0) * norm_w.get(k, 0.0)) for k in present_keys)
        score_pts = int(round(final_0_1 * 1000))

        rec = {
            "symbol": sym,
            "name": uni.loc[uni["symbol"] == sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"] == sym, "theme"].values[0] if "theme" in uni.columns else "",
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,
            "vol_anomaly_score": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_0_1,
            "news_score": news_0_1,
            "score_components": comps,
            "score_weights": raw_w,  # JS側で present-only 正規化
            "detail": {"vol_anomaly": vol},
            "chart_url": f"/charts/{DATE}/{sym}.png",
        }
        records.append(rec)

    # ランキング
    records.sort(key=lambda r: r.get("score_pts", 0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    # JSON 出力
    out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir / "top10.json").write_text(json.dumps(top10, indent=2))

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
