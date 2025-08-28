#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter
- Price/Volume (yfinance) → Volume anomaly 0..1
- Google Trends → breakout score (0..1) from site/data/trends/latest.json
- Insider momentum → 0..1 from site/data/insider/form4_latest.json
- News score → 0..1 from site/data/news/latest.json
- 各コンポーネントは「存在する要素のみ」を正規化重みで合算し、1000点換算
- 併せて Trends / News のランキングと TOP5 フラグを付与（UIで強調表示）
"""

import os, sys, json, math, pathlib, random, datetime
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

# Weights (raw). Only-present components will be normalized.
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.60"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.20"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.10"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS",    "0.10"))

FORM4_JSON   = os.getenv("FORM4_JSON",   f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",    f"{OUT_DIR}/data/news/latest.json")

# ---------------- Data providers ----------------
def yfi_eod_range(symbol, start, end):
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
        return pd.DataFrame(rows)

    import yfinance as yf
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    df = None
    for _ in range(2):
        tmp = yf.download(symbol, start=start_dt.isoformat(), end=end_dt.isoformat(),
                          interval="1d", auto_adjust=True, progress=False, threads=False)
        if tmp is not None and not tmp.empty:
            df = tmp; break
    if df is None or df.empty:
        tkr = yf.Ticker(symbol)
        tmp = tkr.history(start=start_dt.isoformat(), end=end_dt.isoformat(),
                          interval="1d", auto_adjust=True)
        if tmp is None or tmp.empty:
            return pd.DataFrame()
        df = tmp

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

# ---------------- Metrics ----------------
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
    if len(tail) >= 5:
        pct_rank_90 = float((tail <= last_dv).mean())
    else:
        pct_rank_90 = 0.0

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

    end = datetime.date.fromisoformat(DATE)
    start = (end - datetime.timedelta(days=180)).isoformat()
    end_s = end.isoformat()

    records = []
    recent_map = {}

    # まず全銘柄の素点を集める（後でランキング化）
    scratch = []
    for sym in symbols:
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] yfi failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        recent_map[sym] = df

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
        })

    # ランク付け（Trends / News は UI でTOP5強調）
    def rank_map(values_by_sym, reverse=True):
        syms_sorted = sorted(values_by_sym.items(), key=lambda kv: kv[1], reverse=reverse)
        rmap = {}
        for i,(s,v) in enumerate(syms_sorted, start=1):
            rmap[s] = i
        return rmap

    trends_vals = {r["symbol"]: r["trends_breakout"] for r in scratch}
    news_vals   = {r["symbol"]: r["news_score"]      for r in scratch}

    trends_rank = rank_map(trends_vals, reverse=True)
    news_rank   = rank_map(news_vals,   reverse=True)

    # レコード組み立て（スコア合成 → 1000点、ランキング/Top5フラグ付与）
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

        rec = {
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
            "chart_url": f"/charts/{DATE}/{r['symbol']}.png",
            # component ranks / top5
            "trends_rank": trends_rank.get(r["symbol"]),
            "news_rank": news_rank.get(r["symbol"]),
            "trends_top5": (trends_rank.get(r["symbol"], 99) <= 5),
            "news_top5":   (news_rank.get(r["symbol"], 99)   <= 5),
        }
        records.append(rec)

    # ランキング
    records.sort(key=lambda r: r.get("score_pts", 0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    # JSON 出力
    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    # チャート
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
