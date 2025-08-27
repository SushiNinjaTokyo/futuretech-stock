#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (safe fields for UI)
- Volume anomaly (yfinance)
- Insider momentum (Form 4)
- Google Trends breakout
- News composite
- Only-present weights are normalized; export always includes safe defaults so template never sees 'Undefined'.
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
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.40"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.30"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.20"))
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
            vol = random.randint(200_000, 8_000_000)
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
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1]) if len(d) >= 20 else 0.0
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
    last_dv = float(d["dollar_vol"].iloc[-1]) if not d["dollar_vol"].empty else 0.0
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
    plt.figure(figsize=(9, 4.6), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.3)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    out = pathlib.Path(out_dir)/"charts"/date_iso/f"{symbol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()

# ---------------- Small utils ----------------
def pct_rank(vals, x):
    arr = [v for v in vals if isinstance(v, (int,float))]
    if not arr: return 0.0
    arr_sorted = sorted(arr)
    import bisect
    return bisect.bisect_right(arr_sorted, x) / len(arr_sorted)

def safe_float(x, d=0.0):
    try:
        if x is None: return d
        return float(x)
    except Exception:
        return d

# ---------------- Main ----------------
def main():
    ensure_dirs()

    # 入力の読み込み
    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]

    trends_j = load_json_safe(TRENDS_JSON, {"items": {}})
    trends_items = trends_j.get("items", {})

    form4_j = load_json_safe(FORM4_JSON, {"items": {}})
    form4_items = form4_j.get("items", {})

    news_j = load_json_safe(NEWS_JSON, {"items": {}})
    news_items = news_j.get("items", {})

    # 取得期間（180日）…ボラ評価の安定性を優先
    end = datetime.date.fromisoformat(DATE)
    start = (end - datetime.timedelta(days=180)).isoformat()
    end_s = end.isoformat()

    records = []
    recent_map = {}

    # --- まずトレンド/ニュースの倍率・順位のベース（全銘柄相対）
    # trends: raw_breakout (倍率) と score_0_1 の両方に対応
    # news: composite_0_1 or score_0_1、burst_ratio / sentiment のどれが来ても拾う
    all_trends_mult = []
    all_news_comp = []
    for sym in symbols:
        tr = trends_items.get(sym) or {}
        mult = safe_float(tr.get("raw_breakout"), None)
        if mult is None or not math.isfinite(mult): mult = None
        if mult is not None: all_trends_mult.append(mult)

        nw = news_items.get(sym) or {}
        comp = nw.get("composite_0_1", nw.get("score_0_1"))
        comp = safe_float(comp, None)
        if comp is not None and 0.0 <= comp <= 1.0:
            all_news_comp.append(comp)

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

        # Insider momentum 0..1
        f4 = form4_items.get(sym) or {}
        insider_momo = safe_float(f4.get("score_30", f4.get("insider_momo", 0.0)), 0.0)
        if insider_momo < 0: insider_momo = 0.0
        if insider_momo > 1: insider_momo = 1.0

        # Trends
        tr = trends_items.get(sym) or {}
        mult = tr.get("raw_breakout")
        mult = safe_float(mult, None)
        if mult is None or not math.isfinite(mult) or mult <= 0:
            # 既存の 0..1 があれば倍率は欠損のままでもOK
            trends_0_1 = safe_float(tr.get("score_0_1"), 0.0)
            mult = None
        else:
            trends_0_1 = safe_float(tr.get("score_0_1"), 0.0)

        if all_trends_mult:
            # 倍率ランキング（倍率がない場合は0扱い）
            mult_for_rank = mult if (mult is not None and math.isfinite(mult)) else 0.0
            rank_tr = 1 + sum(1 for v in all_trends_mult if v > mult_for_rank)
        else:
            rank_tr = None

        # News (0..1 composite)
        nw = news_items.get(sym) or {}
        news_0_1 = nw.get("composite_0_1", nw.get("score_0_1"))
        news_0_1 = safe_float(news_0_1, 0.0)
        if not (0.0 <= news_0_1 <= 1.0):
            news_0_1 = 0.0

        if all_news_comp:
            rank_news = 1 + sum(1 for v in all_news_comp if v > news_0_1)
        else:
            rank_news = None

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
        present_keys = [k for k,v in comps.items() if v is not None]
        wsum = sum(max(0.0, raw_w.get(k,0.0)) for k in present_keys) or 1.0
        norm_w = {k: (max(0.0, raw_w.get(k,0.0))/wsum) for k in present_keys}

        final_0_1 = sum( (comps.get(k,0.0) * norm_w.get(k,0.0)) for k in present_keys )
        score_pts = int(round(final_0_1 * 1000))

        rec = {
            "symbol": sym,
            "name": uni.loc[uni["symbol"].str.upper()==sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"].str.upper()==sym, "theme"].values[0] if "theme" in uni.columns else "",

            "final_score_0_1": final_0_1,
            "score_pts": score_pts,

            # individual components for UI
            "vol_anomaly_score": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_0_1,
            "news_score": news_0_1,

            # compact badges (倍率 × (Rank)) for Trends/News
            "trends_multiplier": mult,        # may be None
            "trends_rank": rank_tr,           # may be None
            "news_rank": rank_news,           # may be None

            "score_components": comps,
            "score_weights": raw_w,           # UI 正規化

            "detail": {"vol_anomaly": vol},
            "chart_url": f"/charts/{DATE}/{sym}.png",
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

    print(f"Generated top10 for {DATE}: {len(top10)} symbols (universe={len(records)})")

if __name__ == "__main__":
    main()
