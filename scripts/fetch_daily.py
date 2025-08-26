#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (robust)
- yfinance を優先、空や短すぎる場合は Stooq に自動フォールバック
- 末尾の出来高ゼロ行（未確定/休場）を自動ドロップして指標算出
- Google Trends は score_0_1 を優先キーとして合算に必ず参加
- Insider momentum(Form 4) も 0..1 で合算
- コンポーネントは「存在するもの」で重み正規化し 1000 点に換算
"""

import os, sys, json, math, pathlib, random, datetime, io
from zoneinfo import ZoneInfo
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Time helpers ----------------
def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 引け後 18:00 未満は前営業日に倒す
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:  # 土日を金曜へ
        d = d - datetime.timedelta(days=1)
    return d

# ---------------- Config ----------------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Weights（raw）: 存在するキーのみ正規化
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.60"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.20"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.20"))

FORM4_JSON   = os.getenv("FORM4_JSON",   f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")

# ---------------- Providers ----------------
def _mock_df(symbol, start, end):
    dates = pd.date_range(start=start, end=end, freq="B")
    base = 100.0 + random.Random(symbol).random()*20
    rows = []
    for d in dates:
        base *= (1.0 + random.uniform(-0.02, 0.02))
        vol = random.randint(100_000, 5_000_000)
        rows.append({"date": d.strftime("%Y-%m-%d"),
                     "open": base*0.99, "high": base*1.01, "low": base*0.98,
                     "close": base, "volume": vol})
    return pd.DataFrame(rows)

def yfi_eod_range(symbol, start, end):
    if MOCK_MODE:
        return _mock_df(symbol, start, end)

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

def stooq_eod_range(symbol, start, end):
    """
    無償の Stooq CSV にフォールバック（例: nvda.us）
    """
    code = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={code}&i=d"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty:
            return pd.DataFrame()
        df.columns = [c.strip().lower() for c in df.columns]
        # stooq: date, open, high, low, close, volume
        if not {"date","open","high","low","close","volume"}.issubset(df.columns):
            return pd.DataFrame()
        df = df.dropna(subset=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        # 期間で絞る
        s = datetime.date.fromisoformat(start)
        e = datetime.date.fromisoformat(end)
        df_dt = pd.to_datetime(df["date"])
        m = (df_dt.dt.date >= s) & (df_dt.dt.date <= e)
        df = df.loc[m]
        # 数値化
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df[["date","open","high","low","close","volume"]]
    except Exception:
        return pd.DataFrame()

# ---------------- Metrics ----------------
def _drop_trailing_zero_volume(d: pd.DataFrame) -> pd.DataFrame:
    """
    末尾が出来高ゼロ（未確定/休場）なら落としてから使う
    """
    if d is None or d.empty:
        return d
    d2 = d.copy()
    d2["volume"] = pd.to_numeric(d2["volume"], errors="coerce").fillna(0)
    d2 = d2.sort_values("date")
    while len(d2) > 0 and (pd.isna(d2["volume"].iloc[-1]) or float(d2["volume"].iloc[-1]) <= 0):
        d2 = d2.iloc[:-1]
    return d2

def compute_volume_anomaly(df: pd.DataFrame):
    """Return dict with 0..1 score and details. If not enough data, score=0."""
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return {"score": 0.0, "eligible": False}

    d = _drop_trailing_zero_volume(df)
    if d is None or d.empty:
        return {"score": 0.0, "eligible": False}

    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d["close"], errors="coerce")

    if len(d) < 45:  # 少なくとも約2か月分は欲しい
        return {"score": 0.0, "eligible": False}

    # 直近日（末尾ゼロは既に除外済み）
    v_today = float(d["volume"].iloc[-1])

    # RVOL20
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1])
    rvol20 = v_today / v_sma20 if v_sma20 > 0 else 0.0

    # RVOL (過去60日) の z-score
    rvol_series = d["volume"] / d["volume"].rolling(20).mean()
    rvol_60 = rvol_series.tail(60).dropna()
    if len(rvol_60) < 20:
        z60 = 0.0
    else:
        mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
        z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
    z60_u = max(0.0, min(1.0, z60 / 3.0))  # 3σ=1 でクリップ

    # PctRank(90d) by dollar volume（情報表示用）
    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    pct_rank_90 = float((tail <= last_dv).mean()) if len(tail) >= 5 else 0.0

    # 比率中心の合成
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
    }).dropna().tail(13)  # 約3か月
    plt.figure(figsize=(9, 4.6), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.3)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    out = pathlib.Path(out_dir)/"charts"/date_iso/f"{symbol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out); plt.close()

# ---------------- Main ----------------
def main():
    ensure_dirs()

    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]

    trends = load_json_safe(TRENDS_JSON, {"items": {}})
    trends_items = trends.get("items", {}) or {}
    form4 = load_json_safe(FORM4_JSON, {"items": {}})
    form4_items = form4.get("items", {}) or {}

    end = datetime.date.fromisoformat(DATE)
    start = (end - datetime.timedelta(days=180)).isoformat()
    end_s = end.isoformat()

    records, recent_map = [], {}

    for sym in symbols:
        # ---- 価格/出来高：Yahoo → Stooq フォールバック
        df_y = pd.DataFrame()
        try:
            df_y = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] yfinance failed {sym}: {e}", file=sys.stderr)
        need_fallback = (df_y is None) or df_y.empty or (len(df_y) < 45)

        df = df_y
        df_s = pd.DataFrame()
        if need_fallback:
            try:
                df_s = stooq_eod_range(sym, start, end_s)
            except Exception as e:
                print(f"[WARN] stooq failed {sym}: {e}", file=sys.stderr)
            if df_s is not None and not df_s.empty and len(df_s) >= 45:
                df = df_s

        recent_map[sym] = df

        # ---- 異常出来高
        vol = compute_volume_anomaly(df)
        vol_score = float(vol.get("score", 0.0))

        # ---- Trends（score_0_1 を最優先。なければ他候補にフォールバック）
        tr = trends_items.get(sym) or {}
        tb = tr.get("score_0_1", tr.get("breakout_score", tr.get("score", tr.get("z", 0.0))))
        try:
            trends_breakout = float(tb or 0.0)
        except Exception:
            trends_breakout = 0.0
        trends_breakout = max(0.0, min(1.0, trends_breakout))

        # ---- Insider momentum
        f4 = form4_items.get(sym) or {}
        try:
            insider_momo = float(f4.get("score_30", 0.0))
        except Exception:
            insider_momo = 0.0
        insider_momo = max(0.0, min(1.0, insider_momo))

        # ---- 重み正規化（存在しているキーで正規化。0 は存在扱い）
        comps = {
            "volume_anomaly": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_breakout,
        }
        raw_w = {
            "volume_anomaly": W_VOL,
            "insider_momo": W_FORM4,
            "trends_breakout": W_TRENDS,
        }
        keys = list(comps.keys())
        wsum = sum(max(0.0, raw_w.get(k, 0.0)) for k in keys) or 1.0
        norm_w = {k: (max(0.0, raw_w.get(k,0.0))/wsum) for k in keys}

        final_0_1 = sum((float(comps.get(k, 0.0)) * norm_w.get(k, 0.0)) for k in keys)
        score_pts = int(round(final_0_1 * 1000))

        # ---- デバッグログ（原因切り分け用）
        len_y = len(df_y) if (df_y is not None and not df_y.empty) else 0
        len_s = len(df_s) if (df_s is not None and not df_s.empty) else 0
        last_day = "-"
        try:
            if df is not None and not df.empty:
                last_day = str(pd.to_datetime(df["date"]).iloc[-1].date())
        except Exception:
            pass
        print(
            f"[DATA] {sym}: yfi_rows={len_y} stooq_rows={len_s} picked={len(df) if df is not None else 0} "
            f"last={last_day}"
        )
        print(
            f"[SCORE] {sym}: vol={vol_score:.3f} form4={insider_momo:.3f} trends={trends_breakout:.3f} "
            f"-> {score_pts} pts"
        )

        rec = {
            "symbol": sym,
            "name": uni.loc[uni["symbol"]==sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"]==sym, "theme"].values[0] if "theme" in uni.columns else "",
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,
            "vol_anomaly_score": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_breakout,
            "score_components": comps,
            "score_weights": raw_w,  # （JS 側で present-only 正規化表示）
            "detail": {"vol_anomaly": vol},
            "chart_url": f"/charts/{DATE}/{sym}.png",
        }
        records.append(rec)

    # ---- ランキング & 出力
    records.sort(key=lambda r: r.get("score_pts", 0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    # ---- チャート
    for r in top10:
        df = recent_map.get(r["symbol"])
        try:
            if df is not None and not df.empty:
                save_chart_png_weekly_3m(r["symbol"], df, OUT_DIR, DATE)
            else:
                print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    print(f"Generated top10 for {DATE}: {len(top10)} symbols")

if __name__ == "__main__":
    main()
