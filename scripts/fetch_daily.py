#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (robust)
- yfinance から 価格/出来高を取得（endは必ず DATE+1日にして当日分を含める）
- 取得失敗対策：download→history→periodベース の順にフォールバック
- Volume anomaly を 0..1 で算出（大型株バイアスを避ける比率中心）
- Insider momentum (Form 4), Google Trends breakout を 0..1 で合算
- “存在するコンポーネントだけ” 重みを正規化して 1000点化
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
def et_business_date_of(date_iso: str) -> datetime.date:
    d = datetime.date.fromisoformat(date_iso)
    # 週末なら直近金曜へ
    while d.weekday() >= 5:
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
DATE_RAW = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
DATE = et_business_date_of(DATE_RAW).isoformat()

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Weights (raw). Only-present components will be normalized.
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.60"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.20"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.20"))

FORM4_JSON   = os.getenv("FORM4_JSON",   f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")

# yfinanceの控えめスリープ（429/空返り回避）
YF_SLEEP = float(os.getenv("YF_SLEEP", "0.25"))
YF_JITTER = float(os.getenv("YF_JITTER", "0.15"))

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

# ---------------- Data providers ----------------
def yfi_eod_range(symbol, start_iso, end_inclusive_iso):
    """
    Yahooは end 非包含なので end = DATE+1 を渡す。ここでは end_inclusive_iso を受け取り、
    実際のAPIには end_exclusive = end_inclusive + 1日 を使う。
    """
    if MOCK_MODE:
        dates = pd.date_range(start=start_iso, end=end_inclusive_iso, freq="B")
        base = 100.0 + random.Random(symbol).random()*20
        rows = []
        for d in dates:
            base *= (1.0 + random.uniform(-0.02, 0.02))
            vol = random.randint(100_000, 5_000_000)
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "open": base*0.99, "high": base*1.01, "low": base*0.98,
                         "close": base, "volume": vol})
        return pd.DataFrame(rows)

    import yfinance as yf
    start_dt = datetime.date.fromisoformat(start_iso) - datetime.timedelta(days=2)  # バッファ
    end_incl_dt = datetime.date.fromisoformat(end_inclusive_iso)
    end_excl_dt = end_incl_dt + datetime.timedelta(days=1)

    def _normalize(df):
        if df is None or df.empty: return pd.DataFrame()
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
        # 要期間にトリミング（endはinclusive）
        mask = (res["date"] >= start_iso) & (res["date"] <= end_inclusive_iso)
        return res.loc[mask, ["date","open","high","low","close","volume"]]

    # 1) download API
    try:
        df = yf.download(
            symbol,
            start=start_dt.isoformat(),
            end=end_excl_dt.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        time.sleep(YF_SLEEP + random.uniform(0, YF_JITTER))
        if df is not None and not df.empty:
            n = _normalize(df)
            if not n.empty:
                return n
    except Exception as e:
        print(f"[WARN] yf.download failed {symbol}: {e}", file=sys.stderr)

    # 2) history API
    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(
            start=start_dt.isoformat(),
            end=end_excl_dt.isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        time.sleep(YF_SLEEP + random.uniform(0, YF_JITTER))
        n = _normalize(df)
        if not n.empty:
            return n
    except Exception as e:
        print(f"[WARN] tkr.history failed {symbol}: {e}", file=sys.stderr)

    # 3) period ベースのフォールバック（最大300日）
    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(period="300d", interval="1d", auto_adjust=True)
        time.sleep(YF_SLEEP + random.uniform(0, YF_JITTER))
        n = _normalize(df)
        if not n.empty:
            # 期間を再トリミング
            return n
    except Exception as e:
        print(f"[WARN] tkr.history(period) failed {symbol}: {e}", file=sys.stderr)

    return pd.DataFrame()

# ---------------- Metrics ----------------
def compute_volume_anomaly(df: pd.DataFrame):
    """Return dict with 0..1 score and details. If not enough data, score=0."""
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return {"score": 0.0, "eligible": False}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce").ffill()

    if len(d) < 45:  # 少なくとも約2か月分は欲しい
        return {"score": 0.0, "eligible": False}

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
    z60_u = max(0.0, min(1.0, z60 / 3.0))  # 3σ=1

    # PctRank(90d) by dollar volume（参考表示用）
    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    pct_rank_90 = float((tail <= last_dv).mean()) if len(tail) >= 5 else 0.0

    # 大型株バイアスを防ぐ → 比率中心（60%:RVOL, 40%:z）
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
def main():
    ensure_dirs()

    # Universe
    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]

    # External components
    trends = load_json_safe(TRENDS_JSON, {"items": {}})
    trends_items = trends.get("items", {})
    form4 = load_json_safe(FORM4_JSON, {"items": {}})
    form4_items = form4.get("items", {})

    # 取得期間：十分な統計を作れるよう広め
    end_dt = datetime.date.fromisoformat(DATE)          # business adjusted
    start_dt = end_dt - datetime.timedelta(days=220)    # ~10ヶ月分
    start_iso = start_dt.isoformat()
    end_inclusive_iso = end_dt.isoformat()              # 当日含めたい（内部で+1する）

    records = []
    recent_map = {}

    for sym in symbols:
        # 価格/出来高
        try:
            df = yfi_eod_range(sym, start_iso, end_inclusive_iso)
        except Exception as e:
            print(f"[WARN] yfi failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()

        recent_map[sym] = df
        vol = compute_volume_anomaly(df)
        vol_score = float(vol.get("score", 0.0))

        # Trends breakout 0..1
        tr = trends_items.get(sym) or {}
        trends_breakout = float(tr.get("score_0_1") or tr.get("breakout_score") or 0.0)

        # Insider momentum 0..1
        f4 = form4_items.get(sym) or {}
        insider_momo = float(f4.get("score_30", 0.0))

        # --- weights normalization (only-present) ---
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
        present_keys = [k for k,v in comps.items() if v is not None]
        wsum = sum(max(0.0, raw_w.get(k,0.0)) for k in present_keys) or 1.0
        norm_w = {k: (max(0.0, raw_w.get(k,0.0))/wsum) for k in present_keys}

        final_0_1 = sum( (comps.get(k,0.0) * norm_w.get(k,0.0)) for k in present_keys )
        score_pts = int(round(final_0_1 * 1000))

        # name/theme
        def _get(col):
            if col in uni.columns:
                try:
                    return uni.loc[uni["symbol"]==sym, col].values[0]
                except Exception:
                    return ""
            return ""
        rec = {
            "symbol": sym,
            "name": _get("name"),
            "theme": _get("theme"),
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,
            "vol_anomaly_score": vol_score,
            "insider_momo": insider_momo,
            "trends_breakout": trends_breakout,
            "score_components": comps,
            "score_weights": raw_w,  # JS 側で present-only 正規化
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

    # チャート（データがあるもののみ）
    for r in top10:
        df = recent_map.get(r["symbol"])
        try:
            if df is not None and not df.empty:
                save_chart_png_weekly_3m(r["symbol"], df, OUT_DIR, DATE)
            else:
                print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    # 進捗ログ（デバッグ用）
    empty_cnt = sum(1 for s in symbols if (recent_map.get(s) is None or recent_map.get(s).empty))
    print(f"Generated top10 for {DATE}: {len(top10)} symbols (universe={len(symbols)}, empty_ohlcv={empty_cnt})")

if __name__ == "__main__":
    main()
