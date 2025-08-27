#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (complete)
- OHLCV (yfinance) から「Volume anomaly」を 0..1 に算出
- Google Trends (site/data/trends/latest.json) を 0..1 で反映
- Insider momentum (site/data/insider/form4_latest.json) を 0..1 で反映
- News (site/data/news/latest.json) を 0..1 で反映（buzz を主に使用）
- "存在するコンポーネントだけ" を重み正規化して合算 → 1000 点
- テンプレートが読む JSON に必須キーを揃える
- 失敗時でも 0 を入れて落ちないように防御
"""

import os, sys, json, math, pathlib, random, datetime
from typing import Dict, Any
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
    # 18:00 ET より前は前営業日扱い
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    # 週末補正
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

# ---------------- Config ----------------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Weights（raw）。存在する要素のみで正規化
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM", "0.60"))
W_FORM4  = float(os.getenv("WEIGHT_FORM4",   "0.20"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS",  "0.20"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS",    "0.20"))  # デフォルトは 0.20（必要に応じ調整）

FORM4_JSON   = os.getenv("FORM4_JSON",   f"{OUT_DIR}/data/insider/form4_latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON",  f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",    f"{OUT_DIR}/data/news/latest.json")

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
def yfi_eod_range(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    価格/出来高（終値ベース）。不足や失敗時は空 DataFrame。
    """
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

    try:
        import yfinance as yf
    except Exception as e:
        print(f"[WARN] yfinance import failed: {e}", file=sys.stderr)
        return pd.DataFrame()

    # yfinance は end が「非包含」なので +1 日ずらし、余裕をもって start-2d
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    df = pd.DataFrame()
    # 2 回まで download()、だめなら history() を試す
    try:
        for _ in range(2):
            tmp = yf.download(
                symbol, start=start_dt.isoformat(), end=end_dt.isoformat(),
                interval="1d", auto_adjust=True, progress=False, threads=False
            )
            if tmp is not None and not tmp.empty:
                df = tmp
                break
        if df is None or df.empty:
            tkr = yf.Ticker(symbol)
            tmp = tkr.history(
                start=start_dt.isoformat(), end=end_dt.isoformat(),
                interval="1d", auto_adjust=True
            )
            if tmp is not None and not tmp.empty:
                df = tmp
    except Exception as e:
        print(f"[WARN] yfinance failed for {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # 列名を揃える
    df.columns = [str(c).strip().lower() for c in df.columns]
    name_map = {"date":"date","open":"open","high":"high","low":"low",
                "close":"close","adj close":"close","adjclose":"close","volume":"volume"}
    out: Dict[str, Any] = {}
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
def compute_volume_anomaly(df: pd.DataFrame) -> Dict[str, Any]:
    """
    出来高異常スコア（0..1）と詳細。
    - RVOL20（当日出来高 / 20日平均）
    - RVOL の 60日 z-score を 3σ=1 として 0..1 に圧縮
    - DollarVolume の 90日 Percentile
    """
    base = {"score": 0.0, "eligible": False, "rvol20": None, "z60": None,
            "pct_rank_90": None, "dollar_vol": None}
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return base

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce").fillna(method="ffill")
    if len(d) < 45:
        return base

    v_today = float(d["volume"].iloc[-1])
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1])
    rvol20 = (v_today / v_sma20) if v_sma20 > 0 else 0.0

    rvol_series = d["volume"] / d["volume"].rolling(20).mean()
    rvol_60 = rvol_series.tail(60).dropna()
    if len(rvol_60) < 20:
        z60 = 0.0
    else:
        mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
        z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
    z60_u = max(0.0, min(1.0, z60 / 3.0))  # 3σ を 1 としてクリップ

    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    if len(tail) >= 5:
        pct_rank_90 = float((tail <= last_dv).mean())
    else:
        pct_rank_90 = 0.0

    # 比率中心。寄与 60%:RVOL, 40%:z
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
def save_chart_png_weekly_3m(symbol: str, df: pd.DataFrame, out_dir: str, date_iso: str):
    # 週足化（終値は週末、出来高は合計）
    if df is None or df.empty:
        return
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

    # 入力の読み込み
    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]

    trends = load_json_safe(TRENDS_JSON, {"items": {}})
    trends_items = trends.get("items", {})

    form4 = load_json_safe(FORM4_JSON, {"items": {}})
    form4_items = form4.get("items", {})

    news = load_json_safe(NEWS_JSON, {"items": {}, "articles": {}})
    news_items = news.get("items", {})

    # 取得期間（約 6 ヶ月）
    end = datetime.date.fromisoformat(DATE)
    start = (end - datetime.timedelta(days=180)).isoformat()
    end_s = end.isoformat()

    records = []
    recent_map: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        # 価格/出来高
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] yfi failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()

        recent_map[sym] = df

        vol = compute_volume_anomaly(df)
        vol_score = float(vol.get("score") or 0.0)

        # Trends 0..1
        tr = trends_items.get(sym) or {}
        # fetch_trends.py の出力キーに合わせる（score_0_1 or score）
        trends_breakout = float(tr.get("score_0_1", tr.get("score", 0.0)) or 0.0)

        # Insider momentum 0..1（30 日スコアを採用）
        f4 = form4_items.get(sym) or {}
        insider_momo = float(f4.get("score_30", f4.get("insider_momo", 0.0)) or 0.0)

        # News（0..1）: buzz を主採用（将来は 0.7*buzz + 0.3*sent 等にしてもOK）
        nw = news_items.get(sym) or {}
        news_buzz = float(nw.get("buzz_score_0_1", 0.0) or 0.0)
        news_sent = float(nw.get("sent_score_0_1", 0.0) or 0.0)
        news_buzz_mult = nw.get("buzz_multiplier")
        news_buzz_rank = nw.get("buzz_rank")
        news_sent_rank = nw.get("sent_rank")
        news_score = news_buzz  # まずは buzz のみ

        # --- weights normalization (only-present) ---
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
        present_keys = [k for k, v in comps.items() if v is not None]
        wsum = sum(max(0.0, float(raw_w.get(k, 0.0))) for k in present_keys) or 1.0
        norm_w = {k: (max(0.0, float(raw_w.get(k, 0.0))) / wsum) for k in present_keys}

        final_0_1 = sum((float(comps.get(k, 0.0) or 0.0) * norm_w.get(k, 0.0)) for k in present_keys)
        score_pts = int(round(final_0_1 * 1000))

        # レコード（テンプレートが参照するキーは全て埋める）
        rec = {
            "symbol": sym,
            "name": uni.loc[uni["symbol"].str.upper()==sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"].str.upper()==sym, "theme"].values[0] if "theme" in uni.columns else "",
            "final_score_0_1": float(final_0_1),
            "score_pts": int(score_pts),

            # 各コンポーネントの素点
            "vol_anomaly_score": float(vol_score),
            "insider_momo": float(insider_momo),
            "trends_breakout": float(trends_breakout),

            # NEWS 詳細（UI バッジ用）
            "news_buzz": float(news_buzz),
            "news_sent": float(news_sent),
            "news_buzz_mult": news_buzz_mult,
            "news_buzz_rank": news_buzz_rank,
            "news_sent_rank": news_sent_rank,

            # ブレークダウン
            "score_components": comps,
            "score_weights": raw_w,  # JS 側で「存在キーのみ」正規化

            # 詳細：出来高
            "detail": {"vol_anomaly": vol},

            # チャート URL
            "chart_url": f"/charts/{DATE}/{sym}.png",
        }
        records.append(rec)

    # ランキング
    records.sort(key=lambda r: r.get("score_pts", 0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    # JSON 出力（Top10 と All の 2 種）
    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))
    (out_json_dir/"all.json").write_text(json.dumps(records, indent=2))

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

    print(f"Generated top10 for {DATE}: {len(top10)} symbols (universe={len(records)})")

if __name__ == "__main__":
    main()
