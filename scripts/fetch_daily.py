#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (robust)
- OHLCV: yfinance.Ticker.history → Stooq CSV → pandas-datareader(stooq) の順にフェイルオーバー
- Volume anomaly 0..1 を算出
- Google Trends (site/data/trends/latest.json) の breakout を 0..1 で反映
- Insider momentum (site/data/insider/form4_latest.json) を 0..1 で反映
- News (site/data/news/latest.json) のカバレッジを 0..1 で反映（percentile）
- 重みは環境変数で指定し、"存在するコンポーネントだけ" を正規化して 1000 点に換算
- テンプレートが読む JSON に必須キーを必ず出力
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
    # 週末だけ避ける（米祝日はデータ側で欠損になるが問題なし）
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

def usa_market_date_now():
    # レガシー互換（未使用だが環境変数未設定時のフォールバックで使用）
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

# Sleep between vendor calls (yfinance 等の連打対策)
YFI_SLEEP = float(os.getenv("YFI_SLEEP", "0.4"))

# Weights (raw). Only-present components will be normalized.
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
      1) yfinance.Ticker(...).history() で個別取得（downloadは使わない）
      2) 失敗なら Stooq CSV 直叩き
      3) さらに pandas-datareader の stooq
    どれか1つでも取れたら標準化して返す。
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

    # --- 1) yfinance: Ticker.history を優先（downloadは使わない）
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
            # close 列は 'close' 優先、無ければ adj close 系
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

    # --- 2) Stooq CSV 直（無料・軽量）
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

    # --- 3) pandas-datareader (stooq)
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

    # すべて失敗
    print(f"[ERROR] {symbol}: no OHLCV from all sources", file=sys.stderr)
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
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce")
    if len(d) < 45:  # 少なくとも約2か月分は欲しい
        return {"score": 0.0, "eligible": False}

    # 直近日
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
    # 0..1 に圧縮（3σ=1 としてクリップ）
    z60_u = max(0.0, min(1.0, z60 / 3.0))

    # PctRank(90d) by dollar volume
    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    if len(tail) >= 5:
        pct_rank_90 = float((tail <= last_dv).mean())
    else:
        pct_rank_90 = 0.0

    # 寄与 60%:RVOL, 40%:z
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

def ensure_dirs(date_iso):
    (pathlib.Path(OUT_DIR)/"data"/date_iso).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"charts"/date_iso).mkdir(parents=True, exist_ok=True)

# ---------------- Charts ----------------
def save_chart_png_weekly_3m(symbol, df, out_dir, date_iso):
    # 週足化（終値は週末、出来高は合計）
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
    # 取得対象日：常に「レポート日の前営業日」まで
    report_d = datetime.date.fromisoformat(DATE)
    end_day = prev_us_business_day(report_d - datetime.timedelta(days=1))
    start_day = end_day - datetime.timedelta(days=180)
    start = start_day.isoformat()
    end_s = end_day.isoformat()

    ensure_dirs(report_d.isoformat())

    # 入力の読み込み
    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip() for s in uni["symbol"].tolist()]

    trends = load_json_safe(TRENDS_JSON, {"items": {}})
    trends_items = trends.get("items", {})

    form4 = load_json_safe(FORM4_JSON, {"items": {}})
    form4_items = form4.get("items", {})

    news = load_json_safe(NEWS_JSON, {"items": {}})
    news_items = news.get("items", {})

    records = []
    recent_map = {}

    for sym in symbols:
        # 価格/出来高
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] OHLCV failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        recent_map[sym] = df

        # Volume anomaly
        if df is None or df.empty:
            vol_detail = None  # ← テンプレ側で “Unavailable” 表示にするため None
            vol_score = 0.0
        else:
            vol_detail = compute_volume_anomaly(df)
            vol_score = float(vol_detail.get("score", 0.0))

        # Trends breakout 0..1
        tr = trends_items.get(sym) or {}
        trends_breakout = float(tr.get("score_0_1") or tr.get("breakout_score") or 0.0)

        # Insider momentum 0..1
        f4 = form4_items.get(sym) or {}
        insider_momo = float(f4.get("score_30", 0.0))  # 30日側

        # News 0..1 + rank + recent count
        nw = news_items.get(sym) or {}
        news_score = float(nw.get("score_0_1", 0.0))
        news_rank = int(nw.get("rank", 0)) if str(nw.get("rank","")).strip() else None
        news_recent = int(nw.get("recent_count", 0))

        # --- weights normalization (only-present) ---
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
        present_keys = [k for k in comps.keys()]  # すべてのキーを対象にする（0 は 0 として扱う）
        wsum = sum(max(0.0, raw_w.get(k,0.0)) for k in present_keys) or 1.0
        norm_w = {k: (max(0.0, raw_w.get(k,0.0))/wsum) for k in present_keys}

        final_0_1 = sum( (comps.get(k,0.0) * norm_w.get(k,0.0)) for k in present_keys )
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
            "news_rank": news_rank,
            "news_recent_count": news_recent,

            # breakdown & weights
            "score_components": comps,
            "score_weights": raw_w,  # JS 側でも present-only 正規化するが、ここは生の重み

            # details for UI
            "detail": {"vol_anomaly": vol_detail},  # None なら Unavailable 表示
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
