#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scorer & exporter (robust, with DII and enhanced volume anomaly)
- OHLCV: yfinance.Ticker.history → (fallback) Stooq CSV → (fallback) pandas-datareader(stooq)
- Volume anomaly: RVOL20 + zscore(60d) + POT風ボーナス
- DII: site/data/dii/latest.json の score_0_1 を採用（fail時は0）
- Trends / News: 既存の latest.json を採用
- 重みは環境変数、存在キーのみ正規化して 1000点換算
- 価格差分（1D/1W/1M）は ref(=REPORT_DATEの前営業日) 終値に対し 1/5/20本前
- 出力: site/data/<DATE>/top10.json と charts
"""

import os, sys, json, math, pathlib, random, datetime, time, io, csv
from zoneinfo import ZoneInfo
import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- time helpers ----
def prev_us_business_day(d: datetime.date) -> datetime.date:
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 18:00 ET までは前営業日を採用（配信安定のため）
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

# ---- config ----
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV","data/universe.csv")
OUT_DIR      = os.getenv("OUT_DIR","site")
DATE         = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE    = os.getenv("MOCK_MODE","false").lower()=="true"

YFI_SLEEP = float(os.getenv("YFI_SLEEP","0.4"))

# weights
W_VOL    = float(os.getenv("WEIGHT_VOL_ANOM","0.25"))
W_DII    = float(os.getenv("WEIGHT_DII","0.25"))
W_TRENDS = float(os.getenv("WEIGHT_TRENDS","0.30"))
W_NEWS   = float(os.getenv("WEIGHT_NEWS","0.20"))

# inputs
DII_JSON     = os.getenv("DII_JSON",    f"{OUT_DIR}/data/dii/latest.json")
TRENDS_JSON  = os.getenv("TRENDS_JSON", f"{OUT_DIR}/data/trends/latest.json")
NEWS_JSON    = os.getenv("NEWS_JSON",   f"{OUT_DIR}/data/news/latest.json")

# ---------- helpers ----------
def _first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _safe_numeric_series(s):
    """Series 以外や None の場合は None を返す"""
    if s is None or not isinstance(s, pd.Series):
        return None
    return pd.to_numeric(s, errors="coerce")

def _normalize_ohlcv_df(df):
    """
    yfinance / stooq 由来の DataFrame を統一化:
    - 列名小文字化
    - 日付は %Y-%m-%d 文字列
    - 必要列が揃っていれば標準形にして返す。ダメなら空 DataFrame
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    # date 列を確実に作る
    date_col = "date" if "date" in d.columns else None
    if date_col is None:
        # yfinance.history は index が DatetimeIndex のことが多い
        d = d.reset_index(drop=False)
        d.columns = [str(c).strip().lower() for c in d.columns]
        date_col = "date" if "date" in d.columns else "index"

    # close 列の候補（存在チェックのみ。値は後で numeric 変換）
    close_col = _first_existing(d.columns, ["close", "adj close", "adjclose"])
    open_col  = "open"  if "open"  in d.columns else None
    high_col  = "high"  if "high"  in d.columns else None
    low_col   = "low"   if "low"   in d.columns else None
    vol_col   = "volume" if "volume" in d.columns else None

    if close_col is None or vol_col is None:
        return pd.DataFrame()

    # 文字列日付へ
    try:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    except Exception:
        return pd.DataFrame()
    d = d.dropna(subset=[date_col])
    if d.empty:
        return pd.DataFrame()
    d["date"] = d[date_col].dt.strftime("%Y-%m-%d")

    # 数値化（Series 以外は不正とみなす）
    close_s = _safe_numeric_series(d.get(close_col))
    vol_s   = _safe_numeric_series(d.get(vol_col))
    if close_s is None or vol_s is None:
        return pd.DataFrame()

    d["close"]  = close_s
    d["volume"] = vol_s.fillna(0)

    if open_col in d.columns:
        d["open"] = _safe_numeric_series(d.get(open_col))
    else:
        d["open"] = pd.Series([None] * len(d))
    if high_col in d.columns:
        d["high"] = _safe_numeric_series(d.get(high_col))
    else:
        d["high"] = pd.Series([None] * len(d))
    if low_col in d.columns:
        d["low"]  = _safe_numeric_series(d.get(low_col))
    else:
        d["low"]  = pd.Series([None] * len(d))

    d = d.dropna(subset=["close"])  # 終値欠損は除外
    if d.empty:
        return pd.DataFrame()

    out = d[["date","open","high","low","close","volume"]].copy()
    return out.reset_index(drop=True)

# ---- providers ----
def yfi_eod_range(symbol, start, end):
    """
    指定期間の OHLCV を返す（標準列: date/open/high/low/close/volume）
    - yfinance → Stooq CSV → pandas-datareader(stooq) の順で試行
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

    # 1) yfinance (Ticker.history)
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        df = t.history(
            start=(start_dt - datetime.timedelta(days=7)).isoformat(),
            end=(end_dt + datetime.timedelta(days=2)).isoformat(),
            interval="1d",
            auto_adjust=True
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _normalize_ohlcv_df(df)
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if not df.empty:
                time.sleep(YFI_SLEEP)
                return df
    except Exception as e:
        print(f"[WARN] yfinance failed for {symbol}: {e}", file=sys.stderr)

    # 2) Stooq CSV
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        r = requests.get(url, timeout=20)
        if r.ok and "Date,Open,High,Low,Close,Volume" in r.text:
            reader = csv.DictReader(io.StringIO(r.text))
            rows=[]
            for row in reader:
                rows.append({
                    "date": row["Date"],
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low":  row.get("Low"),
                    "close":row.get("Close"),
                    "volume":row.get("Volume"),
                })
            if rows:
                df = pd.DataFrame(rows)
                df = _normalize_ohlcv_df(df)
                df = df[(df["date"] >= start) & (df["date"] <= end)]
                if not df.empty:
                    time.sleep(0.2)
                    return df
    except Exception as e:
        print(f"[WARN] stooq csv failed for {symbol}: {e}", file=sys.stderr)

    # 3) pandas-datareader stooq
    try:
        import pandas_datareader.data as web
        df = web.DataReader(symbol, "stooq",
                            start=start_dt - datetime.timedelta(days=7),
                            end=end_dt + datetime.timedelta(days=2))
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.sort_index().reset_index().rename(columns={
                "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
            })
            df = _normalize_ohlcv_df(df)
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if not df.empty:
                time.sleep(0.2)
                return df
    except Exception as e:
        print(f"[WARN] pandas-datareader failed for {symbol}: {e}", file=sys.stderr)

    print(f"[ERROR] {symbol}: no OHLCV from all sources", file=sys.stderr)
    return pd.DataFrame()

# ---- metrics ----
def compute_volume_anomaly(df: pd.DataFrame):
    """
    強化版 異常出来高スコア 0..1
    - ベース: RVOL20 と z60（60日内の相対位置）を合成
    - 追加: 上位10%閾値超過度合いで加点（POT風; 最大+0.3）
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"score": 0.0, "eligible": False}
    if "volume" not in df.columns or "close" not in df.columns:
        return {"score": 0.0, "eligible": False}

    d = df.copy()
    # 型の健全性確認（Series であること）
    if not isinstance(d["volume"], pd.Series) or not isinstance(d["close"], pd.Series):
        return {"score": 0.0, "eligible": False}

    # 正しい日付順に
    try:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    except Exception:
        return {"score": 0.0, "eligible": False}
    d = d.dropna(subset=["date"]).sort_values("date")

    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["close"])
    if len(d) < 60:
        return {"score": 0.0, "eligible": False}

    v_today = float(d["volume"].iloc[-1])
    v_sma20 = float(d["volume"].rolling(20).mean().iloc[-1])
    rvol20  = v_today / v_sma20 if v_sma20 > 0 else 0.0

    rvol_series = d["volume"] / d["volume"].rolling(20).mean()
    rvol_60 = rvol_series.tail(60).dropna()
    if len(rvol_60) < 20:
        z60 = 0.0
        tail_bonus = 0.0
    else:
        mu, sd = float(rvol_60.mean()), float(rvol_60.std(ddof=0))
        z60 = (rvol20 - mu) / sd if sd > 1e-9 else 0.0
        # POT っぽい上位裾ボーナス（0..0.3）
        q90 = float(rvol_60.quantile(0.9))
        tail_bonus = max(0.0, min(0.3, (rvol20 - q90) / max(1e-9, q90))) if q90 > 0 else 0.0

    z60_u = max(0.0, min(1.0, z60 / 3.0))
    base = 0.55 * max(0.0, min(1.0, rvol20 / 5.0)) + 0.45 * z60_u
    vol_score = max(0.0, min(1.0, base + tail_bonus))

    # Dollar volume rank（参考値）
    d["dollar_vol"] = d["close"] * d["volume"]
    last_dv = float(d["dollar_vol"].iloc[-1])
    tail = d["dollar_vol"].tail(90).dropna()
    pct_rank_90 = float((tail <= last_dv).mean()) if len(tail) >= 5 else 0.0

    return {
        "score": vol_score,
        "eligible": True,
        "rvol20": rvol20,
        "z60": z60,
        "pct_rank_90": pct_rank_90,
        "dollar_vol": last_dv,
    }

def get_closes_for_deltas(df: pd.DataFrame, ref_date_iso: str):
    if df is None or df.empty:
        return None, None, None, None
    d = df.copy()
    try:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    except Exception:
        return None, None, None, None
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    ref_dt = pd.to_datetime(ref_date_iso)
    idx = d.index[d["date"] <= ref_dt]
    if len(idx) == 0:
        return None, None, None, None
    i_base = int(idx[-1])

    def safe_close(i):
        if i < 0 or i >= len(d):
            return None
        try:
            v = d.at[i, "close"]  # スカラー取り出し
            v = float(v)
            return None if math.isnan(v) else v
        except Exception:
            return None

    c0   = safe_close(i_base)
    c_1  = safe_close(i_base-1)
    c_5  = safe_close(i_base-5)
    c_20 = safe_close(i_base-20)
    return c0, c_1, c_5, c_20

def pct_change(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a/b - 1.0) * 100.0

# ---- io helpers ----
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

# ---- charts ----
def save_chart_png_weekly_3m(symbol, df, out_dir, date_iso):
    if df is None or df.empty:
        return
    d = df.copy()
    try:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    except Exception:
        return
    d = d.dropna(subset=["date"]).set_index("date").sort_index()
    if d.empty:
        return

    w = pd.DataFrame({
        "close": d["close"].resample("W-FRI").last(),
        "volume": d["volume"].resample("W-FRI").sum(),
    }).dropna()
    w = w.tail(13)
    if w.empty:
        return

    plt.figure(figsize=(9, 4.6), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.3)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    out = pathlib.Path(out_dir)/"charts"/date_iso/f"{symbol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()

# ---- main ----
def main():
    report_d = datetime.date.fromisoformat(DATE)
    end_day = prev_us_business_day(report_d)
    start_day = end_day - datetime.timedelta(days=400)
    start = start_day.isoformat()
    end_s = end_day.isoformat()

    ensure_dirs(report_d.isoformat())

    uni = pd.read_csv(UNIVERSE_CSV)
    uni["symbol"] = uni["symbol"].astype(str).str.upper().str.strip()
    symbols = uni["symbol"].tolist()

    # inputs
    trends_items = load_json_safe(TRENDS_JSON, {"items":{}}).get("items",{})
    news_items   = load_json_safe(NEWS_JSON,   {"items":{}}).get("items",{})
    dii_items    = load_json_safe(DII_JSON,    {"items":{}}).get("items",{})

    print(f"[INFO] loaded items: trends={len(trends_items)} news={len(news_items)} dii={len(dii_items)}")

    records=[]
    recent_map={}

    for sym in symbols:
        # OHLCV
        try:
            df = yfi_eod_range(sym, start, end_s)
        except Exception as e:
            print(f"[WARN] OHLCV failed {sym}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        recent_map[sym] = df

        # price deltas
        d1=d5=d20=None
        if isinstance(df, pd.DataFrame) and not df.empty:
            c0,c_1,c_5,c_20 = get_closes_for_deltas(df, end_s)
            d1  = pct_change(c0, c_1)
            d5  = pct_change(c0, c_5)
            d20 = pct_change(c0, c_20)

        # volume anomaly
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                vol_detail = compute_volume_anomaly(df)
            except Exception as e:
                print(f"[WARN] vol anomaly failed {sym}: {e}", file=sys.stderr)
                vol_detail = {"score": 0.0, "eligible": False}
        else:
            vol_detail = {"score": 0.0, "eligible": False}
        vol_score = float(vol_detail.get("score",0.0))

        # trends/news/dii
        tr = trends_items.get(sym) or {}
        trends_breakout = float(tr.get("score_0_1") or tr.get("breakout_score") or 0.0)

        nw = news_items.get(sym) or {}
        news_score = float(nw.get("score_0_1",0.0))
        news_recent = int(nw.get("recent_count",0) or 0)

        di = dii_items.get(sym) or {}
        dii_score = float(di.get("score_0_1",0.0))
        dii_ratio = None
        if "ats_share_ratio" in di and str(di.get("ats_share_ratio","")).strip():
            try:
                dii_ratio = float(di.get("ats_share_ratio"))
            except Exception:
                dii_ratio = None

        # weights normalize
        comps = {
            "volume_anomaly": vol_score if vol_detail is not None else 0.0,
            "dii": dii_score,
            "trends_breakout": trends_breakout,
            "news": news_score,
        }
        raw_w = {
            "volume_anomaly": W_VOL,
            "dii": W_DII,
            "trends_breakout": W_TRENDS,
            "news": W_NEWS,
        }
        keys = list(comps.keys())
        wsum = sum(max(0.0, float(raw_w.get(k,0.0))) for k in keys) or 1.0
        norm_w = {k:(max(0.0, float(raw_w.get(k,0.0)))/wsum) for k in keys}

        final_0_1 = sum(float(comps.get(k,0.0))*float(norm_w.get(k,0.0)) for k in keys)
        final_0_1 = max(0.0, min(1.0, final_0_1))
        score_pts = int(round(final_0_1*1000))

        rec = {
            "symbol": sym,
            "name": uni.loc[uni["symbol"]==sym, "name"].values[0] if "name" in uni.columns else "",
            "theme": uni.loc[uni["symbol"]==sym, "theme"].values[0] if "theme" in uni.columns else "",
            "final_score_0_1": final_0_1,
            "score_pts": score_pts,

            "vol_anomaly_score": vol_score,
            "trends_breakout": trends_breakout,
            "news_score": news_score,
            "news_recent_count": news_recent,
            "dii_score": dii_score,
            "dii_ratio": dii_ratio,

            "score_components": comps,
            "score_weights": raw_w,

            "detail": {"vol_anomaly": vol_detail, "dii": di},
            "chart_url": f"/charts/{DATE}/{sym}.png",

            "price_delta_1d": d1,
            "price_delta_1w": d5,
            "price_delta_1m": d20,
        }
        records.append(rec)

    records.sort(key=lambda r: r.get("score_pts",0), reverse=True)
    for i, r in enumerate(records, 1):
        r["rank"] = i

    out_json_dir = pathlib.Path(OUT_DIR)/"data"/DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    top10 = records[:10]
    (out_json_dir/"top10.json").write_text(json.dumps(top10, indent=2))

    for r in top10:
        df = recent_map.get(r["symbol"])
        try:
            if isinstance(df, pd.DataFrame) and not df.empty:
                save_chart_png_weekly_3m(r["symbol"], df, OUT_DIR, DATE)
            else:
                print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    print(f"Generated top10 for {DATE}: {len(top10)} symbols (universe={len(symbols)})")

if __name__ == "__main__":
    main()
