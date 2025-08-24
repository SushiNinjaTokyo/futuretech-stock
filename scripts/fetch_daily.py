#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, pathlib, random, time, datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests

# charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# US 市場日付（引け基準）
# =========================
def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d


# =========================
# 環境変数
# =========================
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()  # yfinance | tiingo
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# 予算管理
BUDGET_JPY_MAX = float(os.getenv("BUDGET_JPY_MAX", "10000"))
SPEND_FILE = os.getenv("SPEND_FILE", f"{OUT_DIR}/data/spend.json")
MANUAL_DAILY_COST_JPY = float(os.getenv("MANUAL_DAILY_COST_JPY", "0"))
PROVIDER_FIXED_JPY = 0.0 if DATA_PROVIDER == "yfinance" else float(os.getenv("TIINGO_MONTHLY_JPY", "1600"))


# =========================
# 予算ユーティリティ
# =========================
def month_key(date_iso: str):
    d = datetime.date.fromisoformat(date_iso)
    return f"{d.year}-{d.month:02d}"

def load_spend():
    p = pathlib.Path(SPEND_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def save_spend(data):
    p = pathlib.Path(SPEND_FILE); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))

def month_spend_total(spend, mkey):
    return float(spend.get(mkey, {}).get("total_jpy", 0))

def add_spend(spend, mkey, date_iso, amount_jpy, memo):
    month = spend.setdefault(mkey, {"items": [], "total_jpy": 0})
    month["items"].append({"date": date_iso, "amount_jpy": amount_jpy, "memo": memo})
    month["total_jpy"] = round(float(month["total_jpy"]) + amount_jpy, 2)

def budget_check():
    spend = load_spend(); mkey = month_key(DATE)
    month_used = month_spend_total(spend, mkey)
    if month_used + MANUAL_DAILY_COST_JPY > BUDGET_JPY_MAX:
        print(f"[BUDGET] Cap reached ({month_used:.0f} + {MANUAL_DAILY_COST_JPY:.0f} > {BUDGET_JPY_MAX:.0f}). Skipping run.")
        return False, spend
    return True, spend

def mark_fixed_costs(spend):
    mkey = month_key(DATE)
    month = spend.setdefault(mkey, {"items": [], "total_jpy": 0})
    if PROVIDER_FIXED_JPY > 0 and not month.get("provider_month_mark"):
        add_spend(spend, mkey, DATE, PROVIDER_FIXED_JPY, f"{DATA_PROVIDER} monthly flat")
        month["provider_month_mark"] = True
    if MANUAL_DAILY_COST_JPY > 0:
        add_spend(spend, mkey, DATE, MANUAL_DAILY_COST_JPY, "Variable API usage (manual)")
    save_spend(spend)


# =========================
# Data Providers
# =========================
def tiingo_eod_range(symbol, start, end):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"token": TIINGO_TOKEN, "startDate": start, "endDate": end, "resampleFreq": "daily"}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty: return df
    if "adjClose" in df.columns: df = df.rename(columns={"adjClose": "close"})
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({
        "date":   pd.to_datetime(df[cols.get("date","date")]).dt.tz_localize(None).dt.strftime("%Y-%m-%d"),
        "open":   pd.to_numeric(df.get("open",  pd.Series(dtype="float64")), errors="coerce"),
        "high":   pd.to_numeric(df.get("high",  pd.Series(dtype="float64")), errors="coerce"),
        "low":    pd.to_numeric(df.get("low",   pd.Series(dtype="float64")), errors="coerce"),
        "close":  pd.to_numeric(df.get("close", pd.Series(dtype="float64")), errors="coerce"),
        "volume": pd.to_numeric(df.get("volume",pd.Series(dtype="float64")), errors="coerce").fillna(0),
    })
    return out[["date","open","high","low","close","volume"]]

def yfi_eod_range(symbol, start, end):
    import yfinance as yf, time
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    def _normalize(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty: return pd.DataFrame()
        df2 = df_in.reset_index(); df2.columns = [str(c).strip().lower() for c in df2.columns]
        cands = {"date":["date","datetime","index"], "open":["open"], "high":["high"], "low":["low"],
                 "close":["close","adj close","adjclose"], "volume":["volume"]}
        out = {}
        for k, vs in cands.items():
            for v in vs:
                if v in df2.columns: out[k] = df2[v]; break
        if "date" not in out or "close" not in out: return pd.DataFrame()
        vol = out.get("volume", pd.Series([0]*len(out["close"])))
        res = pd.DataFrame({
            "date":   pd.to_datetime(out["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d"),
            "open":   pd.to_numeric(out.get("open",  pd.Series(dtype="float64")), errors="coerce"),
            "high":   pd.to_numeric(out.get("high",  pd.Series(dtype="float64")), errors="coerce"),
            "low":    pd.to_numeric(out.get("low",   pd.Series(dtype="float64")), errors="coerce"),
            "close":  pd.to_numeric(out["close"], errors="coerce"),
            "volume": pd.to_numeric(vol, errors="coerce").fillna(0),
        })
        return res[["date","open","high","low","close","volume"]]

    for attempt in range(2):
        df = yf.download(symbol, start=start_dt.isoformat(), end=end_dt.isoformat(),
                         interval="1d", auto_adjust=True, progress=False, threads=False)
        df = _normalize(df)
        if not df.empty: return df
        time.sleep(0.8*(attempt+1))

    df = _normalize(yf.Ticker(symbol).history(period="6mo", interval="1d", auto_adjust=True))
    if not df.empty: return df

    df = _normalize(yf.download(symbol, period="6mo", interval="1d", auto_adjust=True, progress=False, threads=False))
    return df

def get_eod_range(symbol, start, end):
    if MOCK_MODE:
        dates = pd.date_range(start=start, end=end, freq="B")
        base = 100.0 + random.Random(symbol).random()*20
        rows = []
        for d in dates:
            base *= (1.0 + random.uniform(-0.02, 0.02))
            vol = random.randint(100000, 5000000)
            rows.append({"date": d.strftime("%Y-%m-%d"), "open": base*0.99, "high": base*1.01,
                         "low": base*0.98, "close": base, "volume": vol})
        return pd.DataFrame(rows)
    if DATA_PROVIDER == "tiingo":
        if not TIINGO_TOKEN: raise RuntimeError("TIINGO_TOKEN is required for tiingo provider")
        return tiingo_eod_range(symbol, start, end)
    return yfi_eod_range(symbol, start, end)


# =========================
# 既存の軽量メトリクス（互換用）
# =========================
def compute_metrics(df: pd.DataFrame):
    if df is None or df.empty: return None
    for col in ("close","volume"):
        if col not in df.columns: return None
    d = df.copy()
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce").ffill()
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    if len(d) < 2: return 0.0, 1.0
    today_close, prev_close = d["close"].iloc[-1], d["close"].iloc[-2]
    pct_change = 0.0 if pd.isna(today_close) or pd.isna(prev_close) or float(prev_close)==0 else (float(today_close)-float(prev_close))/float(prev_close)
    vol_sma20 = d["volume"].rolling(20).mean().iloc[-1]; today_vol = d["volume"].iloc[-1]
    vol_ratio = 1.0 if pd.isna(vol_sma20) or float(vol_sma20)==0 else float(today_vol)/float(vol_sma20)
    return float(pct_change), float(vol_ratio)


# =========================
# 異常出来高（軽量版：120日で算出）
# =========================
def compute_volume_anomaly_light(df, close_col="close", vol_col="volume"):
    """
    0..1 スコア:
      RVOL20 -> rvol_norm, Z60 -> z_norm, PctRank90 -> pct_norm
      十分な履歴: 0.5*z + 0.3*rvol + 0.2*pct / 不足: 0.65*z + 0.35*rvol
    低流動フィルタ:
      volume >= 200k かつ DollarVol >= 5M
    """
    import numpy as np
    if df is None or df.empty or close_col not in df.columns or vol_col not in df.columns: return None
    d = df.copy()
    d[vol_col] = pd.to_numeric(d[vol_col], errors="coerce").fillna(0)
    d[close_col] = pd.to_numeric(d[close_col], errors="coerce").ffill()
    d = d.tail(120)
    if len(d) < 25: return None

    v = d[vol_col].values; vt = float(v[-1])
    sma20 = pd.Series(v).rolling(20).mean().iloc[-1]
    rvol20 = float(vt / sma20) if pd.notna(sma20) and sma20 > 0 else 0.0

    win = min(60, len(v)-1)
    if win >= 20:
        base = v[-(win+1):-1]
        mu = float(np.mean(base)); sd = float(np.std(base, ddof=0))
        z60 = float((vt - mu) / sd) if sd > 0 else 0.0
    else:
        z60 = 0.0

    pr_win = min(90, len(v)-1)
    if pr_win >= 20:
        hist = v[-pr_win:]
        pct_rank_90 = float((hist <= vt).sum() / len(hist))
    else:
        pct_rank_90 = None

    z_norm = max(0.0, min(1.0, 0.5 + 0.1*z60))
    rvol_norm = max(0.0, min(1.0, rvol20/3.0))
    score_0_1 = (0.65*z_norm + 0.35*rvol_norm) if pct_rank_90 is None else (0.5*z_norm + 0.3*rvol_norm + 0.2*pct_rank_90)

    dollar_vol = vt * float(d[close_col].iloc[-1])
    eligible = (vt >= 200_000) and (dollar_vol >= 5_000_000)

    return {
        "score_0_1": float(score_0_1),
        "rvol20": float(rvol20),
        "z60": float(z60),
        "pct_rank_90": None if pct_rank_90 is None else float(pct_rank_90),
        "z_norm": float(z_norm),
        "rvol_norm": float(rvol_norm),
        "dollar_vol": float(dollar_vol),
        "eligible": bool(eligible),
        "today_volume": float(vt),
        "sma20_volume": None if pd.isna(sma20) else float(sma20),
        "close": float(d[close_col].iloc[-1]),
    }


# =========================
# スコア合成エンジン（将来拡張用）
# =========================
def compose_final_score(components: dict, weights: dict) -> float:
    """
    components: {"vol_anomaly": 0..1, "news": 0..1, ...}
    weights:    {"vol_anomaly": 0.7,  "news": 0.3, ...}
    return: final score in 0..1
    """
    if not components: return 0.0
    w_sum = sum(max(0.0, float(w)) for w in weights.values()) or 1.0
    score = 0.0
    for key, val in components.items():
        w = float(weights.get(key, 0.0)) / w_sum
        v = max(0.0, min(1.0, float(val)))
        score += w * v
    return max(0.0, min(1.0, score))

def to_points(score_0_1: float) -> int:
    return int(round(max(0.0, min(1.0, float(score_0_1))) * 1000))


# =========================
# Charts
# =========================
def save_chart_png_weekly_3m(symbol: str, df_daily: pd.DataFrame, out_dir: str, date_iso: str):
    if df_daily is None or df_daily.empty:
        print(f"[WARN] weekly chart skipped (empty) {symbol}", file=sys.stderr); return
    d = df_daily.copy()
    d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date")
    cutoff = d["date"].max() - pd.Timedelta(days=120)
    d = d[d["date"] >= cutoff]
    if d.empty:
        print(f"[WARN] weekly window empty {symbol}", file=sys.stderr); return
    d = d.set_index("date")
    w = pd.DataFrame({
        "open":   d["open"].resample("W-FRI").first(),
        "high":   d["high"].resample("W-FRI").max(),
        "low":    d["low"].resample("W-FRI").min(),
        "close":  d["close"].resample("W-FRI").last(),
        "volume": d["volume"].resample("W-FRI").sum(),
    }).dropna(how="any")
    if w.empty:
        print(f"[WARN] weekly resample empty {symbol}", file=sys.stderr); return
    charts_dir = pathlib.Path(out_dir) / "charts" / date_iso
    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.4)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    plt.savefig(charts_dir / f"{symbol}.png")
    plt.close()


# =========================
# メイン
# =========================
def main():
    ok, spend = budget_check()
    if not ok: return

    uni = pd.read_csv(UNIVERSE_CSV)
    rows, top10 = [], []

    end = DATE
    start_short = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=120)).isoformat()
    recent_map = {}

    for _, t in uni.iterrows():
        symbol = str(t["symbol"]).strip()
        try:
            df = get_eod_range(symbol, start_short, end)
            if df is not None and not df.empty:
                df = df.rename(columns=lambda c: str(c).strip().lower())
                if "date" not in df.columns and hasattr(df.index, "dtype"):
                    try: df = df.reset_index().rename(columns={"index":"date"})
                    except Exception: pass

            recent_map[symbol] = df

            base_metrics = compute_metrics(df)
            if not base_metrics:
                print(f"[WARN] skip (no metrics) {symbol}", file=sys.stderr); continue
            pct_change, vol_ratio = base_metrics

            # --- 異常出来高（現状はこれのみ採点に使用） ---
            anom = compute_volume_anomaly_light(df)
            if not anom:
                vol_component = 0.0; eligible = False; vol_detail = {}
            else:
                vol_component = anom["score_0_1"]; eligible = anom["eligible"]; vol_detail = anom

            # --- 将来拡張用: コンポーネント & 重み ---
            score_components = {
                "vol_anomaly": float(vol_component),
                # "news": 0.0,
                # "inst_delta": 0.0,
            }
            score_weights = {
                "vol_anomaly": 1.0,  # 現状は出来高のみ
                # "news": 0.0,
                # "inst_delta": 0.0,
            }

            final_0_1 = compose_final_score(score_components, score_weights)
            score_pts = to_points(final_0_1)

            rows.append({
                "symbol": symbol,
                "name":   t.get("name",""),
                "theme":  t.get("theme",""),
                "pct_change": pct_change,
                "vol_ratio":  vol_ratio,
                "news_count": 0,
                "eligible_liquidity": eligible,
                "score_components": score_components,
                "score_weights": score_weights,
                "final_score_0_1": final_0_1,
                "score_pts": score_pts,                 # 0..1000
                "vol_anomaly_score": vol_component,     # 0..1（参考）
                "chart_url": f"/charts/{DATE}/{symbol}.png",
                "tech_note": "Auto tech note TBD",
                "ir_note":   "IR/News summary TBD",
                "detail": {"vol_anomaly": vol_detail}
            })
        except Exception as e:
            print(f"[WARN] {symbol}: {e}", file=sys.stderr)
            continue

    out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(out_json_dir / "top10.json", "w") as f: json.dump(top10, f, indent=2)
        mark_fixed_costs(spend)
        print(f"Generated top10 for {DATE}: 0 symbols (no rows)")
        return

    # 低流動を除外（全滅時はフォールバック）
    eligible_rows = [r for r in rows if r.get("eligible_liquidity", False)]
    target = eligible_rows if eligible_rows else rows

    # スコアでソート
    target.sort(key=lambda x: x.get("final_score_0_1", 0.0), reverse=True)
    top10 = target[:10]

    # rank 付与（1..10）
    for i, r in enumerate(top10, start=1):
        r["rank"] = i

    # JSON出力
    with open(out_json_dir / "top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    # チャート（Technical タブ用）
    if not MOCK_MODE and top10:
        for r in top10:
            try:
                hist = recent_map.get(r["symbol"])
                if hist is None or hist.empty:
                    print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr); continue
                save_chart_png_weekly_3m(r["symbol"], hist, OUT_DIR, DATE)
            except Exception as e:
                print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    mark_fixed_costs(spend)
    print(f"Generated top10 for {DATE}: {len(top10)} symbols")


if __name__ == "__main__":
    main()
