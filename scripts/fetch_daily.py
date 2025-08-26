#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, pathlib, random, datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

# charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Time helpers ----------------
def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 引け後18:00未満なら当日は不完全 → 前営業日にする
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    # 土日補正（土=5, 日=6 → 直近金曜）
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

# ---------------- Env ----------------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()  # yfinance | tiingo
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Budget guard（据え置き）
BUDGET_JPY_MAX = float(os.getenv("BUDGET_JPY_MAX", "10000"))
SPEND_FILE = os.getenv("SPEND_FILE", f"{OUT_DIR}/data/spend.json")
MANUAL_DAILY_COST_JPY = float(os.getenv("MANUAL_DAILY_COST_JPY", "0"))
PROVIDER_FIXED_JPY = 0.0 if DATA_PROVIDER == "yfinance" else float(os.getenv("TIINGO_MONTHLY_JPY", "1600"))

def month_key(date_iso: str):
    d = datetime.date.fromisoformat(date_iso)
    return f"{d.year}-{d.month:02d}"

def load_spend():
    p = pathlib.Path(SPEND_FILE)
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: return {}
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
    spend = load_spend()
    mkey = month_key(DATE)
    month_used = month_spend_total(spend, mkey)
    today_cost = MANUAL_DAILY_COST_JPY
    if month_used + today_cost > BUDGET_JPY_MAX:
        print(f"[BUDGET] Cap reached ({month_used:.0f} + {today_cost:.0f} > {BUDGET_JPY_MAX:.0f}). Skipping run.")
        return False, spend
    return True, spend

def mark_fixed_costs(spend):
    if PROVIDER_FIXED_JPY <= 0:
        return save_spend(spend)
    mkey = month_key(DATE)
    month = spend.setdefault(mkey, {"items": [], "total_jpy": 0})
    if not month.get("provider_month_mark"):
        add_spend(spend, mkey, DATE, PROVIDER_FIXED_JPY, f"{DATA_PROVIDER} monthly flat")
        month["provider_month_mark"] = True
    if MANUAL_DAILY_COST_JPY > 0:
        add_spend(spend, mkey, DATE, MANUAL_DAILY_COST_JPY, "Variable API usage (manual)")
    save_spend(spend)

# ---------------- Data Providers ----------------
def tiingo_eod_range(symbol, start, end):
    import requests
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"token": TIINGO_TOKEN, "startDate": start, "endDate": end, "resampleFreq": "daily"}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    df = pd.DataFrame(r.json())
    if "adjClose" in df.columns:
        df = df.rename(columns={"adjClose":"close"})
    return df

def yfi_eod_range(symbol, start, end):
    import yfinance as yf
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    df = None
    for attempt in range(2):
        tmp = yf.download(
            symbol, start=start_dt.isoformat(), end=end_dt.isoformat(),
            interval="1d", auto_adjust=True, progress=False, threads=False
        )
        if tmp is not None and not tmp.empty:
            df = tmp; break

    if df is None or df.empty:
        tkr = yf.Ticker(symbol)
        tmp = tkr.history(start=start_dt.isoformat(), end=end_dt.isoformat(), interval="1d", auto_adjust=True)
        if tmp is None or tmp.empty: return pd.DataFrame()
        df = tmp

    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    name_map = {
        "date":"date","open":"open","high":"high","low":"low",
        "close":"close","adj close":"close","adjclose":"close","volume":"volume",
    }
    out = {}
    for col in df.columns:
        if col in name_map and name_map[col] not in out:
            out[name_map[col]] = df[col]
    if "date" not in out or "close" not in out or "volume" not in out:
        return pd.DataFrame()
    result = pd.DataFrame({
        "date":   pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d"),
        "open":   pd.to_numeric(out.get("open",   pd.Series(dtype="float64")), errors="coerce"),
        "high":   pd.to_numeric(out.get("high",   pd.Series(dtype="float64")), errors="coerce"),
        "low":    pd.to_numeric(out.get("low",    pd.Series(dtype="float64")), errors="coerce"),
        "close":  pd.to_numeric(out.get("close",  pd.Series(dtype="float64")), errors="coerce"),
        "volume": pd.to_numeric(out.get("volume", pd.Series(dtype="float64")), errors="coerce").fillna(0),
    })
    return result[["date","open","high","low","close","volume"]]

def get_eod_range(symbol, start, end):
    if MOCK_MODE:
        dates = pd.date_range(start=start, end=end, freq="B")
        base = 100.0 + random.Random(symbol).random()*20
        rows = []
        for d in dates:
            base *= (1.0 + random.uniform(-0.02, 0.02))
            vol = random.randint(100000, 5000000)
            rows.append({"date": d.strftime("%Y-%m-%d"), "open": base*0.99, "high": base*1.01, "low": base*0.98, "close": base, "volume": vol})
        return pd.DataFrame(rows)
    if DATA_PROVIDER == "tiingo":
        if not TIINGO_TOKEN:
            raise RuntimeError("TIINGO_TOKEN is required for tiingo provider")
        return tiingo_eod_range(symbol, start, end)
    return yfi_eod_range(symbol, start, end)

# ---------------- Metrics & Charts ----------------
def compute_volume_anomaly(df) -> float:
    """
    出来高アノマリーの素点（比率）:
      today_volume / SMA20_volume
    ※ 後で宇宙内で 0–1 正規化してスコア化
    """
    if df is None or df.empty or "volume" not in df.columns:
        return None
    d = df.copy()
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    if len(d) < 21:
        return None
    vol_sma20 = d["volume"].rolling(20).mean().iloc[-1]
    today_vol = d["volume"].iloc[-1]
    if vol_sma20 is None or vol_sma20 <= 0:
        return None
    return float(today_vol) / float(vol_sma20)

def save_chart_png_weekly_3m(symbol, df, out_dir, date_iso):
    """
    週足（終値ベース）3ヶ月の簡易チャート
    """
    if df is None or df.empty: return
    charts_dir = pathlib.Path(out_dir) / "charts" / date_iso
    charts_dir.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d = d.set_index("date")
    # 週足 OHLC を作る（終値中心で）
    w_close = d["close"].resample("W-FRI").last()
    w_ma5 = w_close.rolling(5).mean()
    w_ma12 = w_close.rolling(12).mean()

    plt.figure(figsize=(9, 4.8), dpi=120)
    plt.plot(w_close.index, w_close.values, linewidth=1.2, label="Close (W)")
    plt.plot(w_ma5.index, w_ma5.values, linewidth=0.9, label="MA5")
    plt.plot(w_ma12.index, w_ma12.values, linewidth=0.9, label="MA12")
    plt.title(f"{symbol} — 3M Weekly")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out = charts_dir / f"{symbol}.png"
    plt.savefig(out); plt.close()

# ---------------- External scores (Insider / Trends) ----------------
def load_insider_map(date_iso: str) -> dict:
    """
    insider_momo を {symbol: 0–1} で返す
    """
    base = pathlib.Path(OUT_DIR) / "data"
    p_latest = base / "insider" / "form4_latest.json"
    p_dated  = base / date_iso / "insider.json"
    data = None
    for p in (p_latest, p_dated):
        if p.exists():
            try:
                data = json.loads(p.read_text()); break
            except Exception:
                pass
    if not data: return {}
    items = data.get("items", {})
    out = {}
    for sym, rec in items.items():
        v = rec.get("insider_momo")
        try: out[sym.upper()] = float(v) if v is not None else 0.0
        except Exception: out[sym.upper()] = 0.0
    return out

def load_trends_scores(date_iso: str) -> dict:
    """
    Google Trends の 0–1 スコアを {symbol: 0–1} で返す
    """
    base = pathlib.Path(OUT_DIR) / "data"
    p_latest = base / "trends" / "latest.json"
    p_dated  = base / date_iso / "trends.json"
    data = None
    for p in (p_latest, p_dated):
        if p.exists():
            try:
                data = json.loads(p.read_text()); break
            except Exception:
                pass
    if not data: return {}
    items = data.get("items", {})
    out = {}
    for sym, rec in items.items():
        v = rec.get("score_0_1", 0.0)
        try: out[sym.upper()] = float(v) if v is not None else 0.0
        except Exception: out[sym.upper()] = 0.0
    return out

# ---------------- Scoring helpers ----------------
def minmax_norm(arr):
    vals = [v for v in arr if v is not None and not np.isnan(v)]
    if not vals: return lambda x: 0.0
    mn, mx = min(vals), max(vals)
    if mx <= mn: return lambda x: 0.0
    return lambda x: (float(x) - mn) / (mx - mn) if x is not None else 0.0

def combine_to_1000(components: dict, weights: dict) -> float:
    """
    components: {name: value in 0..1}
    weights:    {name: weight float}
    → weighted average * 1000
    存在する要素だけで重みを再正規化
    """
    num = 0.0; den = 0.0
    for k, w in weights.items():
        if k in components and components[k] is not None:
            num += float(components[k]) * float(w)
            den += float(w)
    if den <= 0: return 0.0
    return float(num / den) * 1000.0

# ---------------- Main ----------------
def main():
    ok, spend = budget_check()
    if not ok:
        return

    uni = pd.read_csv(UNIVERSE_CSV)
    symbols = [str(s).strip().upper() for s in uni["symbol"].tolist()]

    end = DATE
    start_short = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=90)).isoformat()

    recent_map = {}
    rows = []

    # まず Insider と Trends を事前ロード
    insider_map = load_insider_map(DATE)   # {SYM: 0..1}
    trends_map  = load_trends_scores(DATE) # {SYM: 0..1}

    # --- データ取得 & 出来高アノマリー素点 ---
    vol_ratio_raw = {}
    for sym in symbols:
        try:
            df = get_eod_range(sym, start_short, end)
            if df is not None and not df.empty:
                df = df.rename(columns=lambda c: str(c).strip().lower())
                if "adj close" in df.columns and "close" not in df.columns:
                    df["close"] = df["adj close"]
                if "date" not in df.columns and hasattr(df.index, "dtype"):
                    try:
                        df = df.reset_index().rename(columns={"index":"date"})
                    except Exception:
                        pass
            recent_map[sym] = df
            vr = compute_volume_anomaly(df)  # 素点（>1で多い）
            vol_ratio_raw[sym] = vr
        except Exception as e:
            print(f"[WARN] {sym}: {e}", file=sys.stderr)
            recent_map[sym] = pd.DataFrame()
            vol_ratio_raw[sym] = None

    # --- 出来高アノマリーの 0–1 化 ---
    norm_fn = minmax_norm([v for v in vol_ratio_raw.values() if v is not None])
    vol_score = {sym: float(np.clip(norm_fn(vol_ratio_raw.get(sym)), 0.0, 1.0)) for sym in symbols}

    # --- 行生成（スコア合成用に拡張性を持たせる） ---
    # 重みは後で変えやすいよう dict で持つ
    default_weights = {
        "volume_anomaly": 0.5,
        "insider_momo":   0.3,
        "trends_breakout":0.2,
    }

    for idx, t in uni.iterrows():
        sym = str(t["symbol"]).strip().upper()
        name = str(t.get("name","")).strip()
        theme = str(t.get("theme","")).strip()

        comps = {
            "volume_anomaly": vol_score.get(sym, 0.0),     # 0..1
            "insider_momo":   insider_map.get(sym, 0.0),   # 0..1
            "trends_breakout":trends_map.get(sym, 0.0),    # 0..1
        }
        weights = default_weights.copy()

        total_pts = combine_to_1000(comps, weights)

        rows.append({
            "symbol": sym,
            "name": name,
            "theme": theme,
            "score_components": comps,
            "score_weights": weights,
            "total_score": round(total_pts, 2),
            # 個別に見せるためにも残す
            "vol_anomaly_score": comps["volume_anomaly"],
            "insider_momo": comps["insider_momo"],
            "trends_breakout": comps["trends_breakout"],
        })

    if not rows:
        out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
        out_json_dir.mkdir(parents=True, exist_ok=True)
        with open(out_json_dir / "top10.json", "w") as f:
            json.dump([], f, indent=2)
        mark_fixed_costs(spend)
        print(f"Generated top10 for {DATE}: 0 symbols (no rows)")
        return

    # --- Top10 抽出 ---
    rows.sort(key=lambda x: x["total_score"], reverse=True)
    top10 = rows[:10]

    # --- JSON 出力 ---
    out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json_dir / "top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    # --- 週足3ヶ月チャート生成（MOCK時はスキップ） ---
    if not MOCK_MODE and top10:
        for r in top10:
            try:
                hist = recent_map.get(r["symbol"])
                if hist is None or hist.empty:
                    print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr)
                    continue
                save_chart_png_weekly_3m(r["symbol"], hist, OUT_DIR, DATE)
            except Exception as e:
                print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    mark_fixed_costs(spend)
    print(f"Generated top10 for {DATE}: {len(top10)} symbols")

if __name__ == "__main__":
    main()
