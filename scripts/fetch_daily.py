#!/usr/bin/env python3
import os, sys, json, math, pathlib, random
import pandas as pd
import requests

# charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 先頭のDATE定義を置き換え
import datetime
from zoneinfo import ZoneInfo

def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 引け後18:00未満なら、まだ当日速報は不完全 → 前営業日にする
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    # 週末処理（土=5, 日=6 → 直近の金曜へ）
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

# ---- Config (env) ----
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()  # yfinance | tiingo
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")  # tiingo時のみ使用
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"  # ← 実データ化ではfalse

# ---- Budget Guard (hard cutoff) ----
BUDGET_JPY_MAX = float(os.getenv("BUDGET_JPY_MAX", "10000"))
SPEND_FILE = os.getenv("SPEND_FILE", f"{OUT_DIR}/data/spend.json")
MANUAL_DAILY_COST_JPY = float(os.getenv("MANUAL_DAILY_COST_JPY", "0"))
# 固定費はプロバイダごとに分ける（yfinance=0, tiingo=1600円想定）
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
    # yfinanceは0円なので何もしない。将来有料APIに切替えたら固定費を1度だけ計上
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

# ---------- Data Providers ----------
def tiingo_eod_range(symbol, start, end):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"token": TIINGO_TOKEN, "startDate": start, "endDate": end, "resampleFreq": "daily"}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    df = pd.DataFrame(r.json())
    # Tiingo: columns include 'adjClose' あり。無い場合は 'close'
    if "adjClose" in df.columns:
        df = df.rename(columns={"adjClose":"close"})
    return df

def yfi_eod_range(symbol, start, end):
    import yfinance as yf
    # 1日バッファ（タイムゾーンずれ対策）
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt = datetime.date.fromisoformat(end) + datetime.timedelta(days=1)

    # ★ threads=False / interval='1d' を明示。emptyならリトライ。
    for attempt in range(2):
        df = yf.download(
            symbol,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,   # ← 重要：Runnerでの不安定さ回避
        )
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        # 最後の砦：Ticker().history() で再取得
        tkr = yf.Ticker(symbol)
        hist = tkr.history(
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if hist is None or hist.empty:
            return pd.DataFrame()
        df = hist

    df = df.reset_index().rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date","open","high","low","close","volume"]]

def get_eod_range(symbol, start, end):
    if MOCK_MODE:
        # 120日分のダミーを作る
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
    # default: yfinance
    return yfi_eod_range(symbol, start, end)

# ---------- Metrics & Charts ----------
def compute_metrics(df):
    import numpy as np
    import pandas as pd

    if df is None or len(df) == 0:
        return None

    d = df.copy()

    # 数値化＆欠損処理（明示的に！）
    d["volume"] = pd.to_numeric(d.get("volume"), errors="coerce").fillna(0)
    d["close"]  = pd.to_numeric(d.get("close"),  errors="coerce").ffill()

    if len(d) < 2:
        return 0.0, 1.0

    # 直近と前日の終値を「スカラー」で取得
    today_close = float(d["close"].iloc[-1]) if not pd.isna(d["close"].iloc[-1]) else np.nan
    prev_close  = float(d["close"].iloc[-2]) if not pd.isna(d["close"].iloc[-2]) else np.nan

    # pct_change（0割＆NaN回避）
    if pd.isna(today_close) or pd.isna(prev_close) or prev_close == 0:
        pct_change = 0.0
    else:
        pct_change = (today_close - prev_close) / prev_close

    # 異常出来高比（20日平均の明示計算）
    vol_sma20_series = d["volume"].rolling(20).mean()
    vol_sma20 = float(vol_sma20_series.iloc[-1]) if not pd.isna(vol_sma20_series.iloc[-1]) else np.nan
    today_vol = float(d["volume"].iloc[-1]) if not pd.isna(d["volume"].iloc[-1]) else 0.0

    vol_ratio = 1.0 if (pd.isna(vol_sma20) or vol_sma20 == 0) else (today_vol / vol_sma20)

    return float(pct_change), float(vol_ratio)

def save_chart_png(symbol, df, out_dir, date_iso):
    if df is None or df.empty:
        return
    charts_dir = pathlib.Path(out_dir) / "charts" / date_iso
    charts_dir.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()
    d["ma200"] = d["close"].rolling(200).mean()

    plt.figure(figsize=(9, 4.8), dpi=120)
    plt.plot(d["date"], d["close"], linewidth=1.2)
    plt.plot(d["date"], d["ma20"], linewidth=0.9)
    plt.plot(d["date"], d["ma50"], linewidth=0.9)
    plt.plot(d["date"], d["ma200"], linewidth=0.9)
    plt.title(f"{symbol} — 3Y Daily")
    plt.tight_layout()
    out = charts_dir / f"{symbol}.png"
    plt.savefig(out); plt.close()

def main():
    ok, spend = budget_check()
    if not ok: return

    uni = pd.read_csv(UNIVERSE_CSV)
    rows = []
    end = DATE
    # 4ヶ月分（SMA20用）を最低取得、その後チャート用に3年を別取得
    start_short = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=120)).isoformat()

    for _, t in uni.iterrows():
        symbol = t["symbol"]
        try:
            df = get_eod_range(symbol, start_short, end)
            metrics = compute_metrics(df)
            if not metrics: pct_change, vol_ratio = 0.0, 1.0
            else: pct_change, vol_ratio = metrics
            rows.append({
                "symbol": symbol, "name": t["name"], "theme": t["theme"],
                "pct_change": pct_change, "news_count": 0, "vol_ratio": vol_ratio,
                "tech_note": "Auto tech note TBD", "ir_note": "IR/News summary TBD",
            })
        except Exception as e:
            print(f"[WARN] {symbol}: {e}", file=sys.stderr)
            continue

    # スコアリング
    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.0 for v in vals]
    price_norm = norm([r["pct_change"] for r in rows]) if rows else []
    vol_norm   = norm([r["vol_ratio"] for r in rows]) if rows else []
    for i, r in enumerate(rows):
        r["score"] = 0.6*price_norm[i] + 0.4*vol_norm[i]
    rows.sort(key=lambda x: x["score"], reverse=True)
    top10 = rows[:10]

    # チャート生成（実データ時のみ）
    if not MOCK_MODE:
        start3y = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=365*3+30)).isoformat()
        for r in top10:
            try:
                hist = get_eod_range(r["symbol"], start3y, end)
                save_chart_png(r["symbol"], hist, OUT_DIR, DATE)
            except Exception as e:
                print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    # 出力
    out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json_dir / "top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    # 無料なので固定費計上はしない（将来有料APIに替える場合に有効化）
    mark_fixed_costs(spend)
    print(f"Generated top10 for {DATE}: {len(top10)} symbols")

if __name__ == "__main__":
    main()
