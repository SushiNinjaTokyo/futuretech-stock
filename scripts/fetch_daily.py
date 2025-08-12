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
    # タイムゾーンずれ対策で1〜2日バッファ
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    df = None
    for attempt in range(2):
        tmp = yf.download(
            symbol,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,   # CIでの不安定回避
        )
        if tmp is not None and not tmp.empty:
            df = tmp
            break

    if df is None or df.empty:
        tkr = yf.Ticker(symbol)
        tmp = tkr.history(
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if tmp is None or tmp.empty:
            return pd.DataFrame()
        df = tmp

    df = df.reset_index()

    # すべて小文字に統一しておく（これ超大事）
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 列の別名を吸収
    name_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "close",   # ← adj close を close として扱う
        "adjclose": "close",
        "volume": "volume",
    }
    out = {}
    for col in df.columns:
        if col in name_map and name_map[col] not in out:
            out[name_map[col]] = df[col]

    # 必須列チェック（close/volume が無ければ空で返す）
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

    return result[["date", "open", "high", "low", "close", "volume"]]


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

    if df is None or df.empty:
        return None

    # 必須列がなければスキップ
    for col in ("close", "volume"):
        if col not in df.columns:
            return None

    d = df.copy()
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce").ffill()
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)

    if len(d) < 2:
        return 0.0, 1.0

    today_close = d["close"].iloc[-1]
    prev_close  = d["close"].iloc[-2]
    if pd.isna(today_close) or pd.isna(prev_close) or float(prev_close) == 0.0:
        pct_change = 0.0
    else:
        pct_change = (float(today_close) - float(prev_close)) / float(prev_close)

    vol_sma20 = d["volume"].rolling(20).mean().iloc[-1]
    today_vol = d["volume"].iloc[-1]
    if pd.isna(vol_sma20) or float(vol_sma20) == 0.0:
        vol_ratio = 1.0
    else:
        vol_ratio = float(today_vol) / float(vol_sma20)

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
    top10 = []  # ← 先に空で定義しておく（これで未定義は起きない）

    end = DATE
    # 3か月（90日）のみ取得
    start_short = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=90)).isoformat()

    recent_map = {}  # 90日データをチャートで再利用するために保存

    # ----- データ取得 → 指標算出 -----
    for _, t in uni.iterrows():
        symbol = t["symbol"]
        try:
            df = get_eod_range(symbol, start_short, end)
            recent_map[symbol] = df
            metrics = compute_metrics(df)
            if not metrics:
                # 取れなければスキップ（ここで rows へは入れない）
                print(f"[WARN] skip (no metrics) {symbol}", file=sys.stderr)
                continue
            pct_change, vol_ratio = metrics
            rows.append({
                "symbol": symbol, "name": t["name"], "theme": t["theme"],
                "pct_change": pct_change, "news_count": 0, "vol_ratio": vol_ratio,
                "tech_note": "Auto tech note TBD", "ir_note": "IR/News summary TBD",
            })
        except Exception as e:
            print(f"[WARN] {symbol}: {e}", file=sys.stderr)
            continue

    # ----- rowsが空なら、空のtop10を書いて正常終了 -----
    if not rows:
        out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
        out_json_dir.mkdir(parents=True, exist_ok=True)
        with open(out_json_dir / "top10.json", "w") as f:
            json.dump(top10, f, indent=2)  # 空リストを書き出す
        mark_fixed_costs(spend)
        print(f"Generated top10 for {DATE}: 0 symbols (no rows)")
        return  # ← ここで終了。下のチャート処理に進まない

    # ----- スコアリング → Top10確定 -----
    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.0 for v in vals]

    price_norm = norm([r["pct_change"] for r in rows])
    vol_norm   = norm([r["vol_ratio"] for r in rows])

    for i, r in enumerate(rows):
        r["score"] = 0.6 * price_norm[i] + 0.4 * vol_norm[i]

    rows.sort(key=lambda x: x["score"], reverse=True)
    top10 = rows[:10]  # ← ここで必ず代入される

    # ----- JSON出力（先に書いておく）-----
    out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json_dir / "top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    # ----- 週足3ヶ月チャート生成（MOCK時はスキップ／再取得なし）-----
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
