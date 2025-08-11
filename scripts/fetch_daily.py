
#!/usr/bin/env python3
import os, sys, json, math, csv, pathlib, datetime, random
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Config (env) ----
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or datetime.date.today().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"  # default true so it runs immediately

# Budget guard (hard cutoff)
BUDGET_JPY_MAX = float(os.getenv("BUDGET_JPY_MAX", "10000"))
SPEND_FILE = os.getenv("SPEND_FILE", f"{OUT_DIR}/data/spend.json")
MANUAL_DAILY_COST_JPY = float(os.getenv("MANUAL_DAILY_COST_JPY", "0"))
TIINGO_MONTHLY_JPY = float(os.getenv("TIINGO_MONTHLY_JPY", "1600"))

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
    p = pathlib.Path(SPEND_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
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
    mkey = month_key(DATE)
    month = spend.setdefault(mkey, {"items": [], "total_jpy": 0})
    if not month.get("tiingo_month_mark"):
        add_spend(spend, mkey, DATE, TIINGO_MONTHLY_JPY, "Tiingo monthly flat")
        month["tiingo_month_mark"] = True
    if MANUAL_DAILY_COST_JPY > 0:
        add_spend(spend, mkey, DATE, MANUAL_DAILY_COST_JPY, "Variable API usage (manual)")
    save_spend(spend)

def tiingo_eod(symbol, start, end):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"token": TIINGO_TOKEN, "startDate": start, "endDate": end, "resampleFreq":"daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def compute_metrics(df):
    df = pd.DataFrame(df)
    if df.empty:
        return None
    df["volume"] = df["volume"].fillna(0)
    df["close"] = df["close"].fillna(method="ffill")
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    today = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None
    pct_change = (today["close"] - prev["close"]) / prev["close"] if prev is not None and prev["close"] else 0.0
    vol_ratio = (today["volume"] / today["vol_sma20"]) if today["vol_sma20"] and not math.isnan(today["vol_sma20"]) else 1.0
    return float(pct_change), float(vol_ratio)

def mock_metrics():
    rnd = random.Random(DATE)
    pct = rnd.uniform(-0.05, 0.08)
    vol = rnd.uniform(0.5, 6.0)
    return pct, vol

def main():
    ok, spend = budget_check()
    if not ok:
        return
    uni = pd.read_csv(UNIVERSE_CSV)
    rows = []
    end = DATE
    start = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=120)).isoformat()

    for _, t in uni.iterrows():
        symbol = t["symbol"]
        try:
            if MOCK_MODE or not TIINGO_TOKEN:
                pct_change, vol_ratio = mock_metrics()
            else:
                data = tiingo_eod(symbol, start, end)
                metrics = compute_metrics(data)
                if not metrics:
                    pct_change, vol_ratio = 0.0, 1.0
                else:
                    pct_change, vol_ratio = metrics
            rows.append({
                "symbol": symbol,
                "name": t["name"],
                "theme": t["theme"],
                "pct_change": pct_change,
                "news_count": 0,
                "vol_ratio": vol_ratio,
                "tech_note": "Auto tech note TBD",
                "ir_note": "IR/News summary TBD",
            })
        except Exception as e:
            print(f"[WARN] {symbol}: {e}", file=sys.stderr)
            continue

    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.0 for v in vals]
    price_norm = norm([r["pct_change"] for r in rows]) if rows else []
    vol_norm   = norm([r["vol_ratio"] for r in rows]) if rows else []
    for i, r in enumerate(rows):
        r["score"] = 0.6*price_norm[i] + 0.4*vol_norm[i]
    rows.sort(key=lambda x: x["score"], reverse=True)
    top10 = rows[:10]

    out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json_dir / "top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    mark_fixed_costs(spend)
    print(f"Generated top10 for {DATE}: {len(top10)} symbols")

def tiingo_eod_range(symbol, start, end):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"token": TIINGO_TOKEN, "startDate": start, "endDate": end, "resampleFreq": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def save_chart_png(symbol, df, out_dir, date_iso):
    # df: columns expect date/close/volume etc.
    if df.empty:
        return
    charts_dir = pathlib.Path(out_dir) / "charts" / date_iso
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 整形
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()
    d["ma200"] = d["close"].rolling(200).mean()

    # 価格チャート（デフォ色／凡例なしでスッキリ）
    plt.figure(figsize=(9, 4.8), dpi=120)
    plt.plot(d["date"], d["close"], linewidth=1.2)
    plt.plot(d["date"], d["ma20"], linewidth=0.9)
    plt.plot(d["date"], d["ma50"], linewidth=0.9)
    plt.plot(d["date"], d["ma200"], linewidth=0.9)
    plt.title(f"{symbol} — 3Y Daily")
    plt.tight_layout()
    out = charts_dir / f"{symbol}.png"
    plt.savefig(out)
    plt.close()

# rows.sort(...) の少し後にある top10 作成の直後あたり
top10 = rows[:10]

# チャート保存（実データ時のみ）
if not MOCK_MODE and TIINGO_TOKEN:
    start3y = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=365*3+30)).isoformat()
    for r in top10:
        try:
            hist = tiingo_eod_range(r["symbol"], start3y, DATE)
            # Tiingoは 'adjClose' あり。なければ 'close' を使う
            if "adjClose" in hist.columns:
                hist["close"] = hist["adjClose"]
            save_chart_png(r["symbol"], hist, OUT_DIR, DATE)
        except Exception as e:
            print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
