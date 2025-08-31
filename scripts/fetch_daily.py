#!/usr/bin/env python3
import os, json, math
import pandas as pd
import numpy as np
import yfinance as yf

OUT_DIR = os.environ.get("OUT_DIR","site")
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV","data/universe.csv")
REPORT_DATE = os.environ.get("REPORT_DATE")
MOCK_MODE = os.environ.get("MOCK_MODE","false").lower()=="true"

W_VOL = float(os.environ.get("WEIGHT_VOL_ANOM","0.25"))
W_DII = float(os.environ.get("WEIGHT_DII","0.25"))
W_TR = float(os.environ.get("WEIGHT_TRENDS","0.30"))
W_NE = float(os.environ.get("WEIGHT_NEWS","0.20"))

def load_universe(path):
    df = pd.read_csv(path)
    if "symbol" not in df.columns: raise RuntimeError("universe.csv needs 'symbol'")
    if "name" not in df.columns: df["name"]=df["symbol"]
    return df[["symbol","name"]]

def safe_hist(symbol, period="3mo", interval="1d"):
    try:
        t = yf.Ticker(symbol)
        h = t.history(period=period, interval=interval, auto_adjust=False)
        if isinstance(h, pd.DataFrame) and not h.empty:
            return h
    except Exception as e:
        print(f"[WARN] yfinance failed for {symbol}: {e}")
    return pd.DataFrame()

def compute_volume_anomaly(hist: pd.DataFrame):
    # hist: columns like ["Open","High","Low","Close","Volume"]
    if hist is None or hist.empty or "Volume" not in hist.columns:
        return 0.0
    vol = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    if len(vol) < 5:
        return 0.0
    cur = float(vol.iloc[-1])
    ma20 = float(vol.rolling(20, min_periods=5).mean().iloc[-1])
    if ma20 <= 0: 
        return 0.0
    ratio = cur / ma20
    # 0~1へ圧縮（1以上で頭打ち気味に）
    score = 1.0 - math.exp(-min(ratio, 5.0))
    return max(0.0, min(score, 1.0))

def load_component_score(path):
    if not os.path.exists(path): 
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
        return payload.get("score_0_1", {})
    except Exception:
        return {}

def normalize_series(d: dict, keys, default=0.0):
    # 与えられたkeys順に0~1スコア辞書を返す（無ければdefault）
    return {k: float(d.get(k, default)) for k in keys}

def main():
    os.makedirs(f"{OUT_DIR}/data/top10", exist_ok=True)
    if REPORT_DATE:
        os.makedirs(f"{OUT_DIR}/data/{REPORT_DATE}", exist_ok=True)

    u = load_universe(UNIVERSE_CSV)
    # 参照データ（会社名keyで保存しているためnameで引く）
    trends = load_component_score(f"{OUT_DIR}/data/trends/latest.json")
    news   = load_component_score(f"{OUT_DIR}/data/news/latest.json")
    dii    = load_component_score(f"{OUT_DIR}/data/dii/latest.json")

    results = []
    for _, row in u.iterrows():
        sym = row["symbol"]
        name = row["name"]

        if MOCK_MODE:
            hist = pd.DataFrame({"Volume":[1,2,3,4,5]})
        else:
            hist = safe_hist(sym)

        vol_score = compute_volume_anomaly(hist)

        # name基準で一致させる（fetch_trends/news側がnameキー）
        tr = float(trends.get(name, 0.0))
        ne = float(news.get(name, 0.0))
        di = float(dii.get(name, 0.0))

        final = W_VOL*vol_score + W_TR*tr + W_NE*ne + W_DII*di

        price = float(hist["Close"].iloc[-1]) if ("Close" in hist.columns and not hist.empty) else None
        change = None
        if price and "Close" in hist.columns and len(hist["Close"])>=2:
            prev = float(hist["Close"].iloc[-2])
            if prev:
                change = (price/prev - 1.0)

        results.append({
            "symbol": sym,
            "name": name,
            "vol_anom_0_1": round(vol_score, 6),
            "trends_0_1": round(tr, 6),
            "news_0_1": round(ne, 6),
            "dii_0_1": round(di, 6),
            "final_score_0_1": round(final, 6),
            "price": price,
            "pct_change_1d": change
        })

    # 最終スコアでソート
    results = sorted(results, key=lambda x: x["final_score_0_1"], reverse=True)
    top10 = results[:10]

    payload = {
        "date": REPORT_DATE,
        "universe_size": int(len(u)),
        "items": top10
    }

    # latest と date への二重保存（テンプレートは両方を参照可能）
    with open(f"{OUT_DIR}/data/top10/latest.json","w") as f:
        json.dump(payload, f, indent=2)
    if REPORT_DATE:
        with open(f"{OUT_DIR}/data/top10/{REPORT_DATE}.json","w") as f:
            json.dump(payload, f, indent=2)

    print(f"[INFO] loaded items: trends={len(trends)} news={len(news)} dii={len(dii)}")
    print(f"Generated top10 for {REPORT_DATE}: {len(top10)} symbols (universe={len(u)})")

if __name__ == "__main__":
    main()
