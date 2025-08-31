#!/usr/bin/env python3
import os, json, time, random
import pandas as pd
from pytrends.request import TrendReq

OUT_DIR = os.environ.get("OUT_DIR","site")
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV","data/universe.csv")
REPORT_DATE = os.environ.get("REPORT_DATE")
GEO = os.environ.get("TRENDS_GEO","US")
TIMEFRAME = os.environ.get("TRENDS_TIMEFRAME","today 3-m")
BATCH = int(os.environ.get("TRENDS_BATCH","3"))
SLEEP = float(os.environ.get("TRENDS_SLEEP","4"))
JITTER = float(os.environ.get("TRENDS_JITTER","1.5"))

def load_universe(path):
    df = pd.read_csv(path)
    # 必須: symbol, name
    if "symbol" not in df.columns:
        raise RuntimeError("universe.csv must have 'symbol' column")
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    return df

def main():
    os.makedirs(f"{OUT_DIR}/data/trends", exist_ok=True)
    df_u = load_universe(UNIVERSE_CSV)
    names = df_u["name"].tolist()

    py = TrendReq(hl="en-US", tz=360)
    results = {}
    # BATCHごとに分割
    for i in range(0, len(names), BATCH):
        group = names[i:i+BATCH]
        try:
            py.build_payload(group, cat=0, timeframe=TIMEFRAME, geo=GEO, gprop="")
            df = py.interest_over_time()
            if df.empty:
                continue
            # 各銘柄の直近窓（最後の値）を採用
            latest = df.iloc[-1][group]
            for k, v in latest.items():
                results[k] = float(v)
        except Exception as e:
            print(f"[WARN] Trends error for {group}: {e}")
        time.sleep(SLEEP + random.random()*JITTER)

    # 正規化スコア(0-1)
    if results:
        s = pd.Series(results)
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        score = {k: float(s_norm[k]) for k in s_norm.index}
    else:
        score = {}

    payload = {
        "raw_last": results,          # 直近のinterest
        "score_0_1": score            # 0-1正規化
    }

    with open(f"{OUT_DIR}/data/trends/latest.json","w") as f:
        json.dump(payload, f, indent=2)
    if REPORT_DATE:
        os.makedirs(f"{OUT_DIR}/data/{REPORT_DATE}", exist_ok=True)
        with open(f"{OUT_DIR}/data/{REPORT_DATE}/trends.json","w") as f:
            json.dump(payload, f, indent=2)

    print(f"[TRENDS] saved: {OUT_DIR}/data/trends/latest.json and {OUT_DIR}/data/{REPORT_DATE}/trends.json ({len(results)} symbols)")

if __name__ == "__main__":
    main()
