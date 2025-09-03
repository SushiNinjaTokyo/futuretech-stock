#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Trends から各シンボルの相対ブレイクアウト指標(0..1)を生成。
結果: site/data/trends/latest.json と site/data/{DATE}/trends.json
"""
from __future__ import annotations
import os, sys, json, time, logging, random
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq

logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO"),
    format="%(asctime)sZ [%(levelname)s] [TRENDS] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trends")

OUT_DIR   = Path(os.getenv("OUT_DIR","site"))
UNIVERSE  = Path(os.getenv("UNIVERSE_CSV","data/universe.csv"))
REPORT    = os.getenv("REPORT_DATE") or pd.Timestamp.utcnow().date().isoformat()
GEO       = os.getenv("TRENDS_GEO","US")
TIMEFRAME = os.getenv("TRENDS_TIMEFRAME","today 3-m")
BATCH     = int(float(os.getenv("TRENDS_BATCH","3")))
SLEEP     = float(os.getenv("TRENDS_SLEEP","4"))
JITTER    = float(os.getenv("TRENDS_JITTER","1.5"))
BATCH_RETRIES = int(float(os.getenv("TRENDS_BATCH_RETRIES","4")))

def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE)
    # 必須列: symbol, query (なければ会社名or同一)
    if "query" not in df.columns:
        df["query"] = df.get("name", df["symbol"])
    return df[["symbol","query"]]

def breakout_score_from_series(s: pd.Series) -> float:
    """直近7日平均 / 直近90日平均、を 0..1 にクリップ"""
    if s.empty or s.max() == 0:
        return 0.0
    s = s.astype(float)
    last7  = s.tail(7).mean()
    last90 = s.tail(90).mean() if len(s) >= 90 else s.mean()
    ratio = (last7 / last90) if last90 > 0 else 0.0
    # 1.0 以上は徐々に飽和させる（対数圧縮）
    scaled = max(0.0, min(1.0, (ratio - 0.5) / 1.5))
    return float(round(scaled, 6))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_dir = OUT_DIR / "data" / REPORT
    date_dir.mkdir(parents=True, exist_ok=True)

    u = load_universe()
    pytr = TrendReq(hl="en-US", tz=0)
    items = []

    for i in range(0, len(u), BATCH):
        chunk = u.iloc[i:i+BATCH]
        kw_list = chunk["query"].tolist()
        for attempt in range(1, BATCH_RETRIES+1):
            try:
                pytr.build_payload(kw_list=kw_list, timeframe=TIMEFRAME, geo=GEO)
                df = pytr.interest_over_time()
                break
            except Exception as e:
                if attempt >= BATCH_RETRIES: 
                    log.warning("batch %s failed (%s). fill zeros.", i//BATCH, e)
                    df = pd.DataFrame(index=pd.date_range(end=pd.Timestamp.utcnow(), periods=1))
                    for q in kw_list: df[q]=0
                    break
                sl = SLEEP + random.random()*JITTER
                log.info("retry %d for chunk %d after %.1fs (%s)", attempt, i//BATCH, sl, e)
                time.sleep(sl)

        # score へ
        for _, row in chunk.iterrows():
            q = row["query"]
            sym = row["symbol"]
            ser = df[q] if q in df.columns else pd.Series(dtype=float)
            items.append({"symbol": sym, "breakout": breakout_score_from_series(ser)})

    payload = {"date": REPORT, "items": items, "meta": {"geo": GEO, "timeframe": TIMEFRAME}}
    (OUT_DIR/"data"/"trends").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR/"data"/"trends"/"latest.json","w") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(date_dir/"trends.json","w") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("saved: %s and %s (symbols=%d)", OUT_DIR/"data/trends/latest.json", date_dir/"trends.json", len(items))

if __name__=="__main__":
    main()
