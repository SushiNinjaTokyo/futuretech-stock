#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, json, pathlib, math, datetime, itertools
import pandas as pd
from zoneinfo import ZoneInfo
from pytrends.request import TrendReq

OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
DATE = os.getenv("REPORT_DATE") or datetime.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
# パラメータ（必要に応じて環境変数で変更可）
TRENDS_GEO = os.getenv("TRENDS_GEO", "US")          # 地域
TRENDS_TIMEFRAME = os.getenv("TRENDS_TIMEFRAME", "today 3-m")
TRENDS_BATCH = int(os.getenv("TRENDS_BATCH", "5"))  # 一度に投げるクエリ数（<=5）
TRENDS_SLEEP = float(os.getenv("TRENDS_SLEEP", "2.0"))

def load_universe():
    df = pd.read_csv(UNIVERSE_CSV)
    # symbol, name が理想。name 無ければフォールバック
    df["query"] = df.apply(lambda r: build_query(r), axis=1)
    return df[["symbol", "name", "query"]]

def build_query(row):
    name = str(row.get("name", "") or "").strip()
    sym = str(row.get("symbol", "") or "").strip().upper()
    if name and name.lower() not in ("nan", "none"):
        return name
    # フォールバック：ティッカー + 補助語
    return f"{sym} stock"

def pct_rank(vals, x):
    arr = sorted([v for v in vals if isinstance(v, (int,float)) and not math.isnan(v)])
    if not arr or x is None or math.isnan(x): return 0.0
    import bisect
    k = bisect.bisect_right(arr, x)
    return k / len(arr)

def compute_breakout_score(series: pd.Series):
    """ 直近7日平均 / それ以前8週中央値（0除算防止で微小値） """
    if series is None or series.empty: return None
    s = series.dropna().astype(float)
    if len(s) < 40:  # 3ヶ月なら十分あるはずだが保険
        return None
    recent = s.tail(7).mean()
    prior = s.iloc[:-7]
    # 8週(=56日)以内が望ましいがデータ長に応じ柔軟に中央値
    base = prior.tail(56).median() if len(prior) >= 56 else prior.median()
    if base is None or base <= 0:
        base = 1e-9
    return float(recent / base)

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk: break
        yield chunk

def main():
    uni = load_universe()
    py = TrendReq(hl="en-US", tz=360, retries=2, backoff_factor=0.2)
    results = {}
    for batch in batched(uni.to_dict("records"), TRENDS_BATCH):
        kw_list = [r["query"] for r in batch]
        py.build_payload(kw_list=kw_list, timeframe=TRENDS_TIMEFRAME, geo=TRENDS_GEO)
        df = py.interest_over_time()
        # df: index=date, columns=queries (+ 'isPartial')
        if df is None or df.empty:
            time.sleep(TRENDS_SLEEP); continue
        if "isPartial" in df.columns: df = df.drop(columns=["isPartial"])
        # 各キーワードでスコア算出
        for r in batch:
            q = r["query"]
            sym = r["symbol"]
            series = df.get(q)
            score = compute_breakout_score(series) if series is not None else None
            results[sym] = {
                "query": q,
                "raw_breakout": score,  # >1 でブレイク
            }
        time.sleep(TRENDS_SLEEP)

    # パーセンタイル（宇宙内正規化 0-1）
    raws = [v.get("raw_breakout") for v in results.values() if v.get("raw_breakout") is not None]
    for sym, rec in results.items():
        rb = rec.get("raw_breakout")
        if rb is None:
            rec["score_0_1"] = 0.0
        else:
            # 1.0（=ベースライン）をまたぐので、log を取ってテールを押しつぶす手もあるが、まずは素直にパーセンタイル
            rec["score_0_1"] = pct_rank(raws, rb)

    # 保存
    out_latest = pathlib.Path(OUT_DIR) / "data" / "trends" / "latest.json"
    out_today  = pathlib.Path(OUT_DIR) / "data" / DATE / "trends.json"
    payload = {"as_of": DATE, "geo": TRENDS_GEO, "timeframe": TRENDS_TIMEFRAME, "items": results}
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_today.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, indent=2))
    out_today.write_text(json.dumps(payload, indent=2))
    print(f"[TRENDS] saved: {out_latest} and {out_today} ({len(results)} symbols)")

if __name__ == "__main__":
    main()
