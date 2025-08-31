#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DII fetcher / scorer (robust & cache-first)

目的:
- FINRAのATS(ダーク)出来高をベースに「機関の静かな蓄積」を0..1にスコア化。
- 失敗時は最新保存済みJSON (site/data/dii/latest.json) をそのまま再出力して安全運転。
- 週次データ想定（FINRA ATSは週次公開）。直近週の水準と過去12週のベースを比較し、宇宙内でのパーセンタイルに変換。

出力:
- site/data/dii/latest.json
- site/data/<DATE>/dii.json
  items: {
    "SYMBOL": {
      "as_of": "YYYY-MM-DD",
      "raw_signal": float,
      "score_0_1": float,         # 宇宙内パーセンタイル
      "ats_share_ratio": float,    # 直近週のダーク比率(概念)
      "weeks": int,                # 集計週数
      "note": "source or fallback"
    }, ...
  }

環境変数:
- OUT_DIR (default: site)
- UNIVERSE_CSV (default: data/universe.csv)
- REPORT_DATE (ET基準のレポート日、無ければ et_market_date と同様ロジック)
- DII_LOOKBACK_WEEKS (default 12)
- DII_RECENT_WEEKS (default 4)
- DII_RETRIES / DII_TIMEOUT / DII_SLEEP / DII_JITTER
- DII_SOURCE: "finra_api" | "skip"    # "skip"なら取得スキップして latest.json のみ採用
"""

import os, json, time, random, datetime, pathlib, math
from zoneinfo import ZoneInfo
import pandas as pd
import requests

OUT_DIR = os.getenv("OUT_DIR","site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV","data/universe.csv")
DII_LOOKBACK_WEEKS = int(os.getenv("DII_LOOKBACK_WEEKS","12"))
DII_RECENT_WEEKS   = int(os.getenv("DII_RECENT_WEEKS","4"))
DII_SOURCE = os.getenv("DII_SOURCE","finra_api").lower()

RETRIES=int(os.getenv("DII_RETRIES","3"))
TIMEOUT=int(os.getenv("DII_TIMEOUT","25"))
SLEEP=float(os.getenv("DII_SLEEP","1.0"))
JITTER=float(os.getenv("DII_JITTER","0.6"))

DATE = os.getenv("REPORT_DATE")
if not DATE:
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    if now_et.hour < 20:
        d -= datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    DATE = d.isoformat()

def jitter_sleep(base=SLEEP, jitter=JITTER):
    time.sleep(base + random.uniform(0, jitter))

def req_get(url, params=None, headers=None):
    last=None
    for i in range(1, RETRIES+1):
        try:
            r = requests.get(url, params=params, headers=headers or {"User-Agent":"futuretech-stock/1.0 (dii)"},
                             timeout=TIMEOUT)
            if r.status_code==429:
                wait = min(90, 6*(2**(i-1))) + random.uniform(0,1.2)
                print(f"[DII] 429 cooldown {wait:.1f}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last=e
            wait = 1.2*i + random.uniform(0,1.0)
            print(f"[DII] retry {i}/{RETRIES} after error: {e} (sleep {wait:.1f}s)")
            time.sleep(wait)
    raise last

def pct_rank(vals, x):
    arr = sorted([v for v in vals if isinstance(v,(int,float)) and not math.isnan(v)])
    if not arr or x is None or (isinstance(x,float) and math.isnan(x)):
        return 0.0
    import bisect
    k = bisect.bisect_right(arr, x)
    return k/len(arr)

def load_json_safe(path, default):
    try:
        p=pathlib.Path(path)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default

def ensure_dirs(date_iso):
    (pathlib.Path(OUT_DIR)/"data"/date_iso).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT_DIR)/"data"/"dii").mkdir(parents=True, exist_ok=True)

# ---- ここでデータ取得ロジック ----
def fetch_finra_ats_api(symbols):
    """
    FINRA Data APIを想定したシンプルな集計器。
    実運用では公式仕様に合わせてフィールド名を調整してください。
    ここでは 'week' (ISO日付), 'ats_volume', 'total_volume' を得られると仮定。
    取得できない/失敗時は{}を返し、上位で fallback を実行。
    """
    base_url = "https://api.finra.org/data/ats?symbol="
    out = {}
    for sym in symbols:
        try:
            r = req_get(base_url + sym)
            jitter_sleep()
            j = r.json()
            if not isinstance(j, list) or not j:
                continue
            df = pd.DataFrame(j)
            # 欠損防御
            for col in ("week","ats_volume","total_volume"):
                if col not in df.columns: 
                    raise ValueError(f"missing column {col}")
            df["week"] = pd.to_datetime(df["week"])
            df = df.sort_values("week")
            # 直近 LOOKBACK 週のみ
            df = df.tail(DII_LOOKBACK_WEEKS)
            if df.empty: 
                continue
            # 週ごとの ATS 比率
            df["ratio"] = (pd.to_numeric(df["ats_volume"], errors="coerce") /
                           pd.to_numeric(df["total_volume"], errors="coerce")).clip(lower=0, upper=1)
            # 直近RECENTの平均 vs 過去LOOKBACKの中央値
            recent = df["ratio"].tail(DII_RECENT_WEEKS).mean()
            base   = df["ratio"].median()
            raw_signal = (recent - base)  # 差分（>0で直近ダーク比率↑）
            ats_share_ratio = float(df["ratio"].tail(1).mean())
            out[sym] = {
                "as_of": DATE,
                "raw_signal": float(raw_signal) if pd.notna(raw_signal) else 0.0,
                "ats_share_ratio": float(ats_share_ratio) if pd.notna(ats_share_ratio) else 0.0,
                "weeks": int(len(df)),
                "note": "finra_api"
            }
        except Exception as e:
            print(f"[DII] fail {sym}: {e}")
    return out

def main():
    ensure_dirs(DATE)
    uni = pd.read_csv(UNIVERSE_CSV)
    uni["symbol"] = uni["symbol"].astype(str).str.upper().str.strip()
    symbols = uni["symbol"].tolist()

    items = {}
    used_source = "fallback_latest"

    if DII_SOURCE == "finra_api":
        try:
            items = fetch_finra_ats_api(symbols)
            used_source = "finra_api"
        except Exception as e:
            print(f"[DII] finra_api failed globally: {e}")

    if not items:
        # fallback to latest
        latest = load_json_safe(pathlib.Path(OUT_DIR)/"data"/"dii"/"latest.json", {"items":{}})
        items = latest.get("items", {})
        used_source = "latest_json"

    # 0–1 正規化（宇宙内パーセンタイル）
    raws = [v.get("raw_signal",0.0) for v in items.values()]
    for sym, rec in items.items():
        rs = rec.get("raw_signal",0.0)
        rec["score_0_1"] = round(pct_rank(raws, rs), 6)
        rec["as_of"] = rec.get("as_of") or DATE
        rec["note"] = rec.get("note") or used_source

    payload = {"as_of": DATE, "lookback_weeks": DII_LOOKBACK_WEEKS, "recent_weeks": DII_RECENT_WEEKS, "items": items}

    out_latest = pathlib.Path(OUT_DIR)/"data"/"dii"/"latest.json"
    out_today  = pathlib.Path(OUT_DIR)/"data"/DATE/"dii.json"
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_today.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, indent=2))
    out_today.write_text(json.dumps(payload, indent=2))
    print(f"[DII] saved: {out_latest} and {out_today} (symbols={len(items)}) source={used_source}")

if __name__ == "__main__":
    main()
