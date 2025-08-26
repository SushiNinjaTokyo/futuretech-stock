#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Trends fetcher with robust 429 handling and urllib3 v2 shim.
- Batch size is configurable (default 3).
- Sleeps between batches with jitter.
- Detects 429 and performs exponential cool-down retries per batch.
"""

import os, time, json, pathlib, math, datetime, itertools, random
from typing import List, Dict

# ---- urllib3 v2 互換レイヤー（pytrends が method_whitelist を使う問題の回避）----
def _install_retry_compat_shim():
    try:
        from urllib3.util.retry import Retry as _Retry
        import inspect
        sig = inspect.signature(_Retry.__init__)
        if "allowed_methods" in sig.parameters:
            orig_init = _Retry.__init__
            def compat_init(self, *args, **kwargs):
                if "method_whitelist" in kwargs and "allowed_methods" not in kwargs:
                    kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
                return orig_init(self, *args, **kwargs)
            _Retry.__init__ = compat_init
    except Exception:
        pass
_install_retry_compat_shim()
# ----------------------------------------------------------------------

import pandas as pd
from zoneinfo import ZoneInfo
from pytrends.request import TrendReq
from requests.exceptions import RetryError
import urllib3

OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
DATE = os.getenv("REPORT_DATE") or datetime.datetime.now(ZoneInfo("America/New_York")).date().isoformat()

TRENDS_GEO = os.getenv("TRENDS_GEO", "US")
TRENDS_TIMEFRAME = os.getenv("TRENDS_TIMEFRAME", "today 3-m")
TRENDS_BATCH = int(os.getenv("TRENDS_BATCH", "3"))     # 安全第一なら 3 を推奨（<=5）
TRENDS_SLEEP = float(os.getenv("TRENDS_SLEEP", "4.0")) # バッチ間スリープ（秒）
TRENDS_JITTER = float(os.getenv("TRENDS_JITTER", "1.5"))  # 追加ジッタ（0〜この秒数）
TRENDS_COOLDOWN_BASE = float(os.getenv("TRENDS_COOLDOWN_BASE", "45"))  # 429 時の初回クールダウン（秒）
TRENDS_COOLDOWN_MAX  = float(os.getenv("TRENDS_COOLDOWN_MAX",  "180")) # 429 時の最大クールダウン（秒）
TRENDS_BATCH_RETRIES = int(os.getenv("TRENDS_BATCH_RETRIES", "4"))     # 429/エラー時のバッチ再試行回数

def load_universe():
    df = pd.read_csv(UNIVERSE_CSV)
    df["query"] = df.apply(lambda r: build_query(r), axis=1)
    return df[["symbol", "name", "query"]]

def build_query(row):
    name = str(row.get("name", "") or "").strip()
    sym = str(row.get("symbol", "") or "").strip().upper()
    if name and name.lower() not in ("nan", "none"):
        return name
    return f"{sym} stock"

def pct_rank(vals, x):
    arr = sorted([v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))])
    if not arr or x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    import bisect
    k = bisect.bisect_right(arr, x)
    return k / len(arr)

def compute_breakout_score(series: pd.Series):
    """ recent 7d mean / prior 8w median """
    if series is None or series.empty: return None
    s = series.dropna().astype(float)
    if len(s) < 40:  # 保険
        return None
    recent = s.tail(7).mean()
    prior = s.iloc[:-7]
    base = prior.tail(56).median() if len(prior) >= 56 else prior.median()
    if not base or base <= 0:
        base = 1e-9
    return float(recent / base)

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk: break
        yield chunk

def is_429_error(err: Exception) -> bool:
    txt = repr(err)
    return ("429" in txt) or ("Too Many Requests" in txt) or ("too many 429" in txt)

def cooldown_sleep(attempt: int):
    # 指数バックオフ + 上限 + ジッタ
    sec = min(TRENDS_COOLDOWN_BASE * (2 ** max(0, attempt-1)), TRENDS_COOLDOWN_MAX)
    sec = sec + random.uniform(0, TRENDS_JITTER)
    sec = max(sec, TRENDS_COOLDOWN_BASE)  # 念のため
    print(f"[TRENDS] 429 cooldown sleeping {sec:.1f}s ...", flush=True)
    time.sleep(sec)

def between_batches_sleep():
    sec = TRENDS_SLEEP + random.uniform(0, TRENDS_JITTER)
    print(f"[TRENDS] sleeping {sec:.1f}s between batches ...", flush=True)
    time.sleep(sec)

def fetch_batch(py: TrendReq, kw_list: List[str]):
    """
    1バッチを安全第一で取得。429 やネットエラーは数回までクールダウンして再試行。
    """
    last_err = None
    for attempt in range(1, TRENDS_BATCH_RETRIES+1):
        try:
            py.build_payload(kw_list=kw_list, timeframe=TRENDS_TIMEFRAME, geo=TRENDS_GEO)
            df = py.interest_over_time()
            if df is not None and not df.empty:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                return df
            # 空返りは軽い待機でリトライ
            print(f"[WARN] Trends returned empty frame for {kw_list} (attempt {attempt})", flush=True)
        except (RetryError, urllib3.exceptions.MaxRetryError) as e:
            last_err = e
            if is_429_error(e):
                cooldown_sleep(attempt)
            else:
                print(f"[WARN] Trends retry error (attempt {attempt}): {e}", flush=True)
                time.sleep(2 + attempt)
        except Exception as e:
            last_err = e
            if is_429_error(e):
                cooldown_sleep(attempt)
            else:
                print(f"[WARN] Trends fetch failed (attempt {attempt}): {e}", flush=True)
                time.sleep(2 + attempt)

    print(f"[ERROR] Trends batch failed after retries: {kw_list} :: {last_err}", flush=True)
    return None

def main():
    uni = load_universe()

    # TrendReq: ヘッダを明示、控えめな内部リトライ
    py = TrendReq(
        hl="en-US", tz=360, retries=1, backoff_factor=0.2,
        requests_args={
            "headers": {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            },
            "timeout": 30,
        },
    )

    results: Dict[str, Dict] = {}
    for batch in batched(uni.to_dict("records"), TRENDS_BATCH):
        kw_list = [r["query"] for r in batch]
        print(f"[TRENDS] fetching: {kw_list}", flush=True)
        df = fetch_batch(py, kw_list)
        if df is not None:
            for r in batch:
                q = r["query"]; sym = r["symbol"]
                series = df.get(q)
                score = compute_breakout_score(series) if series is not None else None
                results[sym] = {"query": q, "raw_breakout": score}
        else:
            # 取得失敗バッチ：エントリだけ作る（0 扱い）
            for r in batch:
                sym = r["symbol"]; q = r["query"]
                results[sym] = {"query": q, "raw_breakout": None}

        between_batches_sleep()

    # 0–1 正規化
    raws = [v.get("raw_breakout") for v in results.values() if v.get("raw_breakout") is not None]
    for sym, rec in results.items():
        rb = rec.get("raw_breakout")
        rec["score_0_1"] = pct_rank(raws, rb) if rb is not None else 0.0

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
