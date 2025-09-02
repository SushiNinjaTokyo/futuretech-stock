#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DII (Demand/Interest Index) 取得
- DII_SOURCE=finra_api|fast_volume
  * finra_api: FINRA Reg SHO / Short Volume 系API（失敗時はfast_volumeに自動フォールバック）
  * fast_volume: yfinanceの出来高から短期間の"関心"を代理指標化（高速・安定）
- どちらでも「0..1の score_0_1」と簡単な components を返す。
- 出力: site/data/dii/latest.json, site/data/<date>/dii.json
"""

import os, json, time, math, random
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import requests

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "data/universe.csv").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()

DII_SOURCE = os.environ.get("DII_SOURCE", "finra_api").strip()
LOOKBACK_WEEKS = int(float(os.environ.get("DII_LOOKBACK_WEEKS", "12")))
RECENT_WEEKS   = int(float(os.environ.get("DII_RECENT_WEEKS", "4")))

DATA_ROOT = os.path.join(OUT_DIR, "data")
DATE_DIR  = os.path.join(DATA_ROOT, REPORT_DATE or "today")
OUT_LATEST = os.path.join(DATA_ROOT, "dii", "latest.json")
OUT_DATE   = os.path.join(DATE_DIR, "dii.json")

def ensure_dirs():
    os.makedirs(os.path.dirname(OUT_LATEST), exist_ok=True)
    os.makedirs(DATE_DIR, exist_ok=True)

def load_universe(path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "symbol" not in cols:
        if "ticker" in cols:
            df = df.rename(columns={"ticker":"symbol"})
        else:
            raise RuntimeError("universe.csv requires 'symbol'")
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    recs = df[["symbol","name"]].copy()
    recs["symbol"] = recs["symbol"].astype(str).str.upper().str.strip()
    recs["name"]   = recs["name"].astype(str).str.strip()
    return recs.to_dict(orient="records")

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

# ------------------------
# fast_volume モード
# ------------------------
def dii_from_volume(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    週次近似のボリューム変化から DII を代理で算出。
    - 直近 RECENT_WEEKS とその前 LOOKBACK_WEEKS-RECENT_WEEKS を比較し、比率→0..1
    """
    end_date = REPORT_DATE or datetime.utcnow().strftime("%Y-%m-%d")
    hist = yf.download(
        tickers=" ".join(symbols),
        end=end_date,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )

    items: List[Dict[str, Any]] = []

    def vol_series(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty or "Volume" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Volume"].dropna()
        return s

    for sym in symbols:
        try:
            sub = hist[sym] if isinstance(hist.columns, pd.MultiIndex) and sym in hist.columns.get_level_values(0) else hist
            s = vol_series(sub)
            if s.empty:
                items.append({"symbol": sym, "score_0_1": 0.0, "components": {"volume":0.0}})
                continue
            # 週ごと合計（営業日×約5で集約）
            w = s.resample("W").sum()
            if len(w) < max(4, RECENT_WEEKS+4):
                items.append({"symbol": sym, "score_0_1": 0.0, "components": {"volume":0.0}})
                continue
            recent = float(w.tail(RECENT_WEEKS).mean())
            past   = float(w.iloc[-(LOOKBACK_WEEKS+RECENT_WEEKS): -RECENT_WEEKS].mean()) if len(w) >= (LOOKBACK_WEEKS+RECENT_WEEKS) else float(w.iloc[:-RECENT_WEEKS].mean())
            ratio = 0.0 if past <= 0 else (recent / past)
            score = clamp01((ratio - 1.0) / 1.0)  # 2倍↑で1.0に近づくよう緩やかに
            items.append({
                "symbol": sym,
                "score_0_1": score,
                "components": {"volume": score}
            })
        except Exception:
            items.append({"symbol": sym, "score_0_1": 0.0, "components": {"volume":0.0}})

    return items

# ------------------------
# finra_api モード（失敗しても即フォールバック）
# ------------------------
def dii_from_finra(symbols: List[str]) -> Tuple[List[Dict[str, Any]], bool]:
    """
    FINRA Reg SHO / Short Volume 相当データ（擬似実装）
    - 実環境での不確実性があるため、例外時は fast_volume にフォールバック。
    - ここでは『疑似スコア』: APIレスポンスが得られれば件数ベースで正規化。
    """
    try:
        # 例: 直近12週の銘柄別集計を一括取得（※実API仕様により調整が必要）
        # 実サービス側の仕様差異や429に備え、リクエストは極小回数＋大きめtimeout
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; futuretech-stock/1.0)",
            "Accept": "application/json",
        })
        # ※ ダミーのURL（実エンドポイントに合わせて調整してください）
        # ここが失敗したら即フォールバック
        url = "https://api.finra.org/data/group/otcMarket/regShoDaily"
        params = {
            "limit": 50000,
            # "filter": "...",  # 実APIのフィルター構文にあわせる
        }
        r = session.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return ([], False)
        data = r.json() if r.content else []
        # 簡易集計: シンボルごとの件数を数え、スコア化
        cnt_by_sym: Dict[str, int] = {s: 0 for s in symbols}
        for row in (data or []):
            s = str(row.get("symbol", "")).upper().strip()
            if s in cnt_by_sym:
                cnt_by_sym[s] += 1
        maxcnt = max(1, max(cnt_by_sym.values()) if cnt_by_sym else 1)
        items = [{"symbol": s, "score_0_1": clamp01(cnt / maxcnt), "components": {"finra_hits": clamp01(cnt / maxcnt)}} for s, cnt in cnt_by_sym.items()]
        return (items, True)
    except Exception:
        return ([], False)

def main():
    ensure_dirs()
    uni = load_universe(UNIVERSE_CSV)
    symbols = [x["symbol"] for x in uni]

    items: List[Dict[str, Any]] = []
    used = "fast_volume"

    if DII_SOURCE.lower() == "finra_api":
        recs, ok = dii_from_finra(symbols)
        if ok and recs:
            items = recs
            used = "finra_api"
        else:
            # フォールバック
            items = dii_from_volume(symbols)
            used = "fast_volume"
    else:
        items = dii_from_volume(symbols)
        used = "fast_volume"

    payload = {"date": REPORT_DATE, "items": items, "meta": {"source_used": used, "lookback_weeks": LOOKBACK_WEEKS, "recent_weeks": RECENT_WEEKS}}

    with open(OUT_LATEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_DATE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[DII] saved: {OUT_LATEST} and {OUT_DATE} (symbols={len(items)}) source={used}")

if __name__ == "__main__":
    main()
