#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import random
from typing import Dict, Any, List

import requests
import pandas as pd

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "data/universe.csv").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()

DII_SOURCE = os.environ.get("DII_SOURCE", "finra_api").strip()

# レート制限/不安定性に備えた待機
REQ_SLEEP_BASE = 0.6
REQ_SLEEP_JITTER = 0.4
RETRIES = 5
BACKOFF_BASE = 2.0
BACKOFF_MAX = 120

DATA_ROOT = os.path.join(OUT_DIR, "data")
DATE_DIR = os.path.join(DATA_ROOT, REPORT_DATE) if REPORT_DATE else os.path.join(DATA_ROOT, "today")
LATEST_PATH = os.path.join(DATA_ROOT, "dii", "latest.json")
BYDATE_PATH = os.path.join(DATA_DIR := os.path.join(DATA_ROOT, REPORT_DATE), "dii.json") if REPORT_DATE else None

FINRA_ATS_URL = "https://api.finra.org/data/ats/weeklySummary"  # FINRA ATS weekly summary

def ensure_dirs():
    os.makedirs(os.path.join(DATA_ROOT, "dii"), exist_ok=True)
    if REPORT_DATE:
        os.makedirs(os.path.join(DATA_ROOT, REPORT_DATE), exist_ok=True)

def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        else:
            raise SystemExit("universe.csv に symbol / ticker 列が必要です")
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df[["symbol", "name"]].copy()

def sleep_brief():
    time.sleep(REQ_SLEEP_BASE + random.random() * REQ_SLEEP_JITTER)

def z01_pos(z: float, clamp=3.0) -> float:
    z = max(-clamp, min(clamp, float(z)))
    return max(0.0, 0.5 + z / (2 * clamp))  # 下側は0.5未満に落ちるが最終でmax(0,.)に

def compute_score_from_weeks(week_rows: List[Dict[str, Any]]) -> float:
    """直近8週の 'atsTotalVolume' を利用。直近2週平均 vs 全8週の中央値でロバストz→0..1"""
    if not week_rows:
        return 0.0
    vols = []
    for w in week_rows:
        v = w.get("atsTotalVolume") or w.get("atsTotalVolumeInShares")
        try:
            vols.append(float(v))
        except Exception:
            pass
    if len(vols) < 3:
        return 0.0
    s = pd.Series(vols)
    recent = s.tail(2).mean()
    med = s.median()
    mad = (s - med).abs().median()
    if mad <= 0:
        std = s.std(ddof=0)
        if std == 0:
            return 0.0
        z = (recent - med) / std
    else:
        z = (recent - med) / (1.4826 * mad)
    # 買い圧/フロー上昇を上側重視
    return z01_pos(z)

def fetch_finra_ats(symbol: str) -> List[Dict[str, Any]]:
    """
    FINRA ATS weekly summary: issueSymbolTicker=symbol で過去分が返る
    例: https://api.finra.org/data/ats/weeklySummary?keys=issueSymbolTicker&issueSymbolTicker=NVDA
    """
    # ドキュメント上は keys=issueSymbolTicker を入れると結果が安定する
    params = {
        "keys": "issueSymbolTicker",
        "issueSymbolTicker": symbol.upper(),
        # ソートと件数（最大）: 最新優先で十分
        "sort": "-weekStartDate",
        "limit": 12,
    }
    headers = {"Accept": "application/json"}
    wait = 3
    for i in range(RETRIES):
        try:
            r = requests.get(FINRA_ATS_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                # たまに dict で返ることがあるので配列化
                if isinstance(data, dict):
                    # FINRA 形式によっては "data" キー配下にある場合も
                    arr = data.get("data")
                    if isinstance(arr, list):
                        return arr
                    return []
            # 429/5xx → バックオフ
            time.sleep(min(wait, BACKOFF_MAX))
            wait = min(BACKOFF_MAX, wait * BACKOFF_BASE)
        except Exception:
            time.sleep(min(wait, BACKOFF_MAX))
            wait = min(BACKOFF_MAX, wait * BACKOFF_BASE)
    return []

def main():
    if DII_SOURCE.lower() != "finra_api":
        # 他実装に切り替える余地を残す
        ensure_dirs()
        payload = {"items": {}, "count": 0, "date": REPORT_DATE, "source": DII_SOURCE}
        with open(LATEST_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if BYDATE_PATH:
            with open(BYDATE_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[DII] saved (noop source={DII_SOURCE}): {LATEST_PATH}")
        return

    ensure_dirs()
    uni = load_universe()
    items: Dict[str, Any] = {}

    for row in uni.itertuples(index=False):
        sym = str(row.symbol).upper()
        name = str(row.name)
        week_rows = fetch_finra_ats(sym)
        sleep_brief()

        score01 = compute_score_from_weeks(week_rows)
        rec = {
            "symbol": sym,
            "name": name,
            "recent_weeks": len(week_rows),
            "score_0_1": round(float(score01), 4),
            "points_0_1000": int(round(score01 * 1000)),
        }
        # オプションの可視化補助
        if week_rows:
            try:
                # 直近の週（weekStartDate は YYYY-MM-DD）
                last = week_rows[0]
                rec["last_week"] = last.get("weekStartDate")
                rec["atsTotalVolume"] = float(last.get("atsTotalVolume") or 0)
                rec["exchTotalVolume"] = float(last.get("totalWeeklyShareVolume") or 0)
            except Exception:
                pass

        items[sym] = rec

    payload = {"items": items, "count": len(items), "date": REPORT_DATE, "source": "finra_api"}
    # 保存
    os.makedirs(os.path.dirname(LATEST_PATH), exist_ok=True)
    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if BYDATE_PATH:
        with open(BYDATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[DII] saved: {LATEST_PATH} and {BYDATE_PATH or '(no-date)'} (symbols={len(items)})")

if __name__ == "__main__":
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is empty.")
    main()
