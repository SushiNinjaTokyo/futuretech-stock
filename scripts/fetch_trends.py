#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import math
from typing import Dict, List, Any

import pandas as pd
from pytrends.request import TrendReq

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "data/universe.csv").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()

# 環境変数で調整可能（保守的な既定値）
TRENDS_GEO = os.environ.get("TRENDS_GEO", "US")
TRENDS_TIMEFRAME = os.environ.get("TRENDS_TIMEFRAME", "today 3-m")  # 90日
TRENDS_BATCH = int(os.environ.get("TRENDS_BATCH", "3"))
TRENDS_SLEEP = float(os.environ.get("TRENDS_SLEEP", "4.0"))
TRENDS_JITTER = float(os.environ.get("TRENDS_JITTER", "1.5"))
TRENDS_COOLDOWN_BASE = int(os.environ.get("TRENDS_COOLDOWN_BASE", "45"))
TRENDS_COOLDOWN_MAX = int(os.environ.get("TRENDS_COOLDOWN_MAX", "180"))
TRENDS_BATCH_RETRIES = int(os.environ.get("TRENDS_BATCH_RETRIES", "4"))

DATA_ROOT = os.path.join(OUT_DIR, "data")
DATE_DIR = os.path.join(DATA_ROOT, REPORT_DATE) if REPORT_DATE else os.path.join(DATA_ROOT, "today")
LATEST_PATH = os.path.join(DATA_ROOT, "trends", "latest.json")
BYDATE_PATH = os.path.join(DATA_ROOT, REPORT_DATE, "trends.json") if REPORT_DATE else None

def ensure_dirs():
    os.makedirs(os.path.join(DATA_ROOT, "trends"), exist_ok=True)
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

def sleep_brief(base=TRENDS_SLEEP, jitter=TRENDS_JITTER):
    time.sleep(base + random.random() * jitter)

def robust_z_to_01(z: float, clamp=3.0) -> float:
    z = max(-clamp, min(clamp, float(z)))
    return 0.5 + z / (2 * clamp)  # -clamp→0.0, 0→0.5, +clamp→1.0

def score_breakout_0_1(series: pd.Series) -> float:
    """直近7日平均 vs 直近90日（フレーム全体）平均、robust z を0..1に"""
    s = series.dropna().astype(float)
    if s.empty:
        return 0.0
    # Google Trends は 0..100 正規化値だが曜日効果が強いので7日移動平均
    s_ma = s.rolling(7, min_periods=1).mean()
    recent = s_ma.tail(7).mean()
    base = s_ma.mean()
    if pd.isna(recent) or pd.isna(base) or base == 0:
        return 0.0
    # MAD ベースのロバストz
    med = s_ma.median()
    mad = (s_ma - med).abs().median()
    if mad <= 0:
        # 分散が出ない場合は単純偏差
        z = (recent - base) / (max(1.0, s_ma.std(ddof=0)))
    else:
        z = (recent - med) / (1.4826 * mad)
    # 上側だけ見たいので半波整流
    z = max(0.0, z)
    return robust_z_to_01(z)  # 0..1

def build_query_term(name: str) -> str:
    """社名が長い/曖昧でもそこそこ当たる程度の term を作る"""
    # 例: "NVIDIA Corporation" → "NVIDIA"
    term = name.strip()
    # 括弧・, Inc. などを落とす
    term = term.replace("Inc.", "").replace("Inc", "").replace("Corporation", "").replace("Corp.", "").replace("Corp", "")
    term = term.replace("Ltd.", "").replace("Ltd", "").replace("Co.", "").replace("Co", "")
    term = term.split("(")[0].strip()
    # 単語先頭を利用（2語まで）
    words = [w for w in term.split() if w]
    if not words:
        return name
    if len(words) >= 2:
        return " ".join(words[:2])
    return words[0]

def fetch_batch(pytrends: TrendReq, batch: List[Dict[str, str]]) -> Dict[str, Any]:
    """batch = [{symbol, name}] → symbol -> record"""
    out: Dict[str, Any] = {}
    kw_list = [build_query_term(it["name"]) for it in batch]
    # pytrends build_payload は20語までが安定（ここはバッチで3にしてある）
    pytrends.build_payload(kw_list=kw_list, timeframe=TRENDS_TIMEFRAME, geo=TRENDS_GEO)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return out
    # df: index=date, columns=kw_list (+isPartial)
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    for idx, it in enumerate(batch):
        sym = it["symbol"]
        col = kw_list[idx]
        if col in df.columns:
            s = df[col]
            score01 = score_breakout_0_1(s)
            out[sym] = {
                "symbol": sym,
                "name": it["name"],
                "term": col,
                "timeframe": TRENDS_TIMEFRAME,
                "geo": TRENDS_GEO,
                "score_0_1": round(float(score01), 4),
                "recent_mean": float(s.tail(7).mean()) if not s.tail(7).empty else 0.0,
                "window_mean": float(s.mean()) if not s.empty else 0.0,
                "points_0_1000": int(round(score01 * 1000)),
            }
    return out

def main():
    ensure_dirs()
    uni = load_universe()
    records = uni.to_dict("records")

    # pytrends セッション
    pytrends = TrendReq(hl="en-US", tz=360)

    items: Dict[str, Any] = {}
    for i in range(0, len(records), TRENDS_BATCH):
        chunk = records[i:i+TRENDS_BATCH]
        # リトライ（指数バックオフ）
        ok = False
        wait = TRENDS_COOLDOWN_BASE
        for r in range(TRENDS_BATCH_RETRIES):
            try:
                part = fetch_batch(pytrends, chunk)
                # 1つでも取れていれば成功扱い
                if part:
                    items.update(part)
                ok = True
                break
            except Exception as e:
                # 429/5xx 等を想定
                time.sleep(min(wait, TRENDS_COOLDOWN_MAX))
                wait = min(TRENDS_COOLDOWN_MAX, wait * 2)
        # バッチ間クールダウン
        sleep_brief()
        if not ok:
            # 失敗した場合でも items は更新済み(0件の可能性あり)。続行。
            pass

    payload = {"items": items, "count": len(items), "date": REPORT_DATE}
    # 保存
    os.makedirs(os.path.dirname(LATEST_PATH), exist_ok=True)
    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if BYDATE_PATH:
        with open(BYDATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[TRENDS] saved: {LATEST_PATH} and {BYDATE_PATH or '(no-date)'} (symbols={len(items)})")

if __name__ == "__main__":
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is empty.")
    main()
