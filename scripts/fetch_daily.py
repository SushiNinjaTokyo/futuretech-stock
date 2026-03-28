#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, json, math, time, random
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None


# ======================
# logging
# ======================
def log(level, msg):
    print(f"{datetime.utcnow().isoformat()}Z [{level}] {msg}", flush=True)


# ======================
# env
# ======================
def env(name, default):
    return os.environ.get(name, default)


# ======================
# 再実行防止
# ======================
def already_done(out_dir, report_date):
    path = os.path.join(out_dir, "data", report_date, "top10.json")
    if os.path.exists(path):
        log("INFO", f"Already exists → skip: {path}")
        return True
    return False


# ======================
# データ取得（リトライ付き）
# ======================
def fetch_history(symbol, provider, token=None):
    for attempt in range(3):
        try:
            if provider == "tiingo" and token and pdr:
                end = pd.Timestamp.utcnow()
                start = end - pd.DateOffset(months=12)
                df = pdr.get_data_tiingo(symbol, api_key=token, start=start, end=end)

                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)

                df["Close"] = df["close"]
                df["Volume"] = df["volume"]

            else:
                if yf is None:
                    return None
                df = yf.download(
                    symbol,
                    period="12mo",
                    interval="1d",
                    progress=False,
                    threads=False,
                    auto_adjust=False  # ★ FutureWarning対策
                )

            if df is None or len(df) < 30:
                raise Exception("empty df")

            df = df[["Close", "Volume"]].dropna()
            return df

        except Exception as e:
            log("WARN", f"{symbol} fetch fail {attempt+1}: {e}")
            time.sleep(1.5 + random.random())

    return None


# ======================
# 指標計算（型安全）
# ======================
def pct(series, lag):
    try:
        cur = float(series.iloc[-1])
        prev = float(series.iloc[-1 - lag])
        return (cur / prev - 1) * 100
    except Exception:
        return None


def vol_score(df):
    try:
        vol = df["Volume"]
        rvol = float(vol.iloc[-1]) / (float(vol.tail(21).mean()) + 1e-9)
        return float(min(1.0, math.log1p(rvol) / 2.0))
    except Exception:
        return 0.0


# ======================
# JSON安全化
# ======================
def sanitize(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


# ======================
# main
# ======================
def run():

    OUT_DIR = env("OUT_DIR", "site")
    REPORT_DATE = env("REPORT_DATE", datetime.utcnow().strftime("%Y-%m-%d"))
    PROVIDER = env("DATA_PROVIDER", "yfinance")
    TOKEN = env("TIINGO_TOKEN", None)

    if already_done(OUT_DIR, REPORT_DATE):
        return

    # universe
    universe = pd.read_csv(env("UNIVERSE_CSV", "data/universe.csv"))
    symbols = universe["symbol"].dropna().astype(str).str.upper().tolist()

    rows = []
    fail = []

    for sym in symbols:
        log("INFO", f"Processing {sym}")

        df = fetch_history(sym, PROVIDER, TOKEN)

        if df is None:
            fail.append(sym)
            continue

        try:
            row = {
                "symbol": sym,
                "score": float(vol_score(df)),
                "d1": pct(df["Close"], 1),
                "d5": pct(df["Close"], 5),
                "d20": pct(df["Close"], 20),
            }
            rows.append(row)

        except Exception as e:
            log("WARN", f"{sym} calc error: {e}")
            fail.append(sym)

    if not rows:
        log("ERROR", "no valid data")
        sys.exit(1)

    # ranking
    rows.sort(key=lambda x: x["score"], reverse=True)
    top10 = rows[:10]

    # 出力
    outdir = os.path.join(OUT_DIR, "data", REPORT_DATE)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "top10.json"), "w") as f:
        json.dump(top10, f, indent=2, default=sanitize)

    log("INFO", f"done: {len(top10)} symbols / fail={len(fail)}")


if __name__ == "__main__":
    run()
