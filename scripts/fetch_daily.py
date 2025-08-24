#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, pathlib, random, time
import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# US 市場日付（引け基準）
# =========================
def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    # 引け後18:00未満なら、まだ当日速報は不完全 → 前営業日にする
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    # 週末処理（土=5, 日=6 → 直近の金曜へ）
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d


# =========================
# 環境変数
# =========================
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()  # yfinance | tiingo
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")  # tiingo時のみ
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"  # 実データは false

# 予算管理
BUDGET_JPY_MAX = float(os.getenv("BUDGET_JPY_MAX", "10000"))
SPEND_FILE = os.getenv("SPEND_FILE", f"{OUT_DIR}/data/spend.json")
MANUAL_DAILY_COST_JPY = float(os.getenv("MANUAL_DAILY_COST_JPY", "0"))
# 固定費（yfinance=0, tiingo=1600円想定）
PROVIDER_FIXED_JPY = 0.0 if DATA_PROVIDER == "yfinance" else float(os.getenv("TIINGO_MONTHLY_JPY", "1600"))


# =========================
# 予算ユーティリティ
# =========================
def month_key(date_iso: str):
    d = datetime.date.fromisoformat(date_iso)
    return f"{d.year}-{d.month:02d}"

def load_spend():
    p = pathlib.Path(SPEND_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def save_spend(data):
    p = pathlib.Path(SPEND_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))

def month_spend_total(spend, mkey):
    return float(spend.get(mkey, {}).get("total_jpy", 0))

def add_spend(spend, mkey, date_iso, amount_jpy, memo):
    month = spend.setdefault(mkey, {"items": [], "total_jpy": 0})
    month["items"].append({"date": date_iso, "amount_jpy": amount_jpy, "memo": memo})
    month["total_jpy"] = round(float(month["total_jpy"]) + amount_jpy, 2)

def budget_check():
    spend = load_spend()
    mkey = month_key(DATE)
    month_used = month_spend_total(spend, mkey)
    today_cost = MANUAL_DAILY_COST_JPY
    if month_used + today_cost > BUDGET_JPY_MAX:
        print(f"[BUDGET] Cap reached ({month_used:.0f} + {today_cost:.0f} > {BUDGET_JPY_MAX:.0f}). Skipping run.")
        return False, spend
    return True, spend

def mark_fixed_costs(spend):
    # yfinanceは固定費0円。tiingo等に切替時のみ月1回計上
    mkey = month_key(DATE)
    month = spend.setdefault(mkey, {"items": [], "total_jpy": 0})
    if PROVIDER_FIXED_JPY > 0 and not month.get("provider_month_mark"):
        add_spend(spend, mkey, DATE, PROVIDER_FIXED_JPY, f"{DATA_PROVIDER} monthly flat")
        month["provider_month_mark"] = True
    if MANUAL_DAILY_COST_JPY > 0:
        add_spend(spend, mkey, DATE, MANUAL_DAILY_COST_JPY, "Variable API usage (manual)")
    save_spend(spend)


# =========================
# Data Providers
# =========================
def tiingo_eod_range(symbol, start, end):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"token": TIINGO_TOKEN, "startDate": start, "endDate": end, "resampleFreq": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    # Tiingo: 'adjClose' があれば close に寄せる
    if "adjClose" in df.columns:
        df = df.rename(columns={"adjClose": "close"})
    # 列正規化
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({
        "date":   pd.to_datetime(df[cols.get("date", "date")]).dt.tz_localize(None).dt.strftime("%Y-%m-%d"),
        "open":   pd.to_numeric(df.get("open", pd.Series(dtype="float64")), errors="coerce"),
        "high":   pd.to_numeric(df.get("high", pd.Series(dtype="float64")), errors="coerce"),
        "low":    pd.to_numeric(df.get("low",  pd.Series(dtype="float64")), errors="coerce"),
        "close":  pd.to_numeric(df.get("close", pd.Series(dtype="float64")), errors="coerce"),
        "volume": pd.to_numeric(df.get("volume", pd.Series(dtype="float64")), errors="coerce").fillna(0),
    })
    return out[["date", "open", "high", "low", "close", "volume"]]


def yfi_eod_range(symbol, start, end):
    import yfinance as yf

    # タイムゾーンずれ対策で1〜2日バッファ
    start_dt = datetime.date.fromisoformat(start) - datetime.timedelta(days=2)
    end_dt   = datetime.date.fromisoformat(end)   + datetime.timedelta(days=1)

    def _normalize(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame()
        df2 = df_in.reset_index()
        df2.columns = [str(c).strip().lower() for c in df2.columns]
        cands = {
            "date":   ["date", "datetime", "index"],
            "open":   ["open"],
            "high":   ["high"],
            "low":    ["low"],
            "close":  ["close", "adj close", "adjclose"],
            "volume": ["volume"],
        }
        out = {}
        for k, vs in cands.items():
            for v in vs:
                if v in df2.columns:
                    out[k] = df2[v]
                    break
        if "date" not in out or "close" not in out:
            return pd.DataFrame()
        vol = out.get("volume", pd.Series([0]*len(out["close"])))
        res = pd.DataFrame({
            "date":   pd.to_datetime(out["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d"),
            "open":   pd.to_numeric(out.get("open",   pd.Series(dtype="float64")), errors="coerce"),
            "high":   pd.to_numeric(out.get("high",   pd.Series(dtype="float64")), errors="coerce"),
            "low":    pd.to_numeric(out.get("low",    pd.Series(dtype="float64")), errors="coerce"),
            "close":  pd.to_numeric(out["close"], errors="coerce"),
            "volume": pd.to_numeric(vol, errors="coerce").fillna(0),
        })
        return res[["date", "open", "high", "low", "close", "volume"]]

    # 1) 通常: start/end で取得（最大2回リトライ）
    for attempt in range(2):
        df = yf.download(
            symbol,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,   # CI安定化
        )
        df = _normalize(df)
        if not df.empty:
            return df
        time.sleep(0.8 * (attempt + 1))

    # 2) フォールバック: history(period=6mo)
    df = _normalize(yf.Ticker(symbol).history(period="6mo", interval="1d", auto_adjust=True))
    if not df.empty:
        return df

    # 3) 最終フォールバック: download(period=6mo)
    df = _normalize(yf.download(symbol, period="6mo", interval="1d", auto_adjust=True, progress=False, threads=False))
    return df


def get_eod_range(symbol, start, end):
    if MOCK_MODE:
        # ダミーデータ（営業日）
        dates = pd.date_range(start=start, end=end, freq="B")
        base = 100.0 + random.Random(symbol).random() * 20
        rows = []
        for d in dates:
            base *= (1.0 + random.uniform(-0.02, 0.02))
            vol = random.randint(100000, 5000000)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "open": base * 0.99, "high": base * 1.01, "low": base * 0.98,
                "close": base, "volume": vol
            })
        return pd.DataFrame(rows)
    if DATA_PROVIDER == "tiingo":
        if not TIINGO_TOKEN:
            raise RuntimeError("TIINGO_TOKEN is required for tiingo provider")
        return tiingo_eod_range(symbol, start, end)
    # default: yfinance
    return yfi_eod_range(symbol, start, end)


# =========================
# Metrics
# =========================
def compute_metrics(df: pd.DataFrame):
    """従来の最小メトリクス（前日比・出来高比）※互換用"""
    if df is None or df.empty:
        return None

    # 必須列チェック
    for col in ("close", "volume"):
        if col not in df.columns:
            return None

    d = df.copy()
    d["close"]  = pd.to_numeric(d["close"],  errors="coerce").ffill()
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)

    if len(d) < 2:
        return 0.0, 1.0

    today_close = d["close"].iloc[-1]
    prev_close  = d["close"].iloc[-2]
    if pd.isna(today_close) or pd.isna(prev_close) or float(prev_close) == 0.0:
        pct_change = 0.0
    else:
        pct_change = (float(today_close) - float(prev_close)) / float(prev_close)

    vol_sma20 = d["volume"].rolling(20).mean().iloc[-1]
    today_vol = d["volume"].iloc[-1]
    if pd.isna(vol_sma20) or float(vol_sma20) == 0.0:
        vol_ratio = 1.0
    else:
        vol_ratio = float(today_vol) / float(vol_sma20)

    return float(pct_change), float(vol_ratio)


def compute_volume_anomaly_light(df, close_col="close", vol_col="volume"):
    """
    120日以内の手持ちデータだけで「異常出来高スコア」を算出（軽量版）。
    - RVOL20（出来高/20日平均）
    - z60（60日平均・標準偏差に対する当日出来高のZ値）
    - PctRank90（直近最大90日に対する百分位、データ不足時は未使用）
    正規化:
      z_norm   = clip(0.5 + 0.1*z60, 0, 1)
      rvol_norm= clip(RVOL20/3, 0, 1)
      pct_norm = PctRank90
    合成:
      十分な履歴あり: 0.5*z + 0.3*rvol + 0.2*pct
      不足時         : 0.65*z + 0.35*rvol
    低流動フィルタ（緩め）:
      volume >= 200k かつ DollarVol >= 5M
    """
    import numpy as np
    if df is None or df.empty or close_col not in df.columns or vol_col not in df.columns:
        return None

    d = df.copy()
    d[vol_col] = pd.to_numeric(d[vol_col], errors="coerce").fillna(0)
    d[close_col] = pd.to_numeric(d[close_col], errors="coerce").ffill()
    d = d.tail(120)  # 手持ちのうち最大120本

    if len(d) < 25:  # 最低限
        return None

    v = d[vol_col].values
    vt = float(v[-1])

    # RVOL20
    sma20 = pd.Series(v).rolling(20).mean().iloc[-1]
    rvol20 = float(vt / sma20) if pd.notna(sma20) and sma20 > 0 else 0.0

    # z60（当日を除いた直近win本で統計）
    win = min(60, len(v) - 1)
    if win >= 20:
        base = v[-(win + 1):-1]
        mu = float(np.mean(base))
        sd = float(np.std(base, ddof=0))
        z60 = float((vt - mu) / sd) if sd > 0 else 0.0
    else:
        z60 = 0.0

    # 百分位（最大90日）
    pr_win = min(90, len(v) - 1)
    if pr_win >= 20:
        hist = v[-pr_win:]
        pct_rank_90 = float((hist <= vt).sum() / len(hist))
    else:
        pct_rank_90 = None

    # 正規化
    z_norm = max(0.0, min(1.0, 0.5 + 0.1 * z60))
    rvol_norm = max(0.0, min(1.0, rvol20 / 3.0))
    if pct_rank_90 is None:
        score = 0.65 * z_norm + 0.35 * rvol_norm
    else:
        score = 0.5 * z_norm + 0.3 * rvol_norm + 0.2 * pct_rank_90

    # 低流動フィルタ（テスト段階は緩め）
    dollar_vol = vt * float(d[close_col].iloc[-1])
    eligible = (vt >= 200_000) and (dollar_vol >= 5_000_000)

    return {
        "rvol20": rvol20,
        "z60": z60,
        "pct_rank_90": pct_rank_90,
        "z_norm": z_norm,
        "rvol_norm": rvol_norm,
        "score": float(score),
        "dollar_vol": float(dollar_vol),
        "eligible": bool(eligible),
        "today_volume": float(vt),
        "sma20_volume": float(sma20) if pd.notna(sma20) else None,
        "close": float(d[close_col].iloc[-1]),
    }


# =========================
# Charts
# =========================
def save_chart_png_weekly_3m(symbol: str, df_daily: pd.DataFrame, out_dir: str, date_iso: str):
    """直近3ヶ月相当を週足化してPNG出力"""
    if df_daily is None or df_daily.empty:
        print(f"[WARN] weekly chart skipped (empty) {symbol}", file=sys.stderr)
        return

    d = df_daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    # 直近120日で切り出し（営業日≒3〜4ヶ月）
    cutoff = d["date"].max() - pd.Timedelta(days=120)
    d = d[d["date"] >= cutoff]
    if d.empty:
        print(f"[WARN] weekly window empty {symbol}", file=sys.stderr)
        return

    # 週足OHLC（週末を金曜に揃える）
    d = d.set_index("date")
    w = pd.DataFrame({
        "open":   d["open"].resample("W-FRI").first(),
        "high":   d["high"].resample("W-FRI").max(),
        "low":    d["low"].resample("W-FRI").min(),
        "close":  d["close"].resample("W-FRI").last(),
        "volume": d["volume"].resample("W-FRI").sum(),
    }).dropna(how="any")

    if w.empty:
        print(f"[WARN] weekly resample empty {symbol}", file=sys.stderr)
        return

    charts_dir = pathlib.Path(out_dir) / "charts" / date_iso
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.8), dpi=120)
    plt.plot(w.index, w["close"], linewidth=1.4)
    plt.title(f"{symbol} — 3M Weekly")
    plt.tight_layout()
    plt.savefig(charts_dir / f"{symbol}.png")
    plt.close()


# （残す：日足3年チャート）
def save_chart_png(symbol, df, out_dir, date_iso):
    if df is None or df.empty:
        return
    charts_dir = pathlib.Path(out_dir) / "charts" / date_iso
    charts_dir.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["ma20"]  = d["close"].rolling(20).mean()
    d["ma50"]  = d["close"].rolling(50).mean()
    d["ma200"] = d["close"].rolling(200).mean()

    plt.figure(figsize=(9, 4.8), dpi=120)
    plt.plot(d["date"], d["close"], linewidth=1.2)
    plt.plot(d["date"], d["ma20"],  linewidth=0.9)
    plt.plot(d["date"], d["ma50"],  linewidth=0.9)
    plt.plot(d["date"], d["ma200"], linewidth=0.9)
    plt.title(f"{symbol} — 3Y Daily")
    plt.tight_layout()
    out = charts_dir / f"{symbol}.png"
    plt.savefig(out)
    plt.close()


# =========================
# メイン
# =========================
def main():
    ok, spend = budget_check()
    if not ok:
        return

    uni = pd.read_csv(UNIVERSE_CSV)
    rows = []
    top10 = []

    end = DATE
    # ★ 取得期間を 90 → 120 日に拡張（API増やさず、軽量スコアが成立）
    start_short = (datetime.date.fromisoformat(DATE) - datetime.timedelta(days=120)).isoformat()

    recent_map = {}

    # ----- データ取得 → 指標算出 -----
    for _, t in uni.iterrows():
        symbol = str(t["symbol"]).strip()
        try:
            df = get_eod_range(symbol, start_short, end)

            # 列名を小文字化＆index→dateの保険
            if df is not None and not df.empty:
                df = df.rename(columns=lambda c: str(c).strip().lower())
                if "date" not in df.columns and hasattr(df.index, "dtype"):
                    try:
                        df = df.reset_index().rename(columns={"index": "date"})
                    except Exception:
                        pass

            recent_map[symbol] = df

            # 既存の最小メトリクス（互換）
            metrics = compute_metrics(df)
            if not metrics:
                print(f"[WARN] skip (no metrics) {symbol}", file=sys.stderr)
                continue
            pct_change, vol_ratio = metrics

            # ★ 異常出来高スコア（軽量版）
            anom = compute_volume_anomaly_light(df)
            if not anom:
                # データ不足など
                vol_anom_score = 0.0
                vol_anom_eligible = False
            else:
                vol_anom_score = anom["score"]
                vol_anom_eligible = anom["eligible"]

            rows.append({
                "symbol": symbol,
                "name":   t.get("name", ""),
                "theme":  t.get("theme", ""),
                "pct_change": pct_change,
                "vol_ratio":  vol_ratio,
                "news_count": 0,  # 将来マージ
                "vol_anomaly_score": vol_anom_score,
                "eligible_liquidity": vol_anom_eligible,
                "tech_note": "Auto tech note TBD",   # 使わない方針でもフィールドは残しておく
                "ir_note":   "IR/News summary TBD",
            })
        except Exception as e:
            print(f"[WARN] {symbol}: {e}", file=sys.stderr)
            continue

    # ----- rowsが空なら、空のtop10を書いて正常終了 -----
    out_json_dir = pathlib.Path(OUT_DIR) / "data" / DATE
    out_json_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(out_json_dir / "top10.json", "w") as f:
            json.dump(top10, f, indent=2)
        mark_fixed_costs(spend)
        print(f"Generated top10 for {DATE}: 0 symbols (no rows)")
        return

    # ----- ランキング（異常出来高スコアを主軸） -----
    # 1) 低流動は原則除外
    eligible_rows = [r for r in rows if r.get("eligible_liquidity", False)]
    target = eligible_rows if eligible_rows else rows  # すべて不可の場合は全体でフォールバック

    # 2) スコア = vol_anomaly_score（既に0〜1相当）
    for r in target:
        base = float(r.get("vol_anomaly_score", 0.0))
        # 念のためクリップ
        r["score"] = max(0.0, min(1.0, base))

    # 3) ソート＆Top10
    target.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top10 = target[:10]

    # ----- JSON出力 -----
    with open(out_json_dir / "top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    # ----- 週足3ヶ月チャート生成（MOCK時はスキップ） -----
    if not MOCK_MODE and top10:
        for r in top10:
            try:
                hist = recent_map.get(r["symbol"])
                if hist is None or hist.empty:
                    print(f"[WARN] no data for weekly chart {r['symbol']}", file=sys.stderr)
                    continue
                save_chart_png_weekly_3m(r["symbol"], hist, OUT_DIR, DATE)
            except Exception as e:
                print(f"[WARN] chart {r['symbol']}: {e}", file=sys.stderr)

    mark_fixed_costs(spend)
    print(f"Generated top10 for {DATE}: {len(top10)} symbols")


if __name__ == "__main__":
    main()
