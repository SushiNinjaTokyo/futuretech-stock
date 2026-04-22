#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def env_s(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def env_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_b(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def sanitize(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=sanitize),
        encoding="utf-8",
    )


def load_universe(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        log("WARN", f"Universe CSV missing: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol", list(df.columns)[0])
    name_col = cols.get("name")

    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip().upper()
        if not sym:
            continue
        nm = str(row.get(name_col, "")).strip() if name_col else ""
        out.append({"symbol": sym, "name": nm})
    return out


def first_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")

    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)

        s = x.iloc[:, 0]
        if isinstance(s, pd.DataFrame):
            if s.shape[1] == 0:
                return pd.Series(dtype=float)
            s = s.iloc[:, 0]

        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce")

    return pd.Series(dtype=float)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Close", "Volume"])

    out = pd.DataFrame(index=pd.to_datetime(df.index))

    if isinstance(df.columns, pd.MultiIndex):
        close_col = None
        volume_col = None

        for col in df.columns:
            parts = [str(c).strip().lower() for c in col if c is not None]
            if "close" in parts and close_col is None:
                close_col = col
            if "volume" in parts and volume_col is None:
                volume_col = col

        if close_col is not None:
            out["Close"] = first_series(df.loc[:, close_col]).to_numpy()
        if volume_col is not None:
            out["Volume"] = first_series(df.loc[:, volume_col]).to_numpy()

    else:
        close_src = None
        volume_src = None

        for c in df.columns:
            cl = str(c).strip().lower()
            if cl == "close" and close_src is None:
                close_src = c
            if cl == "volume" and volume_src is None:
                volume_src = c

        if close_src is not None:
            out["Close"] = first_series(df[[close_src]] if isinstance(df[close_src], pd.DataFrame) else df[close_src]).to_numpy()
        if volume_src is not None:
            out["Volume"] = first_series(df[[volume_src]] if isinstance(df[volume_src], pd.DataFrame) else df[volume_src]).to_numpy()

    if "Close" not in out.columns:
        out["Close"] = np.nan
    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    return out[["Close", "Volume"]].dropna()


def pct(series: pd.Series, lag: int) -> Optional[float]:
    try:
        s = first_series(series).dropna()
        if len(s) <= lag:
            return None

        cur = float(s.iloc[-1])
        prev = float(s.iloc[-1 - lag])
        if prev == 0:
            return None

        return (cur / prev - 1.0) * 100.0
    except Exception:
        return None


def fetch_history(symbol: str, provider: str, token: Optional[str], months: int = 12) -> Optional[pd.DataFrame]:
    for attempt in range(3):
        try:
            if provider == "tiingo" and token and pdr is not None:
                end = pd.Timestamp.utcnow().normalize()
                start = end - pd.DateOffset(months=months)
                raw = pdr.get_data_tiingo(
                    symbol,
                    api_key=token,
                    start=start.date(),
                    end=end.date(),
                )

                if isinstance(raw.index, pd.MultiIndex):
                    raw = raw.reset_index(level=0, drop=True)

                if "close" in raw.columns:
                    raw["Close"] = raw["close"]
                if "volume" in raw.columns:
                    raw["Volume"] = raw["volume"]

            else:
                if yf is None:
                    return None

                raw = yf.download(
                    symbol,
                    period=f"{months}mo",
                    interval="1d",
                    progress=False,
                    threads=False,
                    auto_adjust=False,
                )

            if raw is None or len(raw) < 30:
                raise ValueError("empty dataframe")

            df = normalize_ohlcv(raw)
            if len(df) < 30:
                raise ValueError("not enough clean rows")

            return df.sort_index()

        except Exception as e:
            log("WARN", f"{symbol}: fetch_history attempt {attempt + 1} failed: {e}")
            time.sleep(1.5 + attempt * 0.8)

    return None

def fetch_weekly_history(symbol: str, provider: str, token: Optional[str]) -> Optional[pd.DataFrame]:
    """
    週足チャート描画用。
    20週移動平均をできるだけ自然に出したいので、最低6か月分を取得する。
    """
    try:
        if provider == "tiingo" and token and pdr is not None:
            df = fetch_history(symbol, provider, token, months=9)
            if df is None or df.empty:
                return None

            out = pd.DataFrame(index=df.index)
            out["Close"] = first_series(df["Close"]).to_numpy()
            return out.resample("W-FRI").last().dropna()

        if yf is None:
            return None

        raw = yf.download(
            symbol,
            period="9mo",
            interval="1wk",
            progress=False,
            threads=False,
            auto_adjust=False,
        )
        if raw is None or raw.empty:
            return None

        norm = normalize_ohlcv(raw)
        if norm.empty:
            return None

        out = pd.DataFrame(index=norm.index)
        out["Close"] = first_series(norm["Close"]).to_numpy()
        return out.dropna()

    except Exception as e:
        log("WARN", f"{symbol}: weekly history failed: {e}")
        return None

def price_direction_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    出来高異常に「上方向」を掛けるための方向性指標。
    意図:
    - 直近で上昇している
    - 3か月レンジの上側にいる
    - 20日線より上にいる
    を高く評価
    """
    close = first_series(df["Close"]).dropna()
    if len(close) < 30:
        return {
            "ret_5d": 0.0,
            "ret_20d": 0.0,
            "range_pos_63d": 0.0,
            "above_sma20": 0.0,
            "direction_score": 0.0,
        }

    ret_5d = 0.0 if len(close) <= 5 else float(close.iloc[-1] / close.iloc[-6] - 1.0)
    ret_20d = 0.0 if len(close) <= 20 else float(close.iloc[-1] / close.iloc[-21] - 1.0)

    lookback = close.tail(min(63, len(close)))
    cmin = float(lookback.min())
    cmax = float(lookback.max())
    if cmax > cmin:
        range_pos_63d = float((close.iloc[-1] - cmin) / (cmax - cmin))
    else:
        range_pos_63d = 0.5

    sma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
    above_sma20 = 1.0 if float(close.iloc[-1]) >= sma20 else 0.0

    # 上方向性スコア
    # ret5, ret20 はプラスで加点、マイナスは0
    s5 = max(0.0, min(1.0, ret_5d / 0.08))     # +8%で満点近辺
    s20 = max(0.0, min(1.0, ret_20d / 0.15))   # +15%で満点近辺
    srange = max(0.0, min(1.0, range_pos_63d))

    direction_score = 0.35 * s5 + 0.35 * s20 + 0.20 * srange + 0.10 * above_sma20
    direction_score = max(0.0, min(1.0, direction_score))

    return {
        "ret_5d": round(ret_5d * 100.0, 2),
        "ret_20d": round(ret_20d * 100.0, 2),
        "range_pos_63d": round(range_pos_63d, 3),
        "above_sma20": above_sma20,
        "direction_score": round(direction_score, 6),
    }


def compute_vol_anomaly(df: pd.DataFrame) -> Tuple[Dict[str, Any], float]:
    """
    従来の出来高異常 + 上方向性を加味したスコア。
    あなたの意図:
    「普段より出来高が急増して、上方向に向かっている銘柄」
    に合わせている。
    """
    if df is None or len(df) < 30:
        return {"eligible": False}, 0.0

    close = first_series(df["Close"]).dropna()
    vol = first_series(df["Volume"]).dropna()

    n = min(len(close), len(vol))
    if n < 30:
        return {"eligible": False}, 0.0

    close = close.iloc[-n:]
    vol = vol.iloc[-n:]
    dv = close * vol

    rvol20 = float(vol.iloc[-1] / (vol.tail(21).iloc[:-1].mean() + 1e-9))

    lv = np.log1p(vol)
    ref60 = lv.tail(61).iloc[:-1]
    mu = float(ref60.mean()) if len(ref60) else float(lv.mean())
    sd = float(ref60.std(ddof=0)) if len(ref60) else 0.0
    z60 = 0.0 if sd == 0 else float((lv.iloc[-1] - mu) / sd)

    last = dv.tail(91)
    ref = last.iloc[:-1]
    pct_rank_90 = float((ref <= last.iloc[-1]).mean()) if len(ref) else 0.0
    dollar_vol = float(dv.iloc[-1])

    eligible = bool(
        float(close.iloc[-1]) >= 2.0 and
        float(dv.tail(21).iloc[:-1].mean()) >= 1_000_000
    )

    # 既存の出来高異常
    s_rvol = 1.0 - math.exp(-max(0.0, rvol20) / 2.5)
    s_z = 1.0 / (1.0 + math.exp(-z60))
    s_pr = max(0.0, min(1.0, pct_rank_90))
    vol_only_score = max(0.0, min(1.0, 0.45 * s_pr + 0.40 * s_rvol + 0.15 * s_z))

    # 上方向性
    pdm = price_direction_metrics(pd.DataFrame({"Close": close, "Volume": vol}))
    direction_score = float(pdm["direction_score"])

    # 最終的に「出来高異常があり、かつ上方向」を評価
    # 出来高主体 70%、方向性 30%
    score = 0.70 * vol_only_score + 0.30 * direction_score
    score = max(0.0, min(1.0, score))

    detail = {
        "rvol20": round(rvol20, 3),
        "z60": round(z60, 3),
        "pct_rank_90": round(pct_rank_90, 3),
        "dollar_vol": round(dollar_vol, 2),
        "eligible": eligible,
        "ret_5d": pdm["ret_5d"],
        "ret_20d": pdm["ret_20d"],
        "range_pos_63d": pdm["range_pos_63d"],
        "above_sma20": pdm["above_sma20"],
        "direction_score": pdm["direction_score"],
        "vol_only_score": round(vol_only_score, 6),
    }
    return detail, round(score, 6)


def extract_component_map(base_dir: Path, report_date: str, kind: str) -> Dict[str, float]:
    candidates = [
        base_dir / "data" / report_date / f"{kind}.json",
        base_dir / "data" / kind / "latest.json",
    ]

    def extract_score(obj: Any) -> Optional[float]:
        if isinstance(obj, (int, float)):
            try:
                return max(0.0, min(1.0, float(obj)))
            except Exception:
                return None

        if isinstance(obj, dict):
            for key in ("score_0_1", "score01", "normalized", "score", "value"):
                if key in obj:
                    try:
                        return max(0.0, min(1.0, float(obj[key])))
                    except Exception:
                        return None
        return None

    out: Dict[str, float] = {}
    for path in candidates:
        j = read_json(path)
        if not j:
            continue

        payload = j.get("items", j) if isinstance(j, dict) else j
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                sym = str(row.get("symbol", "")).upper()
                if not sym:
                    continue
                s = extract_score(row)
                if s is not None:
                    out[sym] = s

        if out:
            break

    return out


def normalize_weights(w_vol: float, w_dii: float, w_tr: float, w_news: float) -> Tuple[float, float, float, float]:
    ws = np.array([w_vol, w_dii, w_tr, w_news], dtype=float)
    if np.all(np.isfinite(ws)) and ws.sum() > 0:
        ws = ws / ws.sum()
        return tuple(float(x) for x in ws.tolist())
    return (0.40, 0.10, 0.30, 0.20)


def render_chart(chart_dir: Path, symbol: str, weekly_df: Optional[pd.DataFrame]) -> Optional[str]:
    if plt is None or weekly_df is None or weekly_df.empty:
        return None

    try:
        ensure_dir(chart_dir)
        path = chart_dir / f"{symbol}.png"

        x = pd.to_datetime(weekly_df.index)
        y = first_series(weekly_df["Close"]).dropna()
        if y.empty:
            return None

        x = x[-len(y):]
        x = x[-13:]
        y = y.tail(13)

        ma20 = y.rolling(20, min_periods=1).mean()
        high_3m = float(y.max())
        last_x = x[-1]
        last_y = float(y.iloc[-1])

        fig = plt.figure(figsize=(6.4, 3.0))
        ax = fig.add_subplot(111)

        # 終値ライン
        ax.plot(x, y.to_numpy(), linewidth=2.0, label="Close")

        # 20週移動平均
        ax.plot(x, ma20.to_numpy(), linewidth=1.6, linestyle="--", label="20W MA")

        # 3か月高値ライン
        ax.axhline(high_3m, linewidth=1.2, linestyle=":", label="3M High")

        # 直近終値マーカー
        ax.scatter([last_x], [last_y], s=35, zorder=5)

        ax.set_title(f"{symbol} · Weekly · 3M")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

        return f"/charts/{chart_dir.name}/{symbol}.png"

    except Exception as e:
        log("WARN", f"{symbol}: chart render failed: {e}")
        return None


@dataclass
class Config:
    out_dir: Path
    universe_csv: Path
    report_date: str
    provider: str
    tiingo_token: Optional[str]
    mock_mode: bool
    w_vol: float
    w_dii: float
    w_tr: float
    w_news: float


def load_config() -> Config:
    out_dir = Path(env_s("OUT_DIR", "site"))
    universe_csv = Path(env_s("UNIVERSE_CSV", os.path.join("data", "universe.csv")))
    report_date = env_s("REPORT_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    provider = env_s("DATA_PROVIDER", "yfinance").lower()
    tiingo_token = os.getenv("TIINGO_TOKEN") or None
    mock_mode = env_b("MOCK_MODE", False)

    # DIIを0.10へ。Volumeを0.40へ。
    w_vol, w_dii, w_tr, w_news = normalize_weights(
        env_f("WEIGHT_VOL_ANOM", 0.40),
        env_f("WEIGHT_DII", 0.10),
        env_f("WEIGHT_TRENDS", 0.30),
        env_f("WEIGHT_NEWS", 0.20),
    )

    return Config(
        out_dir=out_dir,
        universe_csv=universe_csv,
        report_date=report_date,
        provider=provider,
        tiingo_token=tiingo_token,
        mock_mode=mock_mode,
        w_vol=w_vol,
        w_dii=w_dii,
        w_tr=w_tr,
        w_news=w_news,
    )


def aggregate() -> None:
    cfg = load_config()
    out_day_dir = cfg.out_dir / "data" / cfg.report_date
    ensure_dir(out_day_dir)

    universe = load_universe(cfg.universe_csv)
    if not universe:
        raise SystemExit("Universe is empty")

    dii_map = extract_component_map(cfg.out_dir, cfg.report_date, "dii")
    trends_map = extract_component_map(cfg.out_dir, cfg.report_date, "trends")
    news_map = extract_component_map(cfg.out_dir, cfg.report_date, "news")

    rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")
        log("INFO", f"Processing {sym}")

        df = None if cfg.mock_mode else fetch_history(sym, cfg.provider, cfg.tiingo_token, months=12)
        if df is None and not cfg.mock_mode:
            failed.append(sym)

        d1 = pct(df["Close"], 1) if df is not None else None
        d5 = pct(df["Close"], 5) if df is not None else None
        d20 = pct(df["Close"], 20) if df is not None else None
        vol_detail, vol_score = compute_vol_anomaly(df) if df is not None else ({"eligible": False}, 0.0)

        dii_val = float(dii_map.get(sym, 0.0))
        trends_val = float(trends_map.get(sym, 0.0))
        news_val = float(news_map.get(sym, 0.0))

        comps = {
            "volume_anomaly": round(vol_score, 6),
            "dii": round(dii_val, 6),
            "trends_breakout": round(trends_val, 6),
            "news": round(news_val, 6),
        }
        weights = {
            "volume_anomaly": cfg.w_vol,
            "dii": cfg.w_dii,
            "trends_breakout": cfg.w_tr,
            "news": cfg.w_news,
        }

        final01 = (
            comps["volume_anomaly"] * weights["volume_anomaly"]
            + comps["dii"] * weights["dii"]
            + comps["trends_breakout"] * weights["trends_breakout"]
            + comps["news"] * weights["news"]
        )
        final01 = round(max(0.0, min(1.0, final01)), 6)

        rows.append({
            "symbol": sym,
            "name": nm,
            "final_score_0_1": final01,
            "score_pts": int(round(final01 * 1000)),
            "score_components": comps,
            "score_weights": weights,
            "price_delta_1d": None if d1 is None else round(float(d1), 2),
            "price_delta_1w": None if d5 is None else round(float(d5), 2),
            "price_delta_1m": None if d20 is None else round(float(d20), 2),
            "detail": {"vol_anomaly": vol_detail},
            "chart_url": None,
        })

    rows.sort(key=lambda r: (r["score_pts"], r["final_score_0_1"]), reverse=True)
    top10 = rows[:10]

    chart_dir = cfg.out_dir / "charts" / cfg.report_date
    for idx, item in enumerate(top10, start=1):
        item["rank"] = idx
        if not cfg.mock_mode:
            weekly = fetch_weekly_history(item["symbol"], cfg.provider, cfg.tiingo_token)
            item["chart_url"] = render_chart(chart_dir, item["symbol"], weekly)

    payload = {"date": cfg.report_date, "items": top10}
    write_json(out_day_dir / "top10.json", payload)
    write_json(cfg.out_dir / "data" / "top10" / "latest.json", payload)

    log("INFO", f"Wrote Top10: {out_day_dir / 'top10.json'}")
    if failed:
        log("WARN", f"Failed market data for {len(failed)} symbols: {failed}")


if __name__ == "__main__":
    try:
        aggregate()
    except Exception as e:
        log("ERROR", f"FATAL in fetch_daily: {e}")
        sys.exit(1)
