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
    """
    Open / High / Low / Close / Volume の1次元列へ正規化。
    yfinance / tiingo の返り値差異を吸収する。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = pd.DataFrame(index=pd.to_datetime(df.index))

    targets = ["open", "high", "low", "close", "volume"]

    if isinstance(df.columns, pd.MultiIndex):
        chosen: Dict[str, Any] = {}
        for col in df.columns:
            parts = [str(c).strip().lower() for c in col if c is not None]
            for t in targets:
                if t in parts and t not in chosen:
                    chosen[t] = col

        for t in targets:
            if t in chosen:
                out[t.capitalize()] = first_series(df.loc[:, chosen[t]]).to_numpy()
            else:
                out[t.capitalize()] = np.nan

    else:
        src_map = {str(c).strip().lower(): c for c in df.columns}
        for t in targets:
            src = src_map.get(t)
            if src is not None:
                ser = df[[src]] if isinstance(df[src], pd.DataFrame) else df[src]
                out[t.capitalize()] = first_series(ser).to_numpy()
            else:
                out[t.capitalize()] = np.nan

    return out[["Open", "High", "Low", "Close", "Volume"]].dropna()


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


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


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

                rename_map = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
                for src, dst in rename_map.items():
                    if src in raw.columns:
                        raw[dst] = raw[src]

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

            if raw is None or len(raw) < 60:
                raise ValueError("empty dataframe")

            df = normalize_ohlcv(raw)
            if len(df) < 60:
                raise ValueError("not enough clean rows")

            return df.sort_index()

        except Exception as e:
            log("WARN", f"{symbol}: fetch_history attempt {attempt + 1} failed: {e}")
            time.sleep(1.5 + attempt * 0.8)

    return None


def price_direction_metrics(df: pd.DataFrame) -> Dict[str, float]:
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

    s5 = max(0.0, min(1.0, ret_5d / 0.08))
    s20 = max(0.0, min(1.0, ret_20d / 0.15))
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

    s_rvol = 1.0 - math.exp(-max(0.0, rvol20) / 2.5)
    s_z = 1.0 / (1.0 + math.exp(-z60))
    s_pr = max(0.0, min(1.0, pct_rank_90))
    vol_only_score = max(0.0, min(1.0, 0.45 * s_pr + 0.40 * s_rvol + 0.15 * s_z))

    pdm = price_direction_metrics(pd.DataFrame({"Close": close, "Volume": vol}))
    direction_score = float(pdm["direction_score"])

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


def normalize_weights(w_vol: float, w_comp: float, w_tr: float, w_news: float) -> Tuple[float, float, float, float]:
    ws = np.array([w_vol, w_comp, w_tr, w_news], dtype=float)
    if np.all(np.isfinite(ws)) and ws.sum() > 0:
        ws = ws / ws.sum()
        return tuple(float(x) for x in ws.tolist())
    return (0.35, 0.15, 0.25, 0.25)


def classify_badge_close_pos(v: float) -> Tuple[str, str]:
    if v >= 0.8:
        return "Buy", "buy"
    if v >= 0.55:
        return "Hold", "hold"
    return "Sell", "sell"


def classify_badge_rvol(v: float) -> Tuple[str, str]:
    if v >= 1.8:
        return "Buy", "buy"
    if v >= 1.1:
        return "Hold", "hold"
    return "Sell", "sell"


def classify_badge_thrust(v: float) -> Tuple[str, str]:
    if v >= 0.7:
        return "Buy", "buy"
    if v >= 0.25:
        return "Hold", "hold"
    return "Sell", "sell"


def build_chart_badges(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 25:
        return {
            "close_pos": {"value": None, "label": "N/A", "tone": "hold"},
            "rvol": {"value": None, "label": "N/A", "tone": "hold"},
            "thrust": {"value": None, "label": "N/A", "tone": "hold"},
        }

    close = first_series(df["Close"]).dropna()
    high = first_series(df["High"]).dropna()
    low = first_series(df["Low"]).dropna()
    vol = first_series(df["Volume"]).dropna()

    n = min(len(close), len(high), len(low), len(vol))
    close = close.iloc[-n:]
    high = high.iloc[-n:]
    low = low.iloc[-n:]
    vol = vol.iloc[-n:]

    hi10 = float(high.tail(10).max())
    lo10 = float(low.tail(10).min())
    close_pos = 0.5 if hi10 <= lo10 else float((close.iloc[-1] - lo10) / (hi10 - lo10))
    close_pos = clamp(close_pos)

    avg_vol20 = float(vol.tail(21).iloc[:-1].mean()) if len(vol) >= 21 else float(vol.mean())
    rvol20 = float(vol.iloc[-1] / (avg_vol20 + 1e-9))

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1).dropna()
    atr10 = float(tr.tail(10).mean()) if len(tr) >= 10 else 0.0
    thrust_atr = 0.0 if atr10 <= 0 else max(0.0, float(close.iloc[-1] - close.iloc[-2]) / atr10)

    cp_label, cp_tone = classify_badge_close_pos(close_pos)
    rv_label, rv_tone = classify_badge_rvol(rvol20)
    th_label, th_tone = classify_badge_thrust(thrust_atr)

    return {
        "close_pos": {
            "value": round(close_pos, 3),
            "display": f"{round(close_pos * 100)}%",
            "label": cp_label,
            "tone": cp_tone,
        },
        "rvol": {
            "value": round(rvol20, 2),
            "display": f"{round(rvol20, 2)}x",
            "label": rv_label,
            "tone": rv_tone,
        },
        "thrust": {
            "value": round(thrust_atr, 2),
            "display": f"{round(thrust_atr, 2)} ATR",
            "label": th_label,
            "tone": th_tone,
        },
    }


def render_chart(chart_dir: Path, symbol: str, daily_df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    追加APIを叩かず、既に取得済みの日足OHLCVをそのまま流用して描画する。
    20WMAは廃止し、以下を表示:
    - 10日高値ライン
    - 10DMA / 20DMA
    - 出来高バー + 20日平均出来高
    """
    if plt is None or daily_df is None or daily_df.empty:
        return None

    try:
        ensure_dir(chart_dir)
        path = chart_dir / f"{symbol}.png"

        df = daily_df.copy().sort_index()
        need_cols = {"High", "Low", "Close", "Volume"}
        if not need_cols.issubset(set(df.columns)):
            return None

        df = df.tail(65).copy()
        if len(df) < 25:
            return None

        close = first_series(df["Close"]).dropna()
        high = first_series(df["High"]).dropna()
        vol = first_series(df["Volume"]).dropna()

        n = min(len(close), len(high), len(vol), len(df.index))
        x = pd.to_datetime(df.index[-n:])
        close = close.iloc[-n:]
        high = high.iloc[-n:]
        vol = vol.iloc[-n:]

        close_plot = close.tail(60)
        high_plot = high.tail(60)
        vol_plot = vol.tail(60)
        x_plot = x[-len(close_plot):]

        ma10 = close_plot.rolling(10, min_periods=10).mean()
        ma20 = close_plot.rolling(20, min_periods=20).mean()
        high10 = high_plot.rolling(10, min_periods=10).max()
        vol_ma20 = vol_plot.rolling(20, min_periods=20).mean()

        last_x = x_plot[-1]
        last_y = float(close_plot.iloc[-1])

        bg = "#0b1730"
        line_main = "#5ee7ff"
        line_10dma = "#5b8cff"
        line_20dma = "#9d7bff"
        line_high10 = "#ffcf5a"
        volume_fill = "#2f6fff"
        volume_ma = "#f6c453"
        grid = (1.0, 1.0, 1.0, 0.10)
        tick = "#b9cae1"
        border = "#28405f"

        fig = plt.figure(figsize=(7.2, 4.6), facecolor=bg)
        gs = fig.add_gridspec(100, 1)

        ax = fig.add_subplot(gs[:68, 0])
        axv = fig.add_subplot(gs[74:, 0], sharex=ax)

        ax.set_facecolor(bg)
        axv.set_facecolor(bg)

        ax.plot(x_plot, close_plot.to_numpy(), linewidth=2.2, color=line_main, label="Close", zorder=4)
        if ma10.notna().any():
            ax.plot(x_plot, ma10.to_numpy(), linewidth=1.5, color=line_10dma, label="10DMA", zorder=3)
        if ma20.notna().any():
            ax.plot(x_plot, ma20.to_numpy(), linewidth=1.5, color=line_20dma, label="20DMA", zorder=2)
        if high10.notna().any():
            ax.plot(x_plot, high10.to_numpy(), linewidth=1.2, linestyle="--", color=line_high10, label="10D High", zorder=1)

        ax.scatter([last_x], [last_y], s=28, color=line_main, edgecolors="white", linewidths=0.8, zorder=5)

        axv.bar(x_plot, vol_plot.to_numpy(), width=0.8, color=volume_fill, alpha=0.45)
        if vol_ma20.notna().any():
            axv.plot(x_plot, vol_ma20.to_numpy(), linewidth=1.4, color=volume_ma, label="20D Avg Vol")

        ax.set_title(f"{symbol} · Daily · 3M", color="white", fontsize=12, pad=10, fontweight="bold")

        for axis in (ax, axv):
            axis.grid(True, alpha=0.10, color=grid)
            axis.tick_params(colors=tick, labelsize=8)
            for spine in axis.spines.values():
                spine.set_color(border)

        ax.legend(
            loc="upper left",
            fontsize=7.5,
            frameon=False,
            labelcolor=tick,
            ncol=4,
            handlelength=2.4,
            borderaxespad=0.8,
        )

        axv.legend(
            loc="upper left",
            fontsize=7.0,
            frameon=False,
            labelcolor=tick,
            handlelength=2.0,
            borderaxespad=0.6,
        )

        ax.tick_params(axis="x", labelbottom=False)
        axv.set_ylabel("Vol", color=tick, fontsize=8)
        axv.set_yticks([])
        axv.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
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
    w_comp: float
    w_tr: float
    w_news: float


def load_config() -> Config:
    out_dir = Path(env_s("OUT_DIR", "site"))
    universe_csv = Path(env_s("UNIVERSE_CSV", os.path.join("data", "universe.csv")))
    report_date = env_s("REPORT_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    provider = env_s("DATA_PROVIDER", "yfinance").lower()
    tiingo_token = os.getenv("TIINGO_TOKEN") or None
    mock_mode = env_b("MOCK_MODE", False)

    w_vol, w_comp, w_tr, w_news = normalize_weights(
        env_f("WEIGHT_VOL_ANOM", 0.35),
        env_f("WEIGHT_COMPRESSION", 0.15),
        env_f("WEIGHT_TRENDS", 0.25),
        env_f("WEIGHT_NEWS", 0.25),
    )

    return Config(
        out_dir=out_dir,
        universe_csv=universe_csv,
        report_date=report_date,
        provider=provider,
        tiingo_token=tiingo_token,
        mock_mode=mock_mode,
        w_vol=w_vol,
        w_comp=w_comp,
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

    compression_map = extract_component_map(cfg.out_dir, cfg.report_date, "compression")
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

        comp_val = float(compression_map.get(sym, 0.0))
        trends_val = float(trends_map.get(sym, 0.0))
        news_val = float(news_map.get(sym, 0.0))

        comps = {
            "volume_anomaly": round(vol_score, 6),
            "compression_release": round(comp_val, 6),
            "trends_breakout": round(trends_val, 6),
            "news": round(news_val, 6),
        }
        weights = {
            "volume_anomaly": cfg.w_vol,
            "compression_release": cfg.w_comp,
            "trends_breakout": cfg.w_tr,
            "news": cfg.w_news,
        }

        final01 = (
            comps["volume_anomaly"] * weights["volume_anomaly"]
            + comps["compression_release"] * weights["compression_release"]
            + comps["trends_breakout"] * weights["trends_breakout"]
            + comps["news"] * weights["news"]
        )
        final01 = round(max(0.0, min(1.0, final01)), 6)

        chart_badges = build_chart_badges(df) if df is not None else {
            "close_pos": {"value": None, "display": "N/A", "label": "N/A", "tone": "hold"},
            "rvol": {"value": None, "display": "N/A", "label": "N/A", "tone": "hold"},
            "thrust": {"value": None, "display": "N/A", "label": "N/A", "tone": "hold"},
        }

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
            "detail": {
                "vol_anomaly": vol_detail,
                "chart_badges": chart_badges,
            },
            "chart_url": None,
        })

        # API再取得せず、その場で描画できるよう一時保持
        rows[-1]["_chart_df"] = df

    rows.sort(key=lambda r: (r["score_pts"], r["final_score_0_1"]), reverse=True)
    top10 = rows[:10]

    chart_dir = cfg.out_dir / "charts" / cfg.report_date
    for idx, item in enumerate(top10, start=1):
        item["rank"] = idx
        chart_df = item.pop("_chart_df", None)
        if not cfg.mock_mode and chart_df is not None:
            item["chart_url"] = render_chart(chart_dir, item["symbol"], chart_df)
        else:
            item["chart_url"] = None

    for item in rows[10:]:
        item.pop("_chart_df", None)

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
