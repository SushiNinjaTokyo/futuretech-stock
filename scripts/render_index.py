#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    import yfinance as yf
except Exception:
    yf = None


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
REPORT_DATE = os.getenv("REPORT_DATE")


INDEX_SYMBOLS = {
    "sp500": {
        "label": "S&P 500",
        "symbol": "SPY",
        "tone": "sp500",
    },
    "nasdaq": {
        "label": "NASDAQ",
        "symbol": "QQQ",
        "tone": "nasdaq",
    },
    "russell": {
        "label": "Russell 2000",
        "symbol": "IWM",
        "tone": "russell",
    },
}


def log(level: str, msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}",
        flush=True,
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def copy_asset(src: Path, dst: Path) -> None:
    if not src.exists():
        log("WARN", f"asset missing: {src}")
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def pick_latest_date() -> str:
    if REPORT_DATE:
        return REPORT_DATE

    latest = read_json(OUT_DIR / "data" / "top10" / "latest.json")
    if isinstance(latest, dict) and latest.get("date"):
        return str(latest["date"])

    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    candidates = sorted(
        [
            d.name
            for d in data_dir.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[:4].isdigit()
        ],
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def pct(cur: float, prev: float) -> Optional[float]:
    try:
        if prev == 0 or not math.isfinite(cur) or not math.isfinite(prev):
            return None
        return round((cur / prev - 1.0) * 100.0, 2)
    except Exception:
        return None


def normalize_points(close: pd.Series, n: int = 40) -> List[float]:
    s = pd.to_numeric(close, errors="coerce").dropna().tail(n)
    if len(s) < 2:
        return []

    mn = float(s.min())
    mx = float(s.max())
    if mx <= mn:
        return [50.0 for _ in s]

    return [round((float(x) - mn) / (mx - mn) * 100.0, 3) for x in s]


def fetch_index_history(symbol: str) -> Optional[pd.DataFrame]:
    if yf is None:
        return None

    try:
        raw = yf.download(
            symbol,
            period="3mo",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=False,
        )
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw.xs("Close", level=0, axis=1).iloc[:, 0]
        else:
            close = raw["Close"]

        out = pd.DataFrame(index=pd.to_datetime(raw.index))
        out["Close"] = pd.to_numeric(close, errors="coerce")
        return out.dropna()

    except Exception as e:
        log("WARN", f"index fetch failed: {symbol}: {e}")
        return None


def build_market_pulse(date: str) -> Dict[str, Any]:
    indices: List[Dict[str, Any]] = []

    risk_score = 0
    valid_count = 0

    for key, meta in INDEX_SYMBOLS.items():
        symbol = meta["symbol"]
        df = fetch_index_history(symbol)

        row: Dict[str, Any] = {
            "key": key,
            "label": meta["label"],
            "symbol": symbol,
            "tone": meta["tone"],
            "return_1d": None,
            "return_5d": None,
            "return_20d": None,
            "above_20dma": None,
            "points": [],
        }

        if df is not None and len(df) >= 22:
            close = df["Close"].dropna()
            row["return_1d"] = pct(float(close.iloc[-1]), float(close.iloc[-2])) if len(close) >= 2 else None
            row["return_5d"] = pct(float(close.iloc[-1]), float(close.iloc[-6])) if len(close) >= 6 else None
            row["return_20d"] = pct(float(close.iloc[-1]), float(close.iloc[-21])) if len(close) >= 21 else None
            row["above_20dma"] = bool(float(close.iloc[-1]) >= float(close.tail(20).mean()))
            row["points"] = normalize_points(close, 40)

            valid_count += 1
            if row["above_20dma"]:
                risk_score += 1
            if row["return_5d"] is not None and row["return_5d"] > 0:
                risk_score += 1

        indices.append(row)

    if valid_count == 0:
        regime = "Unknown"
        regime_tone = "neutral"
    else:
        # 3指数 x 2項目 = 最大6点
        if risk_score >= 5:
            regime = "Risk-On"
            regime_tone = "risk-on"
        elif risk_score >= 3:
            regime = "Neutral"
            regime_tone = "neutral"
        else:
            regime = "Risk-Off"
            regime_tone = "risk-off"

    latest_daily = f"/daily/{date}.html"
    if not (OUT_DIR / "daily" / f"{date}.html").exists():
        latest_daily = "/daily/"

    payload = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "regime": regime,
        "regime_tone": regime_tone,
        "indices": indices,
        "links": {
            "daily_latest": latest_daily,
            "backtest": "/backtest/",
        },
    }

    write_json(OUT_DIR / "data" / "index" / "market_pulse_latest.json", payload)
    return payload


def load_latest_top10(date: str) -> Dict[str, Any]:
    candidates = [
        OUT_DIR / "data" / date / "top10.json",
        OUT_DIR / "data" / "top10" / "latest.json",
    ]

    for path in candidates:
        j = read_json(path)
        if isinstance(j, dict):
            items = j.get("items")
            if isinstance(items, list):
                top = items[0] if items else None
                return {
                    "date": j.get("date", date),
                    "items_count": len(items),
                    "top_symbol": top.get("symbol") if isinstance(top, dict) else None,
                    "top_name": top.get("name") if isinstance(top, dict) else None,
                    "top_score": top.get("score_pts") if isinstance(top, dict) else None,
                }

    return {
        "date": date,
        "items_count": 0,
        "top_symbol": None,
        "top_name": None,
        "top_score": None,
    }


def render() -> None:
    date = pick_latest_date()
    market = build_market_pulse(date)
    top10_meta = load_latest_top10(date)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    tpl = env.get_template("index.html.j2")
    html = tpl.render(
        date=date,
        market=market,
        indices=market.get("indices", []),
        top10=top10_meta,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    )

    out_html = OUT_DIR / "index.html"
    write_text(out_html, html)
    log("INFO", f"Rendered index: {out_html}")

    copy_asset(TEMPLATES_DIR / "index.css", OUT_DIR / "assets" / "index.css")


if __name__ == "__main__":
    try:
        render()
    except Exception as e:
        log("ERROR", f"FATAL in render_index: {e}")
        sys.exit(1)