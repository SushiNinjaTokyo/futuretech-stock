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
from typing import Any, Dict, List, Optional

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

DAILY_V2_DIR = OUT_DIR / "data" / "daily-v2"

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


def is_yyyy_mm_dd(s: str) -> bool:
    if len(s) != 10:
        return False
    try:
        pd.Timestamp(s)
        return s[4] == "-" and s[7] == "-"
    except Exception:
        return False


def find_latest_daily_v2_payload() -> Dict[str, Any]:
    """
    Source of truth for the home page daily card:
    site/data/daily-v2/latest.json, or max existing date under daily-v2.
    Never use v1 /daily/YYYY-MM-DD.html links here.
    """
    latest = read_json(DAILY_V2_DIR / "latest.json")
    if isinstance(latest, dict) and latest.get("items") is not None:
        return latest

    candidates: List[str] = []
    if DAILY_V2_DIR.exists():
        for d in DAILY_V2_DIR.iterdir():
            if d.is_dir() and is_yyyy_mm_dd(d.name) and (d / "top10.json").exists():
                candidates.append(d.name)

    if candidates:
        latest_date = sorted(candidates)[-1]
        payload = read_json(DAILY_V2_DIR / latest_date / "top10.json")
        if isinstance(payload, dict):
            return payload

    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "items": [],
        "summary": {},
    }


def pick_latest_date() -> str:
    if REPORT_DATE:
        return REPORT_DATE

    payload = find_latest_daily_v2_payload()
    if payload.get("date"):
        return str(payload["date"])

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


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


def fallback_points(seed: float = 0.0, n: int = 40) -> List[float]:
    pts: List[float] = []
    base = 48.0 + seed
    for i in range(n):
        x = base + math.sin(i / 3.0 + seed) * 18.0 + math.cos(i / 5.0) * 7.0 + i * 0.35
        pts.append(round(max(5.0, min(95.0, x)), 3))
    return pts


def build_market_pulse(date: str) -> Dict[str, Any]:
    indices: List[Dict[str, Any]] = []
    risk_score = 0
    valid_count = 0

    for idx, (key, meta) in enumerate(INDEX_SYMBOLS.items()):
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
            "points": fallback_points(idx * 4.0),
        }

        if df is not None and len(df) >= 22:
            close = df["Close"].dropna()
            row["return_1d"] = pct(float(close.iloc[-1]), float(close.iloc[-2])) if len(close) >= 2 else None
            row["return_5d"] = pct(float(close.iloc[-1]), float(close.iloc[-6])) if len(close) >= 6 else None
            row["return_20d"] = pct(float(close.iloc[-1]), float(close.iloc[-21])) if len(close) >= 21 else None
            row["above_20dma"] = bool(float(close.iloc[-1]) >= float(close.tail(20).mean()))
            row["points"] = normalize_points(close, 40) or row["points"]

            valid_count += 1
            if row["above_20dma"]:
                risk_score += 1
            if row["return_5d"] is not None and row["return_5d"] > 0:
                risk_score += 1

        indices.append(row)

    if valid_count == 0:
        regime = "Unknown"
        regime_tone = "neutral"
    elif risk_score >= 5:
        regime = "Risk-On"
        regime_tone = "risk-on"
    elif risk_score >= 3:
        regime = "Neutral"
        regime_tone = "neutral"
    else:
        regime = "Risk-Off"
        regime_tone = "risk-off"

    payload = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "regime": regime,
        "regime_tone": regime_tone,
        "indices": indices,
        "links": {
            "daily_latest": "/daily/",
            "backtest": "/backtest/",
        },
    }

    write_json(OUT_DIR / "data" / "index" / "market_pulse_latest.json", payload)
    return payload


def score_of(item: Dict[str, Any]) -> Optional[float]:
    for key in ("score_pts", "daily_score", "score", "final_score_pts", "total_score"):
        v = safe_float(item.get(key))
        if v is not None:
            return v
    v = safe_float(item.get("final_score_0_1"))
    if v is not None:
        return round(v * 1000.0, 1)
    return None


def name_of(item: Dict[str, Any]) -> Optional[str]:
    for key in ("name", "company", "company_name", "short_name"):
        if item.get(key):
            return str(item[key])
    return None


def classification_of(item: Dict[str, Any]) -> str:
    for key in ("classification", "triage", "signal_class", "action"):
        if item.get(key):
            return str(item[key])
    score = score_of(item)
    if score is None:
        return "Latest ranking"
    if score >= 750:
        return "Trade setup"
    if score >= 700:
        return "Watch setup"
    return "Monitor"


def load_latest_top10() -> Dict[str, Any]:
    payload = find_latest_daily_v2_payload()
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []

    top = items[0] if items and isinstance(items[0], dict) else None

    return {
        "date": payload.get("date", pick_latest_date()),
        "items_count": len(items),
        "top_symbol": top.get("symbol") if isinstance(top, dict) else None,
        "top_name": name_of(top) if isinstance(top, dict) else None,
        "top_score": score_of(top) if isinstance(top, dict) else None,
        "top_classification": classification_of(top) if isinstance(top, dict) else "Latest ranking",
    }


def render() -> None:
    daily_payload = find_latest_daily_v2_payload()
    date = str(daily_payload.get("date") or pick_latest_date())

    market = build_market_pulse(date)
    top10_meta = load_latest_top10()

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
