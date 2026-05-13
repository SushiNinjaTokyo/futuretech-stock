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
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
TEMPLATES_DIR = ROOT / "templates"
REPORT_DATE = os.getenv("REPORT_DATE", "").strip()


def log(level: str, msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}",
        flush=True,
    )


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("ERROR", f"read_json failed: {path}: {e}")
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def ensure_dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def ensure_list(v: Any) -> list:
    return v if isinstance(v, list) else []


def to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None or v == "":
            return default
        x = float(v)
        if not math.isfinite(x):
            return default
        return x
    except Exception:
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def load_payload() -> dict:
    if REPORT_DATE:
        p = OUT_DIR / "data" / "daily-v2" / REPORT_DATE / "top10.json"
        data = read_json(p)
        if isinstance(data, dict):
            return data

    data = read_json(OUT_DIR / "data" / "daily-v2" / "latest.json")
    if isinstance(data, dict):
        return data

    return {
        "date": REPORT_DATE or "N/A",
        "generated_at": None,
        "summary": {},
        "items": [],
        "market": {},
    }


def find_market_row(market: dict, candidates: list[str]) -> dict:
    lower_map = {str(k).lower(): ensure_dict(v) for k, v in market.items()}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for row in lower_map.values():
        sym = str(row.get("symbol", "")).lower()
        label = str(row.get("label", row.get("name", ""))).lower()
        if any(c.lower() in {sym, label} for c in candidates):
            return row
    return {}


def flat_no_data_points() -> list[float]:
    return [50.0 for _ in range(21)]


def valid_points(v: Any) -> list[float]:
    if not isinstance(v, list):
        return []
    pts: list[float] = []
    for x in v:
        try:
            f = float(x)
            if math.isfinite(f):
                pts.append(round(max(0.0, min(100.0, f)), 3))
        except Exception:
            continue
    return pts if len(pts) >= 2 else []

def build_hero_index_lines(market: dict) -> dict:
    market = ensure_dict(market)
    spec = {
        "sp500": {
            "label": "S&P 500",
            "symbol": "SPY",
            "keys": ["sp500", "s&p 500", "spy", "spx", "^gspc"],
        },
        "nasdaq": {
            "label": "NASDAQ",
            "symbol": "QQQ",
            "keys": ["nasdaq", "qqq", "ndx", "^ixic", "^ndx"],
        },
        "russell": {
            "label": "Russell 2000",
            "symbol": "IWM",
            "keys": ["russell", "iwm", "russell 2000", "^rut"],
        },
    }

    out: dict[str, dict[str, Any]] = {}
    for key, cfg in spec.items():
        row = ensure_dict(find_market_row(market, cfg["keys"]))
        points = valid_points(row.get("points"))
        has_real_points = len(points) >= 2 and str(row.get("data_status", "ok")).lower() != "missing"
        change_1m = to_float(row.get("ret_20d_pct"), None)
        if change_1m is None:
            change_1m = to_float(row.get("change_1m"), None)
        ret_5d = to_float(row.get("ret_5d_pct"), None)
        ret_1d = to_float(row.get("ret_1d_pct"), None)
        out[key] = {
            "label": row.get("label") or row.get("name") or cfg["label"],
            "symbol": row.get("symbol") or cfg["symbol"],
            "regime": row.get("regime") or row.get("state") or "—",
            "change_1m": change_1m,
            "ret_5d_pct": ret_5d,
            "ret_1d_pct": ret_1d,
            "points": points if has_real_points else flat_no_data_points(),
            "data_status": row.get("data_status") or ("ok" if has_real_points else "missing"),
            "has_real_points": has_real_points,
            "pulse_window": row.get("pulse_window") or "1M",
            "pulse_days": row.get("pulse_days") or len(points) or 21,
        }
    return out

def summarize_quality(items: list[dict]) -> dict:
    total = len(items)
    trade = sum(1 for x in items if str(x.get("triage", "")).lower() == "trade")
    watch = sum(1 for x in items if str(x.get("triage", "")).lower() == "watch")
    ignore = sum(1 for x in items if str(x.get("triage", "")).lower() == "ignore")
    scores = [to_float(x.get("score_pts")) for x in items]
    scores = [x for x in scores if x is not None]
    volumes = []
    compression = []
    timing = []
    penalties = []
    for item in items:
        c = ensure_dict(item.get("v2_components"))
        volumes.append(to_float(c.get("volume_liquidity_shock"), 0.0) or 0.0)
        compression.append(to_float(c.get("compression_release"), 0.0) or 0.0)
        timing.append(to_float(c.get("entry_timing"), 0.0) or 0.0)
        penalties.append(to_float(c.get("penalty"), 0.0) or 0.0)
    return {
        "total": total,
        "trade": trade,
        "watch": watch,
        "ignore": ignore,
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "avg_volume": round(sum(volumes) / len(volumes) * 100, 1) if volumes else None,
        "avg_compression": round(sum(compression) / len(compression) * 100, 1) if compression else None,
        "avg_timing": round(sum(timing) / len(timing) * 100, 1) if timing else None,
        "avg_penalty": round(sum(penalties) / len(penalties) * 100, 1) if penalties else None,
    }


def main() -> None:
    payload = load_payload()
    items = ensure_list(payload.get("items"))
    market = ensure_dict(payload.get("market"))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("daily.html.j2")

    html = tpl.render(
        payload=payload,
        date=payload.get("date", "N/A"),
        items=items,
        market=market,
        summary=ensure_dict(payload.get("summary")),
        hero_index_lines=build_hero_index_lines(market),
        quality_summary=summarize_quality(items),
        generated_at=payload.get("generated_at")
        or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    )

    out_html = OUT_DIR / "daily" / "index.html"
    write_text(out_html, html)
    log("INFO", f"Rendered: {out_html}")

    css_src = TEMPLATES_DIR / "daily.css"
    if css_src.exists():
        css_dst = OUT_DIR / "assets" / "daily.css"
        ensure_dir(css_dst.parent)
        shutil.copy2(css_src, css_dst)
        log("INFO", f"Copied CSS: {css_dst}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render_daily_v2: {e}")
        sys.exit(1)
