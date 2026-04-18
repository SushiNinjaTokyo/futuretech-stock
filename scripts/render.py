#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = ROOT / "site"
REPORT_DATE = os.getenv("REPORT_DATE")


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("ERROR", f"read_json failed: {path}: {e}")
        return None


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def copy_asset(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def pick_date_dir() -> str:
    if REPORT_DATE:
        return REPORT_DATE
    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        raise SystemExit("site/data not found")
    cand = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 10], reverse=True)
    if not cand:
        raise SystemExit("no date directories under site/data")
    return cand[0]


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:
            return None
        return f
    except Exception:
        return None


def normalize_item(item: Dict[str, Any], date: str, rank: int) -> Dict[str, Any]:
    sym = str(item.get("symbol", "")).strip().upper()
    nm = str(item.get("name", "")).strip()

    comps_raw = item.get("score_components") or {}
    weights_raw = item.get("score_weights") or {}

    comps = {
        "volume_anomaly": to_float(comps_raw.get("volume_anomaly")) or 0.0,
        "dii": to_float(comps_raw.get("dii")) or 0.0,
        "trends_breakout": to_float(comps_raw.get("trends_breakout")) or 0.0,
        "news": to_float(comps_raw.get("news")) or 0.0,
    }
    weights = {
        "volume_anomaly": to_float(weights_raw.get("volume_anomaly")) or 0.0,
        "dii": to_float(weights_raw.get("dii")) or 0.0,
        "trends_breakout": to_float(weights_raw.get("trends_breakout")) or 0.0,
        "news": to_float(weights_raw.get("news")) or 0.0,
    }

    final01 = to_float(item.get("final_score_0_1")) or 0.0
    score_pts = item.get("score_pts")
    try:
        score_pts_int = int(score_pts) if score_pts is not None else int(round(final01 * 1000))
    except Exception:
        score_pts_int = int(round(final01 * 1000))

    chart_url = item.get("chart_url")
    if chart_url:
        chart_url = str(chart_url)
    else:
        candidate = OUT_DIR / "charts" / date / f"{sym}.png"
        chart_url = f"/charts/{date}/{sym}.png" if candidate.exists() else None

    return {
        "rank": rank,
        "symbol": sym,
        "name": nm,
        "final_score_0_1": round(final01, 6),
        "score_pts": score_pts_int,
        "price_delta_1d": to_float(item.get("price_delta_1d")),
        "price_delta_1w": to_float(item.get("price_delta_1w")),
        "price_delta_1m": to_float(item.get("price_delta_1m")),
        "score_components": comps,
        "score_weights": weights,
        "chart_url": chart_url,
        "detail": item.get("detail") or {},
    }


def load_top10_for(date: str) -> List[Dict[str, Any]]:
    day_path = OUT_DIR / "data" / date / "top10.json"
    latest_path = OUT_DIR / "data" / "top10" / "latest.json"

    j = read_json(day_path)
    if not j:
        j = read_json(latest_path)
    if not j:
        return []

    payload = j.get("items", j) if isinstance(j, dict) else j
    if not isinstance(payload, list):
        return []

    items = payload[:10]
    return [normalize_item(x, date, i + 1) for i, x in enumerate(items)]


def main() -> None:
    date = pick_date_dir()
    items = load_top10_for(date)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("daily.html.j2")

    html = tpl.render(
        date=date,
        top10=items,
        items=items,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    )

    out_html = OUT_DIR / "daily" / f"{date}.html"
    write_text(out_html, html)
    log("INFO", f"Rendered: {out_html}")

    css_src = TEMPLATES_DIR / "daily.css"
    if css_src.exists():
        css_dst = OUT_DIR / "assets" / "daily.css"
        copy_asset(css_src, css_dst)
        log("INFO", f"Copied CSS: {css_dst}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render: {e}")
        sys.exit(1)
