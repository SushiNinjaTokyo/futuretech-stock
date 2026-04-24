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

    cand = sorted(
        [d.name for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 10],
        reverse=True,
    )
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


def clamp01(x: Any) -> float:
    f = to_float(x)
    if f is None:
        return 0.0
    return max(0.0, min(1.0, f))


def canonicalize_score_components(comps_raw: Dict[str, Any]) -> Dict[str, float]:
    comps_raw = comps_raw or {}
    return {
        "volume_anomaly": clamp01(comps_raw.get("volume_anomaly")),
        "compression_release": clamp01(comps_raw.get("compression_release", comps_raw.get("dii"))),
        "trends_breakout": clamp01(comps_raw.get("trends_breakout")),
        "news": clamp01(comps_raw.get("news")),
    }


def canonicalize_score_weights(weights_raw: Dict[str, Any]) -> Dict[str, float]:
    weights_raw = weights_raw or {}
    return {
        "volume_anomaly": max(0.0, to_float(weights_raw.get("volume_anomaly")) or 0.0),
        "compression_release": max(0.0, to_float(weights_raw.get("compression_release", weights_raw.get("dii"))) or 0.0),
        "trends_breakout": max(0.0, to_float(weights_raw.get("trends_breakout")) or 0.0),
        "news": max(0.0, to_float(weights_raw.get("news")) or 0.0),
    }


def normalize_chart_badges(detail_raw: Dict[str, Any]) -> Dict[str, Any]:
    detail_raw = detail_raw or {}
    badges = detail_raw.get("chart_badges") or {}

    def norm_badge(raw: Dict[str, Any], default_tone: str = "hold") -> Dict[str, Any]:
        raw = raw or {}
        tone = str(raw.get("tone", default_tone)).strip().lower()
        return {
            "value": to_float(raw.get("value")),
            "display": str(raw.get("display", "N/A")),
            "label": str(raw.get("label", "N/A")),
            "tone": tone,
        }

    return {
        "close_pos": norm_badge(badges.get("close_pos"), "hold"),
        "rvol": norm_badge(badges.get("rvol"), "hold"),
        "thrust": norm_badge(badges.get("thrust"), "hold"),
        "vol_setup": norm_badge(badges.get("vol_setup"), "flat"),
    }


def normalize_relative_strength(detail_raw: Dict[str, Any]) -> Dict[str, Any]:
    detail_raw = detail_raw or {}
    rel = detail_raw.get("relative_strength") or {}

    def norm_inner(raw: Dict[str, Any], default_tone: str = "neutral") -> Dict[str, Any]:
        raw = raw or {}
        tone = str(raw.get("tone", default_tone)).strip().lower()
        return {
            "value": to_float(raw.get("value")),
            "display": str(raw.get("display", "N/A")),
            "label": str(raw.get("label", "N/A")),
            "tone": tone,
        }

    out: Dict[str, Any] = {}
    for key in ("sp500", "nasdaq", "russell"):
        row = rel.get(key) or {}
        out[key] = {
            "name": str(row.get("name", key)),
            "price": norm_inner(row.get("price"), "neutral"),
            "vol_accel": norm_inner(row.get("vol_accel"), "neutral"),
        }
    return out


def normalize_hero_index_lines(raw: Any) -> Dict[str, Any]:
    raw = raw or {}
    out: Dict[str, Any] = {}

    for key in ("sp500", "nasdaq", "russell"):
        row = raw.get(key) or {}
        pts = row.get("points") or []
        clean_points: List[float] = []
        for p in pts:
            fp = to_float(p)
            if fp is not None:
                clean_points.append(max(0.0, min(100.0, fp)))

        out[key] = {
            "label": str(row.get("label", key.upper())),
            "symbol": str(row.get("symbol", "")),
            "points": clean_points,
            "change_1m": to_float(row.get("change_1m")),
        }

    return out


def normalize_item(item: Dict[str, Any], date: str, rank: int) -> Dict[str, Any]:
    sym = str(item.get("symbol", "")).strip().upper()
    nm = str(item.get("name", "")).strip()

    comps_raw = item.get("score_components") or {}
    weights_raw = item.get("score_weights") or {}

    comps = canonicalize_score_components(comps_raw)
    weights = canonicalize_score_weights(weights_raw)
    final01 = clamp01(item.get("final_score_0_1"))

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

    detail = item.get("detail") or {}
    detail["chart_badges"] = normalize_chart_badges(detail)
    detail["relative_strength"] = normalize_relative_strength(detail)

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
        "detail": detail,
    }


def load_payload_for(date: str) -> Dict[str, Any]:
    day_path = OUT_DIR / "data" / date / "top10.json"
    latest_path = OUT_DIR / "data" / "top10" / "latest.json"

    j = read_json(day_path)
    if not j:
        j = read_json(latest_path)
    if not j:
        return {"hero_index_lines": {}, "items": []}

    payload = j if isinstance(j, dict) else {"items": j}
    items_raw = payload.get("items", [])
    if not isinstance(items_raw, list):
        items_raw = []

    items = [normalize_item(x, date, i + 1) for i, x in enumerate(items_raw[:10])]
    hero_index_lines = normalize_hero_index_lines(payload.get("hero_index_lines"))

    return {
        "hero_index_lines": hero_index_lines,
        "items": items,
    }


def main() -> None:
    date = pick_date_dir()
    payload = load_payload_for(date)
    items = payload["items"]
    hero_index_lines = payload["hero_index_lines"]

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
        hero_index_lines=hero_index_lines,
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
