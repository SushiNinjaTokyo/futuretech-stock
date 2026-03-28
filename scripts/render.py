#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, os, sys, datetime, pathlib, shutil
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = ROOT / "site"
REPORT_DATE = os.environ.get("REPORT_DATE")


# ======================
# logging
# ======================
def log(level, msg):
    print(f"{datetime.datetime.utcnow().isoformat()}Z [{level}] {msg}", flush=True)


# ======================
# util
# ======================
def read_json(p: pathlib.Path) -> Optional[Any]:
    try:
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log("ERROR", f"read_json failed: {p} {e}")
        return None


def write_text(p: pathlib.Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def copy_asset(src: pathlib.Path, dst: pathlib.Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ======================
# 日付決定
# ======================
def pick_date():
    if REPORT_DATE:
        return REPORT_DATE

    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        raise SystemExit("site/data not found")

    dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    dirs = [d for d in dirs if len(d) == 10]

    if not dirs:
        raise SystemExit("no date dirs")

    return sorted(dirs)[-1]


# ======================
# 型安全変換
# ======================
def to_float(x):
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except:
        return None


# ======================
# データ整形（重要）
# ======================
def normalize_item(item: Dict[str, Any], date: str) -> Dict[str, Any]:

    sym = str(item.get("symbol", "")).upper()

    comps = item.get("score_components") or {}
    weights = item.get("score_weights") or {}

    # ← ここが重要：キー統一（diiをそのまま使う）
    comps = {
        "volume_anomaly": to_float(comps.get("volume_anomaly")),
        "dii": to_float(comps.get("dii")),
        "trends_breakout": to_float(comps.get("trends_breakout")),
        "news": to_float(comps.get("news")),
    }

    weights = {
        "volume_anomaly": to_float(weights.get("volume_anomaly")),
        "dii": to_float(weights.get("dii")),
        "trends_breakout": to_float(weights.get("trends_breakout")),
        "news": to_float(weights.get("news")),
    }

    final_score = to_float(item.get("final_score_0_1")) or 0.0

    return {
        "symbol": sym,
        "name": item.get("name", ""),
        "final_score_0_1": final_score,
        "score_pts": int(final_score * 1000),

        "price_delta_1d": to_float(item.get("price_delta_1d")),
        "price_delta_1w": to_float(item.get("price_delta_1w")),
        "price_delta_1m": to_float(item.get("price_delta_1m")),

        "score_components": comps,
        "score_weights": weights,

        "chart_url": item.get("chart_url") or f"/charts/{date}/{sym}.png",
    }


# ======================
# データロード
# ======================
def load_top10(date: str) -> List[Dict[str, Any]]:

    path = OUT_DIR / "data" / date / "top10.json"
    data = read_json(path)

    items = []

    if isinstance(data, dict):
        items = data.get("items", [])
    elif isinstance(data, list):
        items = data

    if not items:
        log("WARN", "top10 empty")
        return []

    return [normalize_item(x, date) for x in items[:10]]


# ======================
# main
# ======================
def main():

    date = pick_date()
    log("INFO", f"render date: {date}")

    items = load_top10(date)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )

    tpl = env.get_template("daily.html.j2")

    html = tpl.render(
        date=date,
        top10=items,
        items=items,
        generated_at=datetime.datetime.utcnow().isoformat()
    )

    out_html = OUT_DIR / "daily" / f"{date}.html"
    write_text(out_html, html)

    log("INFO", f"rendered: {out_html}")

    # CSSコピー
    css = TEMPLATES_DIR / "daily.css"
    if css.exists():
        copy_asset(css, OUT_DIR / "assets" / "daily.css")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL: {e}")
        sys.exit(1)
