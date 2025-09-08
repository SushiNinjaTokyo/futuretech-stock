#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json, os, sys, datetime, pathlib, shutil
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = pathlib.Path(__file__).resolve().parents[1] if (pathlib.Path(__file__).name == "render.py") else pathlib.Path.cwd()
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = ROOT / "site"
REPORT_DATE = os.environ.get("REPORT_DATE")  # e.g. 2025-09-06

CANON_MAP = {
    # 正規化
    "price_vol_anom": "volume_anomaly",
    "vol_anom": "volume_anomaly",
    "vol_anomaly": "volume_anomaly",
    "news_coverage": "news",
    "news_score": "news",
    "dii": "insider_momo",
    "insider": "insider_momo",
    "insider_momentum": "insider_momo",
    "trends": "trends_breakout",
    "trends_peak": "trends_breakout",
    "trends_breakout": "trends_breakout",
    "volume_anomaly": "volume_anomaly",
    "news": "news",
    "insider_momo": "insider_momo",
}

def now_utc_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

def read_json(p: pathlib.Path) -> Optional[Any]:
    try:
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{now_utc_iso()} [ERROR] read_json failed for {p}: {e}", file=sys.stderr)
        return None

def write_text(p: pathlib.Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def copy_asset(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def pick_date_dir() -> str:
    """REPORT_DATE があればそれを、なければ data 配下の最新日付ディレクトリ"""
    if REPORT_DATE:
        return REPORT_DATE
    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        raise SystemExit("[ERROR] site/data が見つかりません")
    cand = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 10], reverse=True)
    if not cand:
        raise SystemExit("[ERROR] site/data/YYYY-MM-DD が見つかりません")
    return cand[0]

def canonize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        out[CANON_MAP.get(k, k)] = v
    return out

def fix_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Undefined 回避のための整形 + 互換キー付与"""
    date = item.get("date") or REPORT_DATE or ""
    sym = (item.get("symbol") or item.get("ticker") or "").upper()

    comps_raw = item.get("score_components") or {}
    weights_raw = item.get("score_weights") or {}
    comps = canonize_keys(comps_raw)
    weights = canonize_keys(weights_raw)

    # 価格差分
    def _optf(x): 
        try:
            return None if x is None else float(x)
        except Exception:
            return None
    d1  = _optf(item.get("price_delta_1d"))
    d5  = _optf(item.get("price_delta_1w"))
    d20 = _optf(item.get("price_delta_1m"))

    # スコア
    final01 = 0.0
    try:
        final01 = float(item.get("final_score_0_1") or 0.0)
        if final01 != final01:  # NaN
            final01 = 0.0
    except Exception:
        final01 = 0.0
    score_pts = int(round(final01 * 1000))

    chart_url = item.get("chart_url") or f"/charts/{date}/{sym}.png"

    fixed = dict(item)
    fixed.update({
        "symbol": sym,
        "score_components": comps,
        "score_weights": weights,
        "price_delta_1d": d1,
        "price_delta_1w": d5,
        "price_delta_1m": d20,
        "final_score_0_1": final01,
        "score_pts": score_pts,
        "chart_url": chart_url,
        # 旧キー互換（JS/テンプレから参照されても良いように）
        "news_score": item.get("news_score", comps.get("news")),
        "trends_breakout": item.get("trends_breakout", comps.get("trends_breakout")),
        "vol_anomaly_score": item.get("vol_anomaly_score", comps.get("volume_anomaly")),
        "insider_momo": item.get("insider_momo", comps.get("insider_momo")),
    })
    return fixed

def load_top10_for(date_str: str) -> List[Dict[str, Any]]:
    day = OUT_DIR / "data" / date_str / "top10.json"
    latest = OUT_DIR / "data" / "top10" / "latest.json"

    payload = read_json(day)
    items: List[Dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    elif isinstance(payload, list):
        items = payload or []

    # フォールバック: 空または欠損なら latest.json
    if not items:
        fb = read_json(latest)
        if isinstance(fb, dict) and isinstance(fb.get("items"), list):
            items = fb["items"]
        elif isinstance(fb, list):
            items = fb or []

    return [fix_item(x) for x in items[:10]]

def main() -> None:
    date = pick_date_dir()
    print(f"{now_utc_iso()} [INFO] Render target date: {date}")

    items = load_top10_for(date)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True, lstrip_blocks=True,
    )
    tpl = env.get_template("daily.html.j2")

    # 互換のため items も注入（旧テンプレが items を見る場合に備える）
    html = tpl.render(
        date=date,
        top10=items,
        items=items,
        generated_at=now_utc_iso(),
    )

    out_html = OUT_DIR / "daily" / f"{date}.html"
    write_text(out_html, html)
    print(f"{now_utc_iso()} [INFO] Rendered daily HTML: {out_html} (template=templates/daily.html.j2)")

    # 任意: CSS アセットコピー（存在すれば）
    css_src = TEMPLATES_DIR / "daily.css"
    if css_src.exists():
        css_dst = OUT_DIR / "assets" / "daily.css"
        copy_asset(css_src, css_dst)
        print(f"{now_utc_iso()} [INFO] [ASSET] copied CSS -> {css_dst}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{now_utc_iso()} [ERROR] FATAL in render: {e}", file=sys.stderr)
        raise
