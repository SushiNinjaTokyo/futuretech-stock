#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]  # repo root
TEMPLATES_DIR = ROOT / "templates"
ASSETS_SRC_CSS = TEMPLATES_DIR / "daily.css"
OUT_DIR = ROOT / "site"
OUT_DAILY_DIR = OUT_DIR / "daily"
OUT_ASSETS_DIR = OUT_DIR / "assets"


def _read_json(p: Path) -> Any:
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def _ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _css_version(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except FileNotFoundError:
        ts = dt.datetime.utcnow().timestamp()
    return dt.datetime.utcfromtimestamp(ts).strftime("%Y%m%d%H%M%S")


def _to_list(obj: Any) -> List[Dict[str, Any]]:
    """dict でも list でも、list[dict] に正規化する"""
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("items", "top10", "data", "symbols"):
            if key in obj and isinstance(obj[key], list):
                return [x for x in obj[key] if isinstance(x, dict)]
        return [obj]
    return []


def _collect_component_keys(items: List[Dict[str, Any]], known_priority: Iterable[str]):
    """components の列見出しを決める（既知キー→その他キーの順）。順序は出現順を尊重。"""
    used_known: List[str] = []
    seen = set()
    # 既知キー（priority）で、実際に一度でも出現したものだけ採用
    for k in known_priority:
        for it in items:
            comps = it.get("components") or {}
            if isinstance(comps, dict) and k in comps:
                if k not in seen:
                    used_known.append(k)
                    seen.add(k)
                break  # Python の break はOK（テンプレ側では使わない）

    # その他キーは出現順で
    other: List[str] = []
    for it in items:
        comps = it.get("components") or {}
        if isinstance(comps, dict):
            for k in comps.keys():
                if k not in seen:
                    other.append(k)
                    seen.add(k)
    return used_known, other


def main():
    report_date = os.environ.get("REPORT_DATE") or dt.datetime.utcnow().strftime("%Y-%m-%d")

    print(f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z [INFO] [TPL] using file: templates/daily.html.j2")

    # 入力ファイル
    in_dir = OUT_DIR / "data" / report_date
    dii_json = _read_json(in_dir / "dii.json") or _read_json(OUT_DIR / "data" / "dii" / "latest.json")
    trends_json = _read_json(in_dir / "trends.json") or _read_json(OUT_DIR / "data" / "trends" / "latest.json")
    news_json = _read_json(in_dir / "news.json") or _read_json(OUT_DIR / "data" / "news" / "latest.json")
    top10_json = _read_json(in_dir / "top10.json") or _read_json(OUT_DIR / "data" / "top10" / "latest.json")

    # 正規化
    dii_items = _to_list(dii_json)
    trends_items = _to_list(trends_json)
    news_items = _to_list(news_json)
    top10_items = _to_list(top10_json)

    # components の既知キー（優先表示順）
    known_component_keys = [
        "score", "volume_anom", "volume", "dii", "trends", "news",
        "momentum", "volatility", "sentiment",
    ]
    comp_keys, other_keys = _collect_component_keys(top10_items, known_component_keys)

    # テンプレート
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("daily.html.j2")

    # CSS 配備
    _ensure_dirs()
    if ASSETS_SRC_CSS.exists():
        shutil.copyfile(ASSETS_SRC_CSS, OUT_ASSETS_DIR / "daily.css")
    css_ver = _css_version(ASSETS_SRC_CSS if ASSETS_SRC_CSS.exists() else OUT_ASSETS_DIR / "daily.css")

    # 出力
    title = f"Daily Top 10 — {report_date}"
    ctx: Dict[str, Any] = dict(
        title=title,
        date=report_date,
        top10=top10_items,
        dii=dii_items,
        trends=trends_items,
        news=news_items,
        css_version=css_ver,
        comp_keys=comp_keys,
        other_keys=other_keys,
    )

    out_html = OUT_DAILY_DIR / f"{report_date}.html"
    rendered = tpl.render(**ctx)
    out_html.write_text(rendered, encoding="utf-8")
    print(f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z [INFO] Rendered daily HTML: {out_html} (template=templates/daily.html.j2)")

    # index.html（なければ作る）
    index_html = OUT_DIR / "index.html"
    if not index_html.exists():
        index_html.write_text(
            f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=/daily/{report_date}.html">',
            encoding="utf-8",
        )

    if (OUT_ASSETS_DIR / "daily.css").exists():
        print(f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z [INFO] [ASSET] copied CSS -> {OUT_ASSETS_DIR / 'daily.css'}")

    print(f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z [INFO] [TIME] render_total: done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z [ERROR] FATAL in render: {e}")
        traceback.print_exc()
        raise
