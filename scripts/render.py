#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
SITE = ROOT / "site"

def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _first_existing(paths):
    for p in paths:
        if p and p.exists():
            return p
    return None

def _as_top10_list(obj):
    """
    入力が list でも dict でも安全に [item, ...] へ。
    必要キーが無い場合も可能な限り補完（symbol, rank, final_score_0_1, score_pts など）
    """
    if not obj:
        return []
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if isinstance(v, dict):
                v = dict(v)
                v.setdefault("symbol", v.get("symbol", k))
                items.append(v)
    else:
        return []

    # rank 補完・スコアの整形
    for i, it in enumerate(items, start=1):
        it.setdefault("rank", it.get("rank", i))
        score01 = it.get("final_score_0_1", it.get("final_score", it.get("score", 0))) or 0
        try:
            score01 = float(score01)
        except Exception:
            score01 = 0.0
        it["final_score_0_1"] = max(0.0, min(1.0, score01))
        it.setdefault("score_pts", int(round(it["final_score_0_1"] * 1000)))
        # 価格差分のフォールバック
        if "price_delta_1d" not in it and "delta_1d" in it: it["price_delta_1d"] = it["delta_1d"]
        if "price_delta_1w" not in it and "delta_1w" in it: it["price_delta_1w"] = it["delta_1_w"] if "delta_1_w" in it else it["delta_1w"]
        if "price_delta_1m" not in it and "delta_1m" in it: it["price_delta_1m"] = it["delta_1m"]
    # rank優先/スコア優先で安定ソート
    items.sort(key=lambda x: (int(x.get("rank", 9999)), -float(x.get("final_score_0_1", 0.0))))
    return items

def main():
    report_date = os.environ.get("REPORT_DATE", "").strip()
    if not report_date:
        print("[WARN] REPORT_DATE not set; trying latest files", file=sys.stderr)

    # 入力ファイルを解決
    top10_p = _first_existing([
        SITE / "data" / "top10" / f"{report_date}.json" if report_date else None,
        SITE / "data" / "top10" / "latest.json",
    ])
    trends_p = _first_existing([
        SITE / "data" / (report_date or "latest") / "trends.json" if report_date else None,
        SITE / "data" / "trends" / "latest.json",
    ])
    news_p = _first_existing([
        SITE / "data" / (report_date or "latest") / "news.json" if report_date else None,
        SITE / "data" / "news" / "latest.json",
    ])
    dii_p = _first_existing([
        SITE / "data" / (report_date or "latest") / "dii.json" if report_date else None,
        SITE / "data" / "dii" / "latest.json",
    ])

    top10 = _load_json(top10_p) or []
    trends = _load_json(trends_p) or {}
    news = _load_json(news_p) or {}
    dii = _load_json(dii_p) or {}

    items = _as_top10_list(top10)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(['html', 'xml']),
        trim_blocks=True, lstrip_blocks=True,
    )
    tmpl = env.get_template("daily.html.j2")

    outdir = SITE / "daily"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{report_date or 'latest'}.html"

    html = tmpl.render(
        date=report_date,
        top10=items,
        trends=trends,
        news=news,
        dii=dii,
    )
    outpath.write_text(html, encoding="utf-8")
    print(f"[OK] wrote: {outpath}")

if __name__ == "__main__":
    main()
