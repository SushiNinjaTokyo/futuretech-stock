#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Any, Dict, List
from jinja2 import Environment, FileSystemLoader, select_autoescape

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()
TEMPLATES_DIR = "templates"


def load_top10_payload(date_str: str) -> Dict[str, Any]:
    """優先: 日付付き、次点: latest.json"""
    by_date = os.path.join(OUT_DIR, "data", date_str, "top10.json")
    latest = os.path.join(OUT_DIR, "data", "latest.json")
    path = by_date if os.path.exists(by_date) else latest
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_comp(components: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for c in components or []:
        if str(c.get("name", "")).strip().lower() == name.lower():
            return c
    return {}


def _norm_item_for_template(it: Dict[str, Any]) -> Dict[str, Any]:
    """
    fetch_daily.py の出力（rank_points / returns / components[]）を
    テンプレート daily.html.j2 が想定している構造へ変換する。
    """
    symbol = it.get("symbol") or ""
    name = it.get("name") or symbol

    # 1) スコア（テンプレは score_pts を使う）
    score_pts = int(it.get("rank_points") or it.get("score_pts") or 0)

    # 2) リターン（テンプレは deltas.d1/d5/d20 を使う）
    r = it.get("returns") or {}
    deltas = {
        "d1":  float(r.get("1D") or 0.0),
        "d5":  float(r.get("1W") or 0.0),
        "d20": float(r.get("1M") or 0.0),
    }

    # 3) components[] から各要素を抽出
    comps = it.get("components") or []

    c_news   = _find_comp(comps, "News coverage")
    c_trend  = _find_comp(comps, "Trend breakout")
    c_vol    = _find_comp(comps, "Volume anomaly")
    c_dii    = _find_comp(comps, "DII model")

    # points は 0..1000 前提、0..1 へ正規化
    def _pts01(c): return max(0.0, min(1.0, float(c.get("points", 0)) / 1000.0))

    news01  = _pts01(c_news)
    trend01 = _pts01(c_trend)
    vol01   = _pts01(c_vol)
    dii01   = _pts01(c_dii)

    # 4) Breakdown 用データ（テンプレの JS がこれを読んで再計算する）
    #    キー名はテンプレ内 label() の対応に合わせる
    score_components = {
        "volume_anomaly": vol01,
        "trends_breakout": trend01,
        "news": news01,
        "dii": dii01,
    }
    # weight は 0..1 で
    def _w(c): 
        w = c.get("weight")
        try:
            return max(0.0, float(w) / (100.0 if float(w) > 1.0 else 1.0))
        except Exception:
            return 0.0

    score_weights = {
        "volume_anomaly": _w(c_vol),
        "trends_breakout": _w(c_trend),
        "news": _w(c_news),
        "dii": _w(c_dii),
    }

    # 5) テンプレが参照するその他フィールド（無いものは無害な既定値）
    #    vol の詳細は現状データに無いので空でOK（テンプレは「No details.」表示）
    detail = {}

    normalized = {
        "symbol": symbol,
        "name": name,
        "score_pts": score_pts,
        "deltas": deltas,
        "detail": detail,

        # <script class="js-payload"> で参照されるフィールド群
        "score_components": score_components,
        "score_weights": score_weights,
        "final_score_0_1": max(0.0, min(1.0, score_pts / 1000.0)),
        "trends_breakout": trend01,
        "vol_anomaly_score": vol01,
        "news_score": news01,
        "news_recent_count": int(it.get("news_recent_count") or 0),  # 無ければ 0
        "dii_score": dii01,
        "dii_components": it.get("dii_components") or {},
    }
    return normalized


def normalize_items_for_template(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_norm_item_for_template(dict(it)) for it in (items or [])]


def make_schema_org(date_str: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "name": f"Daily Top 10 — {date_str}",
        "dateCreated": date_str,
        "numberOfItems": len(items),
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": i + 1,
                "item": {
                    "@type": "Thing",
                    "name": it.get("name") or it.get("symbol"),
                    "identifier": it.get("symbol"),
                },
            }
            for i, it in enumerate(items)
        ],
    }


def main():
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is empty.")

    raw = load_top10_payload(REPORT_DATE)
    raw_items = raw.get("items", [])

    # ★ ここでテンプレ適合フォーマットへ正規化 ★
    items = normalize_items_for_template(raw_items)

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        enable_async=False,
    )
    tpl = env.get_template("daily.html.j2")

    schema_json = make_schema_org(REPORT_DATE, items)

    html = tpl.render(
        date=REPORT_DATE,
        top10=items,               # テンプレは top10（配列）を使う
        items=items,               # 互換用（使ってなくても可）
        schema_json=json.dumps(schema_json, ensure_ascii=False),
    )

    out_dir = os.path.join(OUT_DIR, "daily")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{REPORT_DATE}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # index.html を作成（初回のみ）
    index_path = os.path.join(out_dir, "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f'<meta http-equiv="refresh" content="0; url=./{REPORT_DATE}.html">')

    print(f"Rendered daily HTML: {out_path}")


if __name__ == "__main__":
    main()
