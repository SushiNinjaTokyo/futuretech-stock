#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

OUT_DIR = os.environ.get("OUT_DIR", "site").strip()
REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()
TEMPLATES_DIR = "templates"

def load_top10_payload(date_str: str):
    # 優先: 日付付き、次点: latest.json
    by_date = os.path.join(OUT_DIR, "data", date_str, "top10.json")
    latest  = os.path.join(OUT_DIR, "data", "latest.json")
    path = by_date if os.path.exists(by_date) else latest
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_schema_org(date_str: str, items):
    # シンプルな JSON-LD（必要なら拡張してください）
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

    payload = load_top10_payload(REPORT_DATE)

    # ここが重要: dict 全体ではなく "items" リストをテンプレへ渡す
    items = payload.get("items", [])
    date_s = REPORT_DATE

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml", "j2"])
    )
    tpl = env.get_template("daily.html.j2")

    schema_json = make_schema_org(date_s, items)

    html = tpl.render(
        date=date_s,
        top10=items,             # ← テンプレは top10 を配列として期待
        items=items,             # 予備（どちらの書き方でも動くように）
        schema_json=json.dumps(schema_json, ensure_ascii=False)
    )

    out_dir = os.path.join(OUT_DIR, "daily")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_s}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # index.html（なければ作成/更新）
    index_path = os.path.join(out_dir, "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f'<meta http-equiv="refresh" content="0; url=./{date_s}.html">')

    print(f"Rendered daily HTML: {out_path}")

if __name__ == "__main__":
    main()
