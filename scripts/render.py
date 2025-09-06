#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
REPORT_DATE = os.getenv("REPORT_DATE") or datetime.utcnow().date().isoformat()

TEMPLATE_PATH = Path(os.getenv("TPL_FILE", "templates/daily.html.j2"))
ASSET_CSS_SRC = Path(os.getenv("ASSET_CSS", "assets/daily.css"))

def load_payload(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"date": REPORT_DATE, "top10": [], "weights": {}}
    obj = json.loads(p.read_text() or "{}")
    if isinstance(obj, list):
        # 古い形式のとき（防御）
        return {"date": REPORT_DATE, "top10": obj, "weights": {}}
    if isinstance(obj, dict):
        # 欠損フィールドは初期化
        obj.setdefault("date", REPORT_DATE)
        obj.setdefault("top10", obj.get("items", []))
        obj.setdefault("weights", {})
        if not isinstance(obj["top10"], list):
            obj["top10"] = []
        return obj
    return {"date": REPORT_DATE, "top10": [], "weights": {}}

def copy_assets() -> str:
    # CSS を site/assets 配下に配置し、href を返す
    assets_dir = OUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    css_dst = assets_dir / "daily.css"
    if ASSET_CSS_SRC.exists():
        shutil.copyfile(ASSET_CSS_SRC, css_dst)
    else:
        # 万一 CSS がなければ、最低限のスタイルを生成
        css_dst.write_text("body{font-family:system-ui,Arial,sans-serif;margin:0;padding:24px;color:#111;background:#fafafa}")
    return str(Path("assets") / "daily.css")

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "daily").mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} [INFO] [TPL] using file: {TEMPLATE_PATH}")
    tpl = env.get_template(TEMPLATE_PATH.name)

    top10_json = OUT_DIR / "data" / REPORT_DATE / "top10.json"
    data = load_payload(top10_json)

    css_href = copy_assets()

    html = tpl.render(
        name=f"Daily Top 10 — {data.get('date', REPORT_DATE)}",
        date=data.get("date", REPORT_DATE),
        weights=data.get("weights") or {},
        items=data.get("top10") or [],
        css_href=css_href,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%MZ"),
    )

    out_file = OUT_DIR / "daily" / f"{data.get('date', REPORT_DATE)}.html"
    out_file.write_text(html)
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} [INFO] Rendered daily HTML: {out_file} (template={TEMPLATE_PATH})")
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} [INFO] [ASSET] copied CSS -> {OUT_DIR / 'assets' / 'daily.css'}")

if __name__ == "__main__":
    main()
