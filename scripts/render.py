#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render daily overview HTML from Jinja2 template
- Input: site/data/<DATE>/top10.json
- Template: daily.html.j2 (repo root or scripts/.. と同階層)
- Output: site/daily/<DATE>.html ＋ site/daily/index.html（最新へのリンク用）
"""

import os, json, pathlib, datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

DATE = os.getenv("REPORT_DATE") or datetime.date.today().isoformat()
ROOT = pathlib.Path(__file__).resolve().parent.parent  # repo root を想定（scripts/配下からの相対）
if not (ROOT/"daily.html.j2").exists():
    ROOT = pathlib.Path(__file__).resolve().parent  # 念のため

def load_top10(root: pathlib.Path, date: str):
    p = root/"site"/"data"/date/"top10.json"
    return json.loads(p.read_text())

def ensure_dirs(root: pathlib.Path, date: str):
    (root/"site"/"daily").mkdir(parents=True, exist_ok=True)

def main():
    env = Environment(
        loader=FileSystemLoader(str(ROOT)),
        autoescape=select_autoescape(["html","xml"])
    )
    tmpl = env.get_template("daily.html.j2")

    data = load_top10(ROOT, DATE)
    ensure_dirs(ROOT, DATE)
    out = (ROOT/"site"/"daily"/f"{DATE}.html")
    out.write_text(tmpl.render(date=DATE, items=data), encoding="utf-8")

    # index.html は最新ファイルへの簡易リンク
    (ROOT/"site"/"daily"/"index.html").write_text(
        f"""<!doctype html><meta charset="utf-8">
<title>Daily Overview</title>
<p><a href="{DATE}.html">Open {DATE} overview</a></p>
""",
        encoding="utf-8"
    )
    print(f"Rendered daily HTML: {out}")

if __name__ == "__main__":
    main()
