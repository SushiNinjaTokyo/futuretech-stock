#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, logging, shutil
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO"),
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("render")

ROOT   = Path(".")
OUT    = Path("site")
REPORT = os.getenv("REPORT_DATE") or "latest"

def resolve_template() -> tuple[Path, Environment]:
    cand_dirs = [Path("templates"), ROOT]
    tpl_name = "daily.html.j2"
    for d in cand_dirs:
        p = d / tpl_name
        if p.exists():
            env = Environment(
                loader=FileSystemLoader(str(d)),
                autoescape=select_autoescape(['html','xml']),
                enable_async=False, trim_blocks=True, lstrip_blocks=True,
            )
            log.info("[TPL] using file: %s", p)
            return p, env
    raise FileNotFoundError("'daily.html.j2' not found in search path: %s" % ", ".join(map(str,cand_dirs)))

def copy_assets():
    src_css = Path("templates/daily.css")
    dst_dir = OUT / "assets"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_css.exists():
        shutil.copy2(src_css, dst_dir / "daily.css")
        log.info("[ASSET] copied CSS -> %s", dst_dir / "daily.css")
    else:
        log.warning("[ASSET] templates/daily.css not found; page will be unstyled")

def load_top10(date: str) -> dict:
    p = OUT / "data" / date / "top10.json"
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")
    return json.load(open(p, "r", encoding="utf-8"))

def main():
    tpl_path, env = resolve_template()
    data = load_top10(REPORT)
    tpl  = env.get_template(tpl_path.name)

    schema_json = json.dumps({
        "@context": "https://schema.org",
        "@type": "ItemList",
        "name": f"Daily Top 10 — {data.get('date')}",
        "itemListElement": [
            {"@type":"ListItem","position":i+1,"name":it["symbol"],"url": f"/daily/{data.get('date')}.html#card-{i+1}"}
            for i, it in enumerate(data.get("top10",[]))
        ]
    }, ensure_ascii=False)

    html = tpl.render(date=data.get("date"), top10=data.get("top10", []), schema_json=schema_json)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"daily").mkdir(parents=True, exist_ok=True)
    outp = OUT/"daily"/f"{data.get('date')}.html"
    outp.write_text(html, encoding="utf-8")
    log.info("Rendered daily HTML: %s (template=%s)", outp, tpl_path)

    copy_assets()

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log.error("FATAL in render: %s", e, exc_info=True)
        sys.exit(1)
