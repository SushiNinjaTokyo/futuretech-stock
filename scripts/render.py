#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pathlib, logging, time, sys
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)sZ [%(levelname)s] %(message)s",
)
logging.Formatter.converter = time.gmtime

OUT_DIR = pathlib.Path(os.getenv("OUT_DIR", "site"))
DATA_DIR = OUT_DIR / "data"
REPORT_DATE = os.getenv("REPORT_DATE")
TPL_FILE = pathlib.Path("daily.html.j2")

def read_json(p: pathlib.Path, default=None):
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default

def main():
    if not REPORT_DATE:
        raise RuntimeError("REPORT_DATE missing")

    top10 = read_json(DATA_DIR / REPORT_DATE / "top10.json", default=[])
    date_str = REPORT_DATE

    # schema.org (簡易)
    schema = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": f"Daily Top 10 — {date_str}",
        "description": "AI・Robotics・Space stocks multi-signal ranking",
        "datePublished": date_str,
    }

    env = Environment(
        loader=FileSystemLoader("."),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template(str(TPL_FILE))

    html = tpl.render(
        date=date_str,
        top10=top10,
        schema_json=json.dumps(schema, ensure_ascii=False),
    )

    out = OUT_DIR / "daily" / f"{date_str}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(html)

    logging.info("Rendered daily HTML: %s", out)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("FATAL in render: %s", e)
        sys.exit(1)
