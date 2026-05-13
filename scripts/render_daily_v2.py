#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
TEMPLATES_DIR = ROOT / "templates"
REPORT_DATE = os.getenv("REPORT_DATE", "").strip()


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("ERROR", f"read_json failed: {path}: {e}")
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def load_payload() -> dict:
    if REPORT_DATE:
        p = OUT_DIR / "data" / "daily-v2" / REPORT_DATE / "top10.json"
        data = read_json(p)
        if data:
            return data
    data = read_json(OUT_DIR / "data" / "daily-v2" / "latest.json")
    if data:
        return data
    return {"date": REPORT_DATE or "N/A", "generated_at": None, "summary": {}, "items": [], "market": {}}


def main() -> None:
    payload = load_payload()
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html", "xml"]), trim_blocks=True, lstrip_blocks=True)
    tpl = env.get_template("daily.html.j2")
    html = tpl.render(payload=payload, date=payload.get("date", "N/A"), items=payload.get("items", []), market=payload.get("market", {}), summary=payload.get("summary", {}), generated_at=payload.get("generated_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))
    write_text(OUT_DIR / "daily" / "index.html", html)
    css_src = TEMPLATES_DIR / "daily.css"
    if css_src.exists():
        ensure_dir(OUT_DIR / "assets")
        shutil.copy2(css_src, OUT_DIR / "assets" / "daily.css")
    log("INFO", f"Rendered: {OUT_DIR / 'daily' / 'index.html'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render_daily_v2: {e}")
        sys.exit(1)
