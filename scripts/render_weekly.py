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
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
REPORT_DATE = os.getenv("REPORT_DATE")


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("ERROR", f"read_json failed: {path}: {e}")
        return None


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def copy_asset(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def load_payload() -> dict:
    if REPORT_DATE:
        p = OUT_DIR / "data" / "weekly" / f"{REPORT_DATE}.json"
        data = read_json(p)
        if data:
            return data
    data = read_json(OUT_DIR / "data" / "weekly" / "latest.json")
    if data:
        return data
    return {"date": REPORT_DATE or "N/A", "generated_at": None, "methodology": {}, "summary": {}, "items": []}


def main() -> None:
    payload = load_payload()
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("weekly.html.j2")
    html = tpl.render(
        payload=payload,
        date=payload.get("date", "N/A"),
        items=payload.get("items", []),
        summary=payload.get("summary", {}),
        methodology=payload.get("methodology", {}),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    )
    out_html = OUT_DIR / "weekly" / "index.html"
    write_text(out_html, html)
    log("INFO", f"Rendered: {out_html}")

    css_src = TEMPLATES_DIR / "weekly.css"
    if css_src.exists():
        css_dst = OUT_DIR / "assets" / "weekly.css"
        copy_asset(css_src, css_dst)
        log("INFO", f"Copied CSS: {css_dst}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render_weekly: {e}")
        sys.exit(1)
