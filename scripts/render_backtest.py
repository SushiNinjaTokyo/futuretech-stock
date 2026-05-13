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
    data = read_json(OUT_DIR / "data" / "signals-v2" / "outcomes_latest.json")
    if data:
        return data
    return {"generated_at": None, "summary": {}, "recent": [], "strategy_comparison": [], "average_signal_path": [], "average_spy_path": [], "average_qqq_path": []}


def main() -> None:
    payload = load_payload()
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html", "xml"]), trim_blocks=True, lstrip_blocks=True)
    tpl = env.get_template("backtest.html.j2")
    html = tpl.render(payload=payload, summary=payload.get("summary", {}), recent=payload.get("recent", []), generated_at=payload.get("generated_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))
    write_text(OUT_DIR / "backtest" / "index.html", html)
    css_src = TEMPLATES_DIR / "backtest.css"
    if css_src.exists():
        ensure_dir(OUT_DIR / "assets")
        shutil.copy2(css_src, OUT_DIR / "assets" / "backtest.css")
    log("INFO", f"Rendered: {OUT_DIR / 'backtest' / 'index.html'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render_backtest: {e}")
        sys.exit(1)
