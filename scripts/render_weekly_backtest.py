#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
TEMPLATE_DIR = ROOT / "templates"


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing weekly backtest json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    data_path = OUT_DIR / "data" / "weekly" / "backtest" / "latest.json"
    payload = read_json(data_path)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("weekly_backtest.html.j2")

    html = template.render(
        summary=payload.get("summary", {}),
        recent=payload.get("recent", []),
        snapshots=payload.get("snapshots", []),
        methodology=payload.get("methodology", {}),
        generated_at=payload.get("generated_at", "—"),
    )

    out_html = OUT_DIR / "weekly-backtest" / "index.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    asset_dir = OUT_DIR / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE_DIR / "weekly_backtest.css", asset_dir / "weekly_backtest.css")

    log("INFO", f"Wrote {out_html}")
    log("INFO", f"Wrote {asset_dir / 'weekly_backtest.css'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render_weekly_backtest: {e}")
        sys.exit(1)
