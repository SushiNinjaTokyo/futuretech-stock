#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))


def log(level: str, msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}",
        flush=True,
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def copy_asset(src: Path, dst: Path) -> None:
    if not src.exists():
        log("WARN", f"asset missing: {src}")
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def default_summary() -> Dict[str, Any]:
    return {
        "as_of": None,
        "total_signals": 0,
        "completed_5d": 0,
        "completed_10d": 0,
        "completed_20d": 0,
        "win_rate_5d": None,
        "win_rate_10d": None,
        "win_rate_20d": None,
        "avg_return_5d": None,
        "avg_return_10d": None,
        "avg_return_20d": None,
        "median_return_20d": None,
        "avg_max_gain_20d": None,
        "avg_max_drawdown_20d": None,
        "rank_buckets": [],
        "profiles": [],
    }


def load_summary() -> Dict[str, Any]:
    j = read_json(OUT_DIR / "data" / "signals" / "summary_latest.json")
    if isinstance(j, dict):
        base = default_summary()
        base.update(j)
        return base
    return default_summary()


def load_recent_outcomes() -> List[Dict[str, Any]]:
    j = read_json(OUT_DIR / "data" / "signals" / "outcomes_latest.json")
    if isinstance(j, dict):
        items = j.get("items", [])
        if isinstance(items, list):
            return items[:50]
    if isinstance(j, list):
        return j[:50]
    return []


def render() -> None:
    summary = load_summary()
    recent = load_recent_outcomes()

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    tpl = env.get_template("backtest.html.j2")
    html = tpl.render(
        summary=summary,
        recent=recent,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    )

    out_html = OUT_DIR / "backtest" / "index.html"
    write_text(out_html, html)
    log("INFO", f"Rendered backtest: {out_html}")

    copy_asset(TEMPLATES_DIR / "backtest.css", OUT_DIR / "assets" / "backtest.css")


if __name__ == "__main__":
    try:
        render()
    except Exception as e:
        log("ERROR", f"FATAL in render_backtest: {e}")
        sys.exit(1)