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
if not OUT_DIR.is_absolute():
    OUT_DIR = (ROOT / OUT_DIR).resolve()
else:
    OUT_DIR = OUT_DIR.resolve()

REPORT_DATE = os.getenv("REPORT_DATE")


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("ERROR", f"read_json failed: {safe_relative(path)}: {e}")
        return None


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def copy_asset(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def load_payload() -> dict[str, Any]:
    if REPORT_DATE:
        p = OUT_DIR / "data" / "weekly" / f"{REPORT_DATE}.json"
        data = read_json(p)
        if isinstance(data, dict):
            return data

    latest = OUT_DIR / "data" / "weekly" / "latest.json"
    data = read_json(latest)
    if isinstance(data, dict):
        return data

    return {
        "date": REPORT_DATE or "N/A",
        "generated_at": None,
        "methodology": {},
        "summary": {},
        "items": [],
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})

    items = payload.get("items")
    if not isinstance(items, list):
        items = []

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    methodology = payload.get("methodology")
    if not isinstance(methodology, dict):
        methodology = {}

    weights = methodology.get("weights")
    if not isinstance(weights, dict):
        weights = {}

    # Build-weekly-screening output schema.
    summary_defaults = {
        "total_candidates": 0,
        "valid_items": len(items),
        "fresh_breakouts": 0,
        "leaders": 0,
        "constructive_setups": 0,
        "early_watch": 0,
        "extended": 0,
        "avoid": 0,
    }

    for item in items:
        sig = item.get("signal")
        if sig == "C Early Watch":
            summary_defaults["early_watch"] += 1
        elif sig == "D Extended":
            summary_defaults["extended"] += 1
        elif sig == "E Avoid":
            summary_defaults["avoid"] += 1

    for k, v in summary_defaults.items():
        summary.setdefault(k, v)

    methodology.setdefault("name", "Minervini-inspired Weekly Screening")
    methodology.setdefault("total_points", 1000)
    methodology.setdefault("weights", weights)
    methodology.setdefault("benchmark", "SPY")
    methodology.setdefault("fundamentals_enabled", False)

    payload["items"] = items
    payload["summary"] = summary
    payload["methodology"] = methodology
    payload.setdefault("date", REPORT_DATE or "N/A")
    payload.setdefault("generated_at", None)

    return payload


def main() -> None:
    payload = normalize_payload(load_payload())

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
    log("INFO", f"Rendered: {safe_relative(out_html)}")

    css_src = TEMPLATES_DIR / "weekly.css"
    if css_src.exists():
        css_dst = OUT_DIR / "assets" / "weekly.css"
        copy_asset(css_src, css_dst)
        log("INFO", f"Copied CSS: {safe_relative(css_dst)}")
    else:
        log("WARN", "templates/weekly.css not found. Skip CSS copy.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render_weekly: {e}")
        sys.exit(1)
