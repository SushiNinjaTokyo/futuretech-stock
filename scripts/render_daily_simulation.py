#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
if not OUT_DIR.is_absolute():
    OUT_DIR = (ROOT / OUT_DIR).resolve()
else:
    OUT_DIR = OUT_DIR.resolve()
TEMPLATES_DIR = ROOT / "templates"
SIM_JSON = OUT_DIR / "data" / "daily" / "simulation" / "latest.json"


def log(level: str, msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing daily simulation json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def ensure_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    payload["summary"] = ensure_dict(payload.get("summary"))
    for k in ["equity_curve", "closed_trades", "open_positions", "exit_reason_summary", "triage_summary", "archetype_summary", "theme_summary", "strategy_comparison", "strategy_results"]:
        payload[k] = ensure_list(payload.get(k))
    payload["policy"] = ensure_dict(payload.get("policy"))
    payload["highlights"] = ensure_dict(payload.get("highlights"))

    # Add chart extents to avoid heavy template logic.
    curve = payload["equity_curve"]
    values: List[float] = []
    for row in curve:
        if not isinstance(row, dict):
            continue
        for key in ["portfolio_equity", "benchmark_equity", "secondary_benchmark_equity", "external_capital"]:
            v = to_float(row.get(key))
            if v is not None:
                values.append(v)
    payload["chart_min"] = min(values) if values else 0
    payload["chart_max"] = max(values) if values else 1
    return payload


def copy_asset(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def main() -> None:
    payload = normalize(read_json(SIM_JSON))
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("daily_simulation.html.j2")
    html = tpl.render(payload=payload, generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))
    out = OUT_DIR / "daily-simulation" / "index.html"
    ensure_dir(out.parent)
    out.write_text(html, encoding="utf-8")
    log("INFO", f"Rendered: {out}")
    css = TEMPLATES_DIR / "daily_simulation.css"
    if css.exists():
        copy_asset(css, OUT_DIR / "assets" / "daily_simulation.css")
        log("INFO", f"Copied CSS: {OUT_DIR / 'assets' / 'daily_simulation.css'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR", f"FATAL in render_daily_simulation: {exc}")
        sys.exit(1)
