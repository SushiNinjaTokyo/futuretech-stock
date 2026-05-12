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
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}",
        flush=True,
    )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing weekly simulation json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def main() -> None:
    data_path = OUT_DIR / "data" / "weekly" / "simulation" / "latest.json"
    payload = read_json(data_path)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template("weekly_simulation.html.j2")

    html = template.render(
        payload=payload,
        policy=ensure_dict(payload.get("policy")),
        summary=ensure_dict(payload.get("summary")),
        snapshots=ensure_list(payload.get("snapshots")),
        open_positions=ensure_list(payload.get("open_positions")),
        closed_trades=ensure_list(payload.get("closed_trades")),
        trade_log=ensure_list(payload.get("trade_log")),
        equity_curve=ensure_list(payload.get("equity_curve")),
        exit_reason_summary=ensure_list(payload.get("exit_reason_summary")),
        buy_type_summary=ensure_list(payload.get("buy_type_summary")),
        add_on_sequence_summary=ensure_list(payload.get("add_on_sequence_summary")),
        score_band_summary=ensure_list(payload.get("score_band_summary")),
        signal_type_summary=ensure_list(payload.get("signal_type_summary")),
        theme_summary=ensure_list(payload.get("theme_summary")),
        regime_summary=ensure_list(payload.get("regime_summary")),
        exposure_summary=ensure_dict(payload.get("exposure_summary")),
        liquidity_warnings=ensure_list(payload.get("liquidity_warnings")),
        strategy_comparison=ensure_list(payload.get("strategy_comparison")),
        strategy_highlights=ensure_dict(payload.get("strategy_highlights")),
        generated_at=payload.get("generated_at", "—"),
    )

    out_html = OUT_DIR / "weekly-simulation" / "index.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    asset_dir = OUT_DIR / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    css_src = TEMPLATE_DIR / "weekly_simulation.css"
    css_dst = asset_dir / "weekly_simulation.css"
    if not css_src.exists():
        raise FileNotFoundError(f"Missing weekly simulation css: {css_src}")
    shutil.copyfile(css_src, css_dst)

    log("INFO", f"Wrote {out_html}")
    log("INFO", f"Wrote {css_dst}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in render_weekly_simulation: {e}")
        sys.exit(1)
