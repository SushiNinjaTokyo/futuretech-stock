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
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


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


def ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def default_summary() -> Dict[str, Any]:
    return {
        "as_of": None, "generated_at": None, "tracking_policy": {}, "total_signals": 0, "active_signals": 0,
        "primary_benchmark": "SPY", "secondary_benchmark": "QQQ",
        "avg_return_1d": None, "avg_return_3d": None, "avg_return_5d": None, "avg_return_10d": None, "avg_return_20d": None,
        "avg_current_return": None, "avg_mfe": None, "avg_mae": None, "mfe_mae_ratio": None,
        "follow_through_rate": None, "stop_hit_5pct_rate": None,
        "rank_buckets": [], "score_buckets": [], "volume_buckets": [], "rule_buckets": [], "profiles": [], "archetypes": [], "regime_buckets": [],
        "quality_matrix": [], "signal_path": [], "peak_distribution": [], "strategy_comparison": [], "diagnostics": {},
    }


def load_summary() -> Dict[str, Any]:
    base = default_summary()
    j = read_json(OUT_DIR / "data" / "signals" / "summary_latest.json")
    if isinstance(j, dict):
        base.update(j)
    for key in ["rank_buckets", "score_buckets", "volume_buckets", "rule_buckets", "profiles", "archetypes", "regime_buckets", "quality_matrix", "signal_path", "peak_distribution", "strategy_comparison"]:
        base[key] = ensure_list(base.get(key))
    base["diagnostics"] = ensure_dict(base.get("diagnostics"))
    base["tracking_policy"] = ensure_dict(base.get("tracking_policy"))
    return base


def load_recent_outcomes() -> List[Dict[str, Any]]:
    j = read_json(OUT_DIR / "data" / "signals" / "outcomes_latest.json")
    if isinstance(j, dict):
        return [x for x in ensure_list(j.get("items")) if isinstance(x, dict)][:300]
    if isinstance(j, list):
        return [x for x in j if isinstance(x, dict)][:300]
    return []


def render() -> None:
    summary = load_summary()
    recent = load_recent_outcomes()
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html", "xml"]), trim_blocks=True, lstrip_blocks=True)
    tpl = env.get_template("backtest.html.j2")
    html = tpl.render(summary=summary, recent=recent, generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))
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
