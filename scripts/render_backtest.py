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


def ensure_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def ensure_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def normalize_strategy_comparison(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in ensure_list(items):
        if not isinstance(raw, dict):
            continue
        avg_ret = raw.get("avg_return_pct")
        # Backward-compatible aliases, in case an older JSON used different names.
        if avg_ret is None:
            avg_ret = raw.get("net_return_pct")
        if avg_ret is None:
            avg_ret = raw.get("return_pct")
        max_dd = raw.get("max_drawdown_pct")
        if max_dd is None:
            max_dd = raw.get("max_dd_pct")
        out.append({
            "name": raw.get("name") or raw.get("label") or "Strategy",
            "trades": raw.get("trades") if raw.get("trades") is not None else raw.get("closed_trades", 0),
            "avg_return_pct": avg_ret,
            "max_drawdown_pct": max_dd,
            "win_rate": raw.get("win_rate"),
            "return_drawdown_ratio": raw.get("return_drawdown_ratio"),
        })
    return out


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload or {})
    p["summary"] = ensure_dict(p.get("summary"))
    p["recent"] = ensure_list(p.get("recent"))
    p["strategy_comparison"] = normalize_strategy_comparison(p.get("strategy_comparison"))
    p["average_signal_path"] = ensure_list(p.get("average_signal_path"))
    p["average_spy_path"] = ensure_list(p.get("average_spy_path"))
    p["average_qqq_path"] = ensure_list(p.get("average_qqq_path"))
    p["score_volume_heatmap"] = ensure_list(p.get("score_volume_heatmap"))
    p["peak_distribution"] = ensure_list(p.get("peak_distribution"))
    p["profile_buckets"] = ensure_list(p.get("profile_buckets"))
    p["rule_buckets"] = ensure_list(p.get("rule_buckets"))
    p["regime_buckets"] = ensure_list(p.get("regime_buckets"))
    p["loser_diagnostics"] = ensure_list(p.get("loser_diagnostics"))
    p["range"] = ensure_dict(p.get("range"))
    p["methodology"] = ensure_dict(p.get("methodology"))
    return p


def load_payload() -> Dict[str, Any]:
    data = read_json(OUT_DIR / "data" / "signals-v2" / "outcomes_latest.json")
    if isinstance(data, dict):
        return normalize_payload(data)
    return normalize_payload({
        "generated_at": None,
        "summary": {},
        "recent": [],
        "strategy_comparison": [],
        "average_signal_path": [],
        "average_spy_path": [],
        "average_qqq_path": [],
    })


def main() -> None:
    payload = load_payload()
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("backtest.html.j2")
    html = tpl.render(
        payload=payload,
        summary=payload.get("summary", {}),
        recent=payload.get("recent", []),
        generated_at=payload.get("generated_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    )
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
