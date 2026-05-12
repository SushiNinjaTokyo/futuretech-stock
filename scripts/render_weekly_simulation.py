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


def ensure_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        v = float(value)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def first_number(*values: Any) -> Optional[float]:
    for value in values:
        v = to_float(value)
        if v is not None:
            return v
    return None


def set_if_missing(d: Dict[str, Any], key: str, value: Any) -> None:
    if d.get(key) is None and value is not None:
        d[key] = value


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize older weekly-simulation JSON schemas so render_only can safely
    re-render after template changes. This prevents template errors such as
    missing portfolio_equity on older equity_curve rows.
    """
    payload = dict(payload or {})
    summary = ensure_dict(payload.get("summary"))

    external = first_number(
        summary.get("total_new_capital"),
        summary.get("external_capital"),
        summary.get("new_capital"),
        summary.get("total_base_cost"),
    )
    portfolio = first_number(
        summary.get("portfolio_equity"),
        summary.get("current_equity"),
        summary.get("current_value"),
        summary.get("gross_current_value"),
    )
    spy = first_number(
        summary.get("spy_equity"),
        summary.get("spy_equivalent"),
        summary.get("benchmark_equity"),
    )
    cash = first_number(summary.get("cash"), 0.0)
    market = first_number(
        summary.get("market_value"),
        summary.get("open_market_value"),
        summary.get("current_market"),
        (portfolio - cash) if portfolio is not None and cash is not None else None,
    )

    set_if_missing(summary, "total_new_capital", external)
    set_if_missing(summary, "portfolio_equity", portfolio)
    set_if_missing(summary, "spy_equity", spy)
    set_if_missing(summary, "cash", cash)
    set_if_missing(summary, "market_value", market)

    if portfolio is not None and external not in (None, 0):
        set_if_missing(summary, "net_pl_value", portfolio - external)
        set_if_missing(summary, "net_return_pct", (portfolio / external - 1.0) * 100.0)
    if spy is not None and external not in (None, 0):
        set_if_missing(summary, "spy_return_pct", (spy / external - 1.0) * 100.0)
    if portfolio is not None and spy is not None:
        set_if_missing(summary, "alpha_value", portfolio - spy)
    if summary.get("net_return_pct") is not None and summary.get("spy_return_pct") is not None:
        set_if_missing(summary, "alpha_pct", float(summary["net_return_pct"]) - float(summary["spy_return_pct"]))

    if summary.get("max_drawdown_pct") not in (None, 0):
        try:
            ratio = abs(float(summary.get("net_return_pct", 0.0)) / float(summary["max_drawdown_pct"]))
            set_if_missing(summary, "return_drawdown_ratio", round(ratio, 2))
        except Exception:
            pass

    if summary.get("current_exposure_pct") is None and portfolio not in (None, 0) and market is not None:
        summary["current_exposure_pct"] = market / portfolio * 100.0

    summary.setdefault("benchmark_symbol", payload.get("benchmark_symbol") or "SPY")
    payload["summary"] = summary

    curve = []
    for raw in ensure_list(payload.get("equity_curve")):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row_external = first_number(
            row.get("external_capital"),
            row.get("total_new_capital"),
            row.get("new_capital"),
            external,
        )
        row_cash = first_number(row.get("cash"), 0.0)
        row_portfolio = first_number(
            row.get("portfolio_equity"),
            row.get("current_equity"),
            row.get("equity"),
            row.get("portfolio_value"),
        )
        row_spy = first_number(row.get("spy_equity"), row.get("spy_equivalent"), row.get("benchmark_equity"))
        row_market = first_number(
            row.get("market_value"),
            row.get("open_market_value"),
            (row_portfolio - row_cash) if row_portfolio is not None and row_cash is not None else None,
        )

        row["external_capital"] = row_external
        row["cash"] = row_cash
        row["portfolio_equity"] = row_portfolio
        row["spy_equity"] = row_spy
        row["market_value"] = row_market
        if row_portfolio is not None and row_spy is not None:
            row["alpha_value"] = first_number(row.get("alpha_value"), row_portfolio - row_spy)
        else:
            row["alpha_value"] = first_number(row.get("alpha_value"))
        row["drawdown_pct"] = first_number(row.get("drawdown_pct"), 0.0)
        if row.get("exposure_pct") is None and row_portfolio not in (None, 0) and row_market is not None:
            row["exposure_pct"] = row_market / row_portfolio * 100.0
        else:
            row["exposure_pct"] = first_number(row.get("exposure_pct"), 0.0)
        curve.append(row)

    # Force the final curve point to align with the KPI summary when possible.
    if portfolio is not None or spy is not None or external is not None:
        final = dict(curve[-1]) if curve else {}
        final["date"] = final.get("date") or summary.get("as_of") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        final["point_type"] = "current"
        final["external_capital"] = external
        final["portfolio_equity"] = portfolio
        final["spy_equity"] = spy
        final["cash"] = cash
        final["market_value"] = market
        final["alpha_value"] = (portfolio - spy) if portfolio is not None and spy is not None else summary.get("alpha_value")
        final["drawdown_pct"] = first_number(summary.get("current_drawdown_pct"), final.get("drawdown_pct"), 0.0)
        final["exposure_pct"] = first_number(summary.get("current_exposure_pct"), final.get("exposure_pct"), 0.0)
        if curve:
            curve[-1] = final
        else:
            curve.append(final)

    payload["equity_curve"] = curve
    return payload


def main() -> None:
    data_path = OUT_DIR / "data" / "weekly" / "simulation" / "latest.json"
    payload = normalize_payload(read_json(data_path))

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
        best_trade=ensure_dict(payload.get("best_trade")),
        worst_trade=ensure_dict(payload.get("worst_trade")),
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
