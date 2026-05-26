from __future__ import annotations

"""Render AI Arena Positions / Portfolio History page.

The page is intentionally render-only. It does not change simulation, execution,
ranking, cash accounting, or trade logic. It reads the current positions JSON and
optionally enriches it with closed trade history from simulation/latest.json.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from render_common import OUT_DIR, copy_asset, env, read_json, write_text

POSITIONS_JSON = Path(
    os.getenv(
        "AI_ARENA_POSITIONS_JSON",
        str(OUT_DIR / "data/japan/ai-arena/positions/latest.json"),
    )
)
SIMULATION_JSON = Path(
    os.getenv(
        "AI_ARENA_SIMULATION_JSON",
        str(OUT_DIR / "data/japan/ai-arena/simulation/latest.json"),
    )
)

RECENT_CLOSED_LIMIT = int(os.getenv("AI_ARENA_POSITIONS_RECENT_CLOSED_LIMIT", "10"))
MAX_HISTORY_TOTAL = int(os.getenv("AI_ARENA_POSITIONS_MAX_HISTORY_TOTAL", "80"))

STYLE_LABELS = {
    "daily_striker": "Momentum Strike",
    "weekly_sage": "Trend Core",
    "risk_sentinel": "Capital Shield",
    "discovery_scout": "Hidden Alpha",
    "contrarian_monk": "Reversal Hunt",
}

PROFILE_LABELS = {
    "daily_v2_core": "Momentum Strike",
    "daily_stage1": "Opening Strike",
    "daily_stage2": "Momentum Chase",
    "weekly_stage1": "Trend Watch",
    "weekly_stage2": "Trend Core",
    "weekly_core": "Trend Core",
    "risk_defender": "Capital Shield",
    "risk_stage1": "Risk Guard",
    "risk_stage2": "Capital Shield",
    "discovery_alpha": "Hidden Alpha",
    "discovery_stage1": "Discovery Watch",
    "discovery_stage2": "Early Breakout",
    "contrarian_reentry": "Reversal Hunt",
    "contrarian_stage1": "Pullback Watch",
    "contrarian_stage2": "Reversal Setup",
}

EXIT_REASON_LABELS = {
    "stop_loss": "Risk exit",
    "max_holding": "Time exit",
    "take_profit": "Profit take",
    "signal_exit": "Signal exit",
    "risk_off": "Risk-off exit",
}


def num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def int_num(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def fmt_jpy(value: Any) -> str:
    return f"¥{num(value):,.0f}"


def fmt_signed_jpy(value: Any) -> str:
    v = num(value)
    sign = "+" if v > 0 else ""
    return f"{sign}¥{v:,.0f}"


def fmt_pct(value: Any) -> str:
    v = num(value)
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def pnl_class(value: Any) -> str:
    v = num(value)
    if v > 0:
        return "pos"
    if v < 0:
        return "neg"
    return "flat"


def short_date(value: Any) -> str:
    s = str(value or "")
    if not s:
        return "—"
    try:
        return datetime.fromisoformat(s[:10]).strftime("%Y-%m-%d")
    except Exception:
        return s[:10]


def clean_reason(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return "Signal-based entry."
    for raw, label in sorted(PROFILE_LABELS.items(), key=lambda x: -len(x[0])):
        s = re.sub(rf"\b{re.escape(raw)}\b", label, s)
    s = re.sub(r"\bscore\s+([0-9]+(?:\.[0-9]+)?)", r"score \1", s, flags=re.I)
    s = s.replace("no daily confirmation", "without daily confirmation")
    s = s.replace("penalty drag", "risk penalty")
    return s[:220]


def clean_exit_reason(text: Any) -> str:
    s = str(text or "").strip()
    return EXIT_REASON_LABELS.get(s, clean_reason(s) if s else "Closed")


def trade_sort_key(trade: dict[str, Any]) -> str:
    return str(trade.get("exit_date") or trade.get("entry_date") or "")


def enrich_open_position(p: dict[str, Any]) -> dict[str, Any]:
    out = dict(p)
    out["entry_date_fmt"] = short_date(out.get("entry_date"))
    out["current_date_fmt"] = short_date(out.get("current_date"))
    out["entry_price_fmt"] = fmt_jpy(out.get("entry_price"))
    out["current_price_fmt"] = fmt_jpy(out.get("current_price"))
    out["market_value_fmt"] = fmt_jpy(out.get("market_value_jpy"))
    out["unrealized_pnl_fmt"] = fmt_signed_jpy(out.get("unrealized_pnl_jpy"))
    out["unrealized_return_fmt"] = fmt_pct(out.get("unrealized_return_pct"))
    out["pnl_class"] = pnl_class(out.get("unrealized_pnl_jpy"))
    out["return_bar_width"] = min(100, max(4, abs(num(out.get("unrealized_return_pct"))) * 4.5))
    out["entry_reason_clean"] = clean_reason(out.get("entry_reason"))
    out["holding_days"] = int_num(out.get("holding_days"))
    out["shares_fmt"] = f"{int_num(out.get('shares')):,}"
    return out


def enrich_closed_trade(t: dict[str, Any]) -> dict[str, Any]:
    out = dict(t)
    out["entry_date_fmt"] = short_date(out.get("entry_date"))
    out["exit_date_fmt"] = short_date(out.get("exit_date"))
    out["entry_price_fmt"] = fmt_jpy(out.get("entry_price"))
    out["exit_price_fmt"] = fmt_jpy(out.get("exit_price"))
    out["pnl_fmt"] = fmt_signed_jpy(out.get("pnl_jpy"))
    out["return_fmt"] = fmt_pct(out.get("return_pct"))
    out["pnl_class"] = pnl_class(out.get("pnl_jpy"))
    out["exit_reason_clean"] = clean_exit_reason(out.get("exit_reason"))
    out["entry_reason_clean"] = clean_reason(out.get("entry_reason"))
    out["holding_days"] = int_num(out.get("holding_days"))
    out["shares_fmt"] = f"{int_num(out.get('shares')):,}"
    return out


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def agent_style_label(agent: dict[str, Any]) -> str:
    aid = str(agent.get("agent_id") or "")
    profile = str((agent.get("summary") or {}).get("screening_profile") or "")
    return STYLE_LABELS.get(aid) or PROFILE_LABELS.get(profile) or str(agent.get("class") or "AI Agent")


def enrich_agent(agent: dict[str, Any], sim_agent: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(agent)
    if sim_agent:
        # Keep current positions payload as the base, but use simulation for full history.
        out.setdefault("personality", sim_agent.get("personality"))
        out.setdefault("philosophy", sim_agent.get("philosophy"))
        out["closed_trades"] = list(sim_agent.get("closed_trades") or [])
        if not out.get("open_positions"):
            out["open_positions"] = list(sim_agent.get("open_positions") or [])
        out["equity_curve"] = list(sim_agent.get("equity_curve") or [])
    else:
        out["closed_trades"] = list(out.get("closed_trades") or [])
        out["equity_curve"] = list(out.get("equity_curve") or [])

    summary = dict(out.get("summary") or {})
    out["style_label"] = agent_style_label(out)
    out["return_fmt"] = fmt_pct(summary.get("return_pct"))
    out["equity_fmt"] = fmt_jpy(summary.get("portfolio_equity_jpy"))
    out["cash_fmt"] = fmt_jpy(summary.get("cash_jpy"))
    out["market_value_fmt"] = fmt_jpy(summary.get("market_value_jpy"))
    out["return_class"] = pnl_class(summary.get("return_pct"))

    opens = [enrich_open_position(p) for p in (out.get("open_positions") or [])]
    out["open_positions"] = sorted(opens, key=lambda p: abs(num(p.get("market_value_jpy"))), reverse=True)
    closed = [enrich_closed_trade(t) for t in (out.get("closed_trades") or [])]
    closed = sorted(closed, key=trade_sort_key, reverse=True)
    out["closed_trades"] = closed
    out["recent_closed_trades"] = closed[:RECENT_CLOSED_LIMIT]

    open_pnl = sum(num(p.get("unrealized_pnl_jpy")) for p in opens)
    open_value = sum(num(p.get("market_value_jpy")) for p in opens)
    realized_pnl = sum(num(t.get("pnl_jpy")) for t in closed)
    wins = [t for t in closed if num(t.get("pnl_jpy")) > 0]
    losses = [t for t in closed if num(t.get("pnl_jpy")) < 0]
    returns = [num(t.get("return_pct")) for t in closed]
    holding = [num(t.get("holding_days")) for t in closed]

    out["position_summary"] = {
        "open_count": len(opens),
        "closed_count": len(closed),
        "open_market_value_jpy": open_value,
        "open_market_value_fmt": fmt_jpy(open_value),
        "unrealized_pnl_jpy": open_pnl,
        "unrealized_pnl_fmt": fmt_signed_jpy(open_pnl),
        "unrealized_class": pnl_class(open_pnl),
        "realized_pnl_jpy": realized_pnl,
        "realized_pnl_fmt": fmt_signed_jpy(realized_pnl),
        "realized_class": pnl_class(realized_pnl),
        "win_rate_fmt": fmt_pct((len(wins) / len(closed) * 100) if closed else summary.get("win_rate_pct", 0)),
        "avg_return_fmt": fmt_pct(avg(returns)),
        "avg_holding_fmt": f"{avg(holding):.1f}d" if holding else "—",
    }
    out["largest_open"] = max(opens, key=lambda p: num(p.get("market_value_jpy")), default=None)
    out["best_open"] = max(opens, key=lambda p: num(p.get("unrealized_pnl_jpy")), default=None)
    out["worst_open"] = min(opens, key=lambda p: num(p.get("unrealized_pnl_jpy")), default=None)
    out["best_closed"] = max(closed, key=lambda t: num(t.get("pnl_jpy")), default=None)
    out["worst_closed"] = min(closed, key=lambda t: num(t.get("pnl_jpy")), default=None)
    return out


def build_payload(positions: dict[str, Any], simulation: dict[str, Any]) -> dict[str, Any]:
    base_agents = list(positions.get("agents") or [])
    sim_by_id = {a.get("agent_id"): a for a in simulation.get("agents") or []}
    if not base_agents:
        base_agents = list(simulation.get("agents") or [])

    agents = [enrich_agent(a, sim_by_id.get(a.get("agent_id"))) for a in base_agents]
    all_closed: list[dict[str, Any]] = []
    for a in agents:
        for t in a.get("closed_trades") or []:
            tt = dict(t)
            tt["agent_name"] = a.get("name")
            tt["agent_style_label"] = a.get("style_label")
            tt["ui_tone"] = a.get("ui_tone")
            all_closed.append(tt)
    all_closed = sorted(all_closed, key=trade_sort_key, reverse=True)

    all_open = [p for a in agents for p in (a.get("open_positions") or [])]
    total_open_value = sum(num(p.get("market_value_jpy")) for p in all_open)
    total_unrealized = sum(num(p.get("unrealized_pnl_jpy")) for p in all_open)
    total_realized = sum(num(t.get("pnl_jpy")) for t in all_closed)
    wins = [t for t in all_closed if num(t.get("pnl_jpy")) > 0]

    out = dict(positions or {})
    if not out.get("range"):
        out["range"] = simulation.get("range")
    if not out.get("season"):
        out["season"] = simulation.get("season")
    if not out.get("generated_at"):
        out["generated_at"] = simulation.get("generated_at")
    out["agents"] = agents
    out["history"] = {
        "recent_closed_trades": all_closed[:MAX_HISTORY_TOTAL],
        "closed_count": len(all_closed),
        "open_count": len(all_open),
        "total_open_value_fmt": fmt_jpy(total_open_value),
        "total_unrealized_fmt": fmt_signed_jpy(total_unrealized),
        "total_unrealized_class": pnl_class(total_unrealized),
        "total_realized_fmt": fmt_signed_jpy(total_realized),
        "total_realized_class": pnl_class(total_realized),
        "win_rate_fmt": fmt_pct((len(wins) / len(all_closed) * 100) if all_closed else 0),
        "best_closed": max(all_closed, key=lambda t: num(t.get("pnl_jpy")), default=None),
        "worst_closed": min(all_closed, key=lambda t: num(t.get("pnl_jpy")), default=None),
    }
    out["render_note"] = {
        "positions_json": str(POSITIONS_JSON),
        "simulation_json_loaded": bool(simulation),
        "recent_closed_limit_per_agent": RECENT_CLOSED_LIMIT,
        "max_history_total": MAX_HISTORY_TOTAL,
    }
    return out


def main() -> None:
    positions = read_json(POSITIONS_JSON, {})
    simulation = read_json(SIMULATION_JSON, {})
    payload = build_payload(positions, simulation)
    html = env().get_template("ai_arena_positions_jp.html.j2").render(positions=payload)
    write_text(OUT_DIR / "japan/ai-arena/positions/index.html", html)
    copy_asset("ai_arena_positions_jp.css")


if __name__ == "__main__":
    main()
