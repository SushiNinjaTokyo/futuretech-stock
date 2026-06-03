#!/usr/bin/env python3
from __future__ import annotations

"""Build the GPT-4o powered AI Arena Live Council payload.

This script turns already-generated AI Arena facts into a high-quality live
conversation product for `/japan/ai-arena/log/`.

Design model
------------
Evidence -> Topic -> Debate Plan -> Dialogue -> Validator -> Memory

Why this exists
---------------
A great generative content page cannot be a list of generic GPT comments. The
script first extracts concrete simulation evidence, scores conversation topics,
plans a debate with explicit agent conflict, asks GPT-4o to write an evidence-
bound conversation, validates the result, and stores lightweight memory so later
sessions can review earlier hypotheses.

The page is still static-hosting friendly. GitHub Actions can run this script at
Open +30m, Midday, Close, and Night. The generated messages are then revealed in
browser at a 3-5 minute cadence.

Required environment
--------------------
OPENAI_API_KEY                              Required by default.
OUT_DIR                                     Default: site
AI_ARENA_WAR_ROOM_MODEL                     Default: gpt-4o
AI_ARENA_WAR_ROOM_SESSION_TYPE              auto | open_council | midday_council | close_council | night_strategy_lab | weekly_arena_review
AI_ARENA_WAR_ROOM_MESSAGES                  Optional override for target message count.
AI_ARENA_WAR_ROOM_MIN_DELAY_SECONDS         Default: 180
AI_ARENA_WAR_ROOM_MAX_DELAY_SECONDS         Default: 300
AI_ARENA_WAR_ROOM_HISTORY_DAYS              Default: 14
AI_ARENA_WAR_ROOM_TEMPERATURE               Default: 0.82
AI_ARENA_WAR_ROOM_ALLOW_FALLBACK            Default: false. Local emergency only.
AI_ARENA_WAR_ROOM_OPENAI_MIN_INTERVAL_SECONDS Default: 25. Minimum spacing between GPT requests.
AI_ARENA_WAR_ROOM_OPENAI_MAX_RETRIES           Default: 8. Retries before failing the run.
AI_ARENA_WAR_ROOM_OPENAI_429_BASE_SLEEP_SECONDS Default: 60. First long cooldown after 429.
AI_ARENA_WAR_ROOM_MAX_GPT_TOPIC_CALLS          Default: 99. Safety cap only; not a quality fallback.
"""

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - GitHub runners install PyYAML via requirements.
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
if not OUT_DIR.is_absolute():
    OUT_DIR = (ROOT / OUT_DIR).resolve()

BASE = OUT_DIR / "data" / "japan" / "ai-arena"
WAR_ROOM_DIR = BASE / "war-room"
AGENTS_YAML = ROOT / "data" / "agents" / "jp_agents.yml"
JST = timezone(timedelta(hours=9))

MODEL = os.getenv("AI_ARENA_WAR_ROOM_MODEL", "gpt-4o")
SESSION_TYPE_ENV = os.getenv("AI_ARENA_WAR_ROOM_SESSION_TYPE", "auto").strip().lower()
MIN_DELAY_SECONDS = int(os.getenv("AI_ARENA_WAR_ROOM_MIN_DELAY_SECONDS", "180"))
MAX_DELAY_SECONDS = int(os.getenv("AI_ARENA_WAR_ROOM_MAX_DELAY_SECONDS", "300"))
HISTORY_DAYS = int(os.getenv("AI_ARENA_WAR_ROOM_HISTORY_DAYS", "14"))
TEMPERATURE = float(os.getenv("AI_ARENA_WAR_ROOM_TEMPERATURE", "0.82"))
ALLOW_FALLBACK = os.getenv("AI_ARENA_WAR_ROOM_ALLOW_FALLBACK", "false").lower() in {"1", "true", "yes", "on"}
# GPT quality policy:
# Do not fill production conversations with deterministic fallback when OpenAI is rate-limited.
# Instead, pace requests aggressively and wait through 429 cooldowns. If the API remains unavailable,
# fail the run so the previous published latest.json remains intact rather than publishing weak text.
OPENAI_MIN_INTERVAL_SECONDS = float(os.getenv("AI_ARENA_WAR_ROOM_OPENAI_MIN_INTERVAL_SECONDS", "25"))
OPENAI_MAX_RETRIES = int(os.getenv("AI_ARENA_WAR_ROOM_OPENAI_MAX_RETRIES", "8"))
OPENAI_429_BASE_SLEEP_SECONDS = float(os.getenv("AI_ARENA_WAR_ROOM_OPENAI_429_BASE_SLEEP_SECONDS", "60"))
MAX_GPT_TOPIC_CALLS = int(os.getenv("AI_ARENA_WAR_ROOM_MAX_GPT_TOPIC_CALLS", "99"))
RATE_LIMIT_HIT = False
GPT_TOPIC_CALLS = 0
LAST_OPENAI_CALL_TS = 0.0

CANONICAL_NAMES = {
    "daily_striker": "KYOU",
    "weekly_sage": "NAGARE",
    "risk_sentinel": "MAMORU",
    "discovery_scout": "SAGURI",
    "contrarian_monk": "MATSU",
    "reversal_snapback": "KAESHI",
    "value_mispricing": "HIZUMI",
}

CANONICAL_COLORS = {
    "daily_striker": "#FF4B5C",
    "weekly_sage": "#B779FF",
    "risk_sentinel": "#7DF9FF",
    "discovery_scout": "#5DFFB1",
    "contrarian_monk": "#FFD166",
    "reversal_snapback": "#FF4FD8",
    "value_mispricing": "#4F46E5",
}

AGENT_PERSONAS = {
    "daily_striker": {
        "name": "KYOU",
        "role": "daily momentum striker",
        "state": "SCANNING MOMENTUM",
        "voice": "fast, direct, aggressive, obsessed with price acceleration and confirmation",
        "edge": "detects pressure before it becomes consensus",
        "weakness": "can overpay for speed when noise is high",
    },
    "weekly_sage": {
        "name": "NAGARE",
        "role": "medium-term trend sage",
        "state": "READING FLOW",
        "voice": "calm, structural, skeptical of one-day moves, focused on persistence",
        "edge": "holds winners while trend structure remains intact",
        "weakness": "accepts uncomfortable drawdown to let flow breathe",
    },
    "risk_sentinel": {
        "name": "MAMORU",
        "role": "risk sentinel",
        "state": "RISK GATE ACTIVE",
        "voice": "protective, precise, audits drawdown, sizing, concentration, and opportunity cost",
        "edge": "keeps the Arena alive when other agents chase heat",
        "weakness": "may surrender upside by demanding too much safety",
    },
    "discovery_scout": {
        "name": "SAGURI",
        "role": "hidden alpha discovery scout",
        "state": "HUNTING EARLY SIGNALS",
        "voice": "curious, early, energetic, hunts overlooked signals before they are obvious",
        "edge": "finds clean early repricing before the leaderboard notices",
        "weakness": "one sharp winner can mask low hit-rate exploration",
    },
    "contrarian_monk": {
        "name": "MATSU",
        "role": "patient pullback monk",
        "state": "WAITING FOR PULLBACK",
        "voice": "minimalist, patient, dry, refuses bad entries and treats waiting as a weapon",
        "edge": "avoids paying the first-entry tax when others chase",
        "weakness": "flat positions can become dead capital",
    },
    "reversal_snapback": {
        "name": "KAESHI",
        "role": "oversold snapback hunter",
        "state": "SNAPBACK WATCH",
        "voice": "quick, playful but exact, watches compression, exhaustion, and rebound capture",
        "edge": "sees rebound asymmetry where others only see damage",
        "weakness": "winning often is not enough if payoff capture is too small",
    },
    "value_mispricing": {
        "name": "HIZUMI",
        "role": "mispricing and valuation signal",
        "state": "TESTING MISPRICING",
        "voice": "intellectual, exacting, trap-aware, separates price movement from value evidence",
        "edge": "filters value traps and protects against false bargains",
        "weakness": "can wait too long while faster agents harvest price action",
    },
}

SESSION_DEFAULTS = {
    "open_council": {
        "title": "Open +30m Council",
        "market_phase": "open_plus_30",
        "target_messages": 12,
        "tone": "urgent, sharp, market-open tension, no long speeches",
        "purpose": "Judge whether the first thirty minutes confirm or reject yesterday's assumptions.",
    },
    "midday_council": {
        "title": "Midday Council",
        "market_phase": "midday_review",
        "target_messages": 14,
        "tone": "analytical, cool, evidence-first, focused on persistence after the morning move",
        "purpose": "Review whether the morning move has persistence and what should be watched into the afternoon.",
    },
    "close_council": {
        "title": "Post-Close Council",
        "market_phase": "post_close",
        "target_messages": 22,
        "tone": "deep, editorial, decisive, strongest daily content",
        "purpose": "Explain the day through winners, risk, attribution, exits, and tomorrow's hypotheses.",
    },
    "night_strategy_lab": {
        "title": "Night Strategy Lab",
        "market_phase": "night_reflection",
        "target_messages": 14,
        "tone": "quiet, strategic, reflective, hypothesis-driven",
        "purpose": "Turn the latest Arena evidence into tomorrow's watch items and strategy debates.",
    },
    "weekly_arena_review": {
        "title": "Weekly Arena Review",
        "market_phase": "weekly_review",
        "target_messages": 26,
        "tone": "comprehensive, attribution-heavy, strategic, reflective",
        "purpose": "Review the week through performance attribution, agent behavior, and next-week hypotheses.",
    },
}

GENERIC_BLACKLIST = [
    "eyes on",
    "vigilance maintained",
    "still in the game",
    "market noise",
    "hidden surprise",
    "calm before the storm",
    "potential energy",
    "pathways",
    "ready to spring",
    "momentum is not waiting",
    "continuous scrutiny",
    "watch closely",
    "stay vigilant",
    "only time will tell",
    "anything can happen",
]

BANNED_INVESTMENT_PHRASES = [
    "strong buy",
    "strong sell",
    "target price",
    "guaranteed",
    "easy money",
    "you should buy",
    "you should sell",
    "must buy",
    "must sell",
]

MESSAGE_TYPES = {
    "opening_observation",
    "challenge",
    "rebuttal",
    "risk_audit",
    "evidence_drop",
    "position_review",
    "leaderboard_read",
    "exit_watch",
    "watch_item",
    "hypothesis_review",
    "closing_signal",
}


@dataclass
class Topic:
    topic_id: str
    topic_type: str
    priority: int
    headline: str
    editorial_angle: str
    evidence_numbers: list[str]
    linked_symbols: list[str] = field(default_factory=list)
    required_agents: list[str] = field(default_factory=list)
    challenger_agents: list[str] = field(default_factory=list)
    why_it_matters: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_type": self.topic_type,
            "priority": self.priority,
            "headline": self.headline,
            "editorial_angle": self.editorial_angle,
            "evidence_numbers": self.evidence_numbers,
            "linked_symbols": self.linked_symbols,
            "required_agents": self.required_agents,
            "challenger_agents": self.challenger_agents,
            "why_it_matters": self.why_it_matters,
        }


def now_jst() -> datetime:
    return datetime.now(JST)


def iso_jst(dt: datetime | None = None) -> str:
    return (dt or now_jst()).astimezone(JST).isoformat(timespec="seconds")


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        print(f"WARN missing JSON: {path}")
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARN failed to read JSON {path}: {exc}")
        return fallback


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"WARN failed to read YAML {path}: {exc}")
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        rel = path.relative_to(ROOT)
    except Exception:
        rel = path
    print(f"Wrote {rel}")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def fmt_pct(value: Any, digits: int = 2) -> str:
    return f"{to_float(value):+.{digits}f}%"


def fmt_jpy(value: Any) -> str:
    return f"¥{to_float(value):,.0f}"


def clean_text(value: Any, limit: int = 520) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bKAKERU\b", "KYOU", text)
    text = re.sub(r"\bSATORI\b", "NAGARE", text)
    for phrase in BANNED_INVESTMENT_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.I)
    return text[:limit].strip(" -—,:;.")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "topic"


def agent_name(agent_id: str) -> str:
    return CANONICAL_NAMES.get(agent_id, agent_id.upper())


def agent_image(agent_id: str) -> str:
    return f"/assets/ai-arena/agents/{agent_id}.png"


def safe_first(items: list[dict[str, Any]], default: dict[str, Any] | None = None) -> dict[str, Any]:
    return items[0] if items else (default or {})


def pick_session_type(dt: datetime) -> str:
    if SESSION_TYPE_ENV and SESSION_TYPE_ENV != "auto":
        if SESSION_TYPE_ENV not in SESSION_DEFAULTS:
            raise SystemExit(f"Unknown AI_ARENA_WAR_ROOM_SESSION_TYPE={SESSION_TYPE_ENV}")
        return SESSION_TYPE_ENV
    if dt.weekday() >= 5:
        return "weekly_arena_review"
    hm = dt.hour * 60 + dt.minute
    if 9 * 60 + 20 <= hm <= 10 * 60 + 20:
        return "open_council"
    if 11 * 60 + 45 <= hm <= 12 * 60 + 45:
        return "midday_council"
    if 15 * 60 + 5 <= hm <= 17 * 60 + 30:
        return "close_council"
    return "night_strategy_lab"


def normalize_ranking(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        rows = raw.get("ranking") or raw.get("agents") or raw.get("rows") or []
    else:
        rows = raw or []
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        agent_id = row.get("agent_id") or row.get("id") or row.get("agent") or ""
        name = row.get("name") or row.get("agent_name") or agent_name(agent_id)
        return_pct = to_float(row.get("return_pct", row.get("total_return_pct", row.get("return", 0))))
        mdd = to_float(row.get("max_drawdown_pct", row.get("mdd_pct", row.get("mdd", 0))))
        win_rate = to_float(row.get("win_rate_pct", row.get("win_rate", 0)))
        equity = to_float(row.get("equity_jpy", row.get("equity", 0)))
        out.append({
            "rank": to_int(row.get("rank"), idx + 1),
            "agent_id": agent_id,
            "name": CANONICAL_NAMES.get(agent_id, str(name)),
            "return_pct": return_pct,
            "return_label": row.get("return_label") or fmt_pct(return_pct),
            "equity_jpy": equity,
            "equity_label": row.get("equity_label") or fmt_jpy(equity),
            "max_drawdown_pct": mdd,
            "mdd_label": row.get("mdd_label") or fmt_pct(mdd),
            "win_rate_pct": win_rate,
            "win_rate_label": row.get("win_rate_label") or fmt_pct(win_rate),
            "trade_count": to_int(row.get("trade_count", row.get("trades", 0))),
            "open_count": to_int(row.get("open_count", row.get("open_positions", 0))),
        })
    out.sort(key=lambda x: x["rank"])
    return out


def normalize_positions(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        rows = raw.get("open_positions") or raw.get("positions") or raw.get("open") or []
    else:
        rows = raw or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        agent_id = row.get("agent_id") or ""
        ticker = row.get("ticker") or row.get("symbol") or ""
        name = row.get("name") or row.get("company_name") or ""
        pnl = to_float(row.get("pnl_pct", row.get("unrealized_return_pct", row.get("return_pct", 0))))
        holding_days = to_int(row.get("holding_days"), 0)
        out.append({
            **row,
            "agent_id": agent_id,
            "agent_name": row.get("agent_name") or agent_name(agent_id),
            "ticker": ticker,
            "name": name,
            "holding_days": holding_days,
            "pnl_pct": pnl,
            "pnl_label": row.get("pnl_label") or fmt_pct(pnl),
            "entry_price": to_float(row.get("entry_price", 0)),
            "last_price": to_float(row.get("last_price", 0)),
            "high_water_return_pct": to_float(row.get("high_water_return_pct", pnl)),
            "weight_pct": to_float(row.get("weight_pct", 0)),
            "bucket": row.get("bucket") or "",
        })
    return out


def normalize_orders(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        rows = raw.get("events") or raw.get("feed") or raw.get("orders") or raw.get("log") or []
    else:
        rows = raw or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        agent_id = row.get("agent_id") or ""
        ticker = row.get("ticker") or row.get("linked_symbol") or row.get("symbol") or ""
        side = row.get("side") or row.get("badge") or row.get("action") or row.get("event_type") or ""
        out.append({
            "agent_id": agent_id,
            "agent_name": row.get("agent_name") or agent_name(agent_id),
            "ticker": ticker,
            "name": row.get("name") or row.get("linked_name") or row.get("company_name") or "",
            "side": str(side).upper(),
            "badge": row.get("badge") or str(side).upper(),
            "reason_code": row.get("reason_code") or row.get("reason") or "",
            "show_at": row.get("show_at") or row.get("timestamp") or row.get("scheduled_at") or "",
        })
    return out[:80]


def build_agents(ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Use YAML descriptions where available, but canonicalize names/colors so the UI
    # cannot regress into old five-agent naming.
    yaml_payload = read_yaml(AGENTS_YAML)
    yaml_agents = {}
    if isinstance(yaml_payload.get("agents"), list):
        for item in yaml_payload["agents"]:
            if isinstance(item, dict):
                yaml_agents[item.get("id") or item.get("agent_id") or ""] = item
    rank_by_agent = {r.get("agent_id"): r for r in ranking}
    agents: list[dict[str, Any]] = []
    for agent_id, persona in AGENT_PERSONAS.items():
        y = yaml_agents.get(agent_id, {})
        r = rank_by_agent.get(agent_id, {})
        agents.append({
            "agent_id": agent_id,
            "name": persona["name"],
            "role": y.get("role") or persona["role"].title(),
            "style_label": y.get("style_label") or persona["role"].title(),
            "description": y.get("description") or persona["edge"],
            "voice": persona["voice"],
            "edge": persona["edge"],
            "weakness": persona["weakness"],
            "state": persona["state"],
            "image": y.get("image") or agent_image(agent_id),
            "color": CANONICAL_COLORS[agent_id],
            "rank": r.get("rank"),
            "return_pct": r.get("return_pct"),
            "return_label": r.get("return_label", "—"),
            "mdd_label": r.get("mdd_label", "—"),
            "win_rate_label": r.get("win_rate_label", "—"),
            "trade_count": r.get("trade_count", 0),
        })
    return agents


def load_inputs() -> dict[str, Any]:
    ranking_raw = read_json(BASE / "ranking" / "latest.json", {})
    positions_raw = read_json(BASE / "positions" / "latest.json", {})
    log_raw = read_json(BASE / "log" / "latest.json", {})
    events_raw = read_json(BASE / "events" / "latest.json", {})
    summary_raw = read_json(BASE / "summary" / "latest.json", {})

    ranking = normalize_ranking(ranking_raw)
    open_positions = normalize_positions(positions_raw)
    orders = normalize_orders(log_raw) or normalize_orders(events_raw)

    # Some exporters keep richer portfolio analytics under positions/latest.json.
    portfolio = positions_raw.get("portfolio", {}) if isinstance(positions_raw, dict) else {}
    if not portfolio:
        portfolio = {
            "open_position_count": len(open_positions),
            "total_market_value_jpy": sum(to_float(p.get("last_price")) for p in open_positions),
        }

    return {
        "ranking": ranking,
        "open_positions": open_positions,
        "orders": orders,
        "portfolio": portfolio,
        "summary": summary_raw,
        "agents": build_agents(ranking),
    }


def add_topic(topics: list[Topic], topic: Topic) -> None:
    if topic.headline and topic.evidence_numbers:
        topics.append(topic)


def extract_topics(ctx: dict[str, Any], session_type: str) -> list[Topic]:
    ranking: list[dict[str, Any]] = ctx["ranking"]
    positions: list[dict[str, Any]] = ctx["open_positions"]
    portfolio: dict[str, Any] = ctx.get("portfolio") or {}
    orders: list[dict[str, Any]] = ctx.get("orders") or []
    topics: list[Topic] = []

    leader = safe_first(ranking)
    runner = ranking[1] if len(ranking) > 1 else {}
    risk_best = min(ranking, key=lambda r: abs(to_float(r.get("max_drawdown_pct"))), default={})
    win_rate_best = max(ranking, key=lambda r: to_float(r.get("win_rate_pct")), default={})
    kaeshi = next((r for r in ranking if r.get("agent_id") == "reversal_snapback"), {})
    hizumi = next((r for r in ranking if r.get("agent_id") == "value_mispricing"), {})

    if leader:
        add_topic(topics, Topic(
            topic_id="leaderboard_battle_" + slug(leader.get("name", "leader")),
            topic_type="leaderboard_battle",
            priority=94,
            headline=f"{leader['name']} leads the Arena at {leader['return_label']}, but the drawdown tells a second story",
            editorial_angle="Debate whether the current leader is truly efficient or simply the most aggressive strategy currently being rewarded.",
            evidence_numbers=[
                f"#{leader['rank']} {leader['name']}",
                f"return {leader['return_label']}",
                f"max drawdown {leader['mdd_label']}",
                f"win rate {leader['win_rate_label']}",
            ],
            required_agents=[leader.get("agent_id", "weekly_sage"), "risk_sentinel"],
            challenger_agents=["daily_striker", "value_mispricing", runner.get("agent_id", "contrarian_monk")],
            why_it_matters="Readers should understand the difference between raw leadership and risk-adjusted quality.",
        ))

    winners = sorted([p for p in positions if to_float(p.get("pnl_pct")) > 0], key=lambda p: to_float(p.get("pnl_pct")), reverse=True)
    best_open = safe_first(winners)
    if best_open:
        add_topic(topics, Topic(
            topic_id="best_open_alpha_" + slug(best_open.get("ticker", "open")),
            topic_type="best_open_alpha",
            priority=98,
            headline=f"{best_open['agent_name']} owns the cleanest open alpha: {best_open['ticker']} {best_open['pnl_label']}",
            editorial_angle="The most interesting live winner may not belong to the leaderboard leader. Make the agents debate whether it is signal quality or one-position luck.",
            evidence_numbers=[
                f"{best_open['ticker']} {best_open.get('name','')}",
                f"open return {best_open['pnl_label']}",
                f"holding days {best_open['holding_days']}",
                f"bucket {best_open.get('bucket') or 'n/a'}",
            ],
            linked_symbols=[best_open.get("ticker", "")],
            required_agents=[best_open.get("agent_id", "discovery_scout"), "risk_sentinel", "value_mispricing"],
            challenger_agents=["daily_striker", "weekly_sage"],
            why_it_matters="This identifies the current open position generating the strongest live mark-to-market evidence.",
        ))

    flat_positions = sorted(
        [p for p in positions if abs(to_float(p.get("pnl_pct"))) <= 0.25 and to_int(p.get("holding_days")) >= 8],
        key=lambda p: to_int(p.get("holding_days")),
        reverse=True,
    )
    flat = safe_first(flat_positions)
    if flat:
        add_topic(topics, Topic(
            topic_id="dead_capital_" + slug(flat.get("ticker", "flat")),
            topic_type="dead_capital",
            priority=87,
            headline=f"{flat['ticker']} is flat after {flat['holding_days']} days: setup compression or dead capital?",
            editorial_angle="Force a debate between patience, snapback potential, and opportunity cost. Avoid saying flat automatically means bullish.",
            evidence_numbers=[
                f"{flat['ticker']} {flat.get('name','')}",
                f"open return {flat['pnl_label']}",
                f"holding days {flat['holding_days']}",
                f"agent {flat['agent_name']}",
            ],
            linked_symbols=[flat.get("ticker", "")],
            required_agents=[flat.get("agent_id", "contrarian_monk"), "risk_sentinel"],
            challenger_agents=["daily_striker", "reversal_snapback", "value_mispricing"],
            why_it_matters="A flat position can still consume capital and attention; the Arena should evaluate time risk, not only price risk.",
        ))

    allocation = portfolio.get("allocation_by_agent") if isinstance(portfolio, dict) else []
    if isinstance(allocation, list) and allocation:
        top_alloc = max(allocation, key=lambda a: to_float(a.get("weight_pct")))
        if to_float(top_alloc.get("weight_pct")) >= 25:
            top_agent_id = top_alloc.get("agent_id", "")
            add_topic(topics, Topic(
                topic_id="concentration_risk_" + slug(top_agent_id or "agent"),
                topic_type="risk_council",
                priority=93,
                headline=f"{top_alloc.get('agent_name') or agent_name(top_agent_id)} controls {to_float(top_alloc.get('weight_pct')):.1f}% of open market value",
                editorial_angle="Discuss whether current Arena returns are diversified or concentrated in one style. This must be a risk conversation, not a celebration.",
                evidence_numbers=[
                    f"allocation {to_float(top_alloc.get('weight_pct')):.1f}%",
                    f"positions {to_int(top_alloc.get('position_count'))}",
                    f"unrealized P/L {fmt_jpy(top_alloc.get('unrealized_pnl_jpy'))}",
                ],
                required_agents=["risk_sentinel", top_agent_id or "weekly_sage"],
                challenger_agents=["value_mispricing", "contrarian_monk"],
                why_it_matters="Style concentration can make the Arena look stronger than its underlying diversification.",
            ))

    best_contrib = portfolio.get("best_ticker_contribution") if isinstance(portfolio, dict) else []
    worst_contrib = portfolio.get("worst_ticker_contribution") if isinstance(portfolio, dict) else []
    if isinstance(best_contrib, list) and best_contrib:
        b = best_contrib[0]
        add_topic(topics, Topic(
            topic_id="performance_attribution_" + slug(b.get("ticker", "best")),
            topic_type="performance_attribution",
            priority=82,
            headline=f"Attribution check: {b.get('ticker')} has contributed {fmt_jpy(b.get('total_pnl_jpy'))}",
            editorial_angle="Separate past realized profit engines from today's open-position story. Ask whether the current book is still driven by the same sources of alpha.",
            evidence_numbers=[
                f"best contributor {b.get('ticker')}",
                f"total P/L {fmt_jpy(b.get('total_pnl_jpy'))}",
                f"closed trades {to_int(b.get('closed_trades'))}",
            ],
            linked_symbols=[b.get("ticker", "")],
            required_agents=["value_mispricing", "weekly_sage"],
            challenger_agents=["risk_sentinel", "discovery_scout"],
            why_it_matters="Readers learn whether current leadership is backed by repeatable process or a few historical winners.",
        ))
    if isinstance(worst_contrib, list) and worst_contrib:
        w = worst_contrib[0]
        add_topic(topics, Topic(
            topic_id="loss_attribution_" + slug(w.get("ticker", "worst")),
            topic_type="risk_council",
            priority=78,
            headline=f"Loss attribution: {w.get('ticker')} remains the largest historical drag at {fmt_jpy(w.get('total_pnl_jpy'))}",
            editorial_angle="Use the worst contributor as a failure-mode discussion: what did the Arena learn, and which agent behavior should avoid repeating it?",
            evidence_numbers=[
                f"worst contributor {w.get('ticker')}",
                f"total P/L {fmt_jpy(w.get('total_pnl_jpy'))}",
                f"closed trades {to_int(w.get('closed_trades'))}",
            ],
            linked_symbols=[w.get("ticker", "")],
            required_agents=["risk_sentinel", "value_mispricing"],
            challenger_agents=["daily_striker", "contrarian_monk"],
            why_it_matters="Loss attribution keeps the page from becoming a highlight reel and improves trust.",
        ))

    if hizumi:
        add_topic(topics, Topic(
            topic_id="selectivity_hizumi",
            topic_type="strategy_clash",
            priority=74,
            headline=f"HIZUMI is quiet but selective: {hizumi.get('return_label')} return, {hizumi.get('mdd_label')} MDD, {hizumi.get('win_rate_label')} win rate",
            editorial_angle="Debate whether low drawdown and high win rate are underappreciated alpha, or simply insufficient capture.",
            evidence_numbers=[
                f"HIZUMI return {hizumi.get('return_label')}",
                f"MDD {hizumi.get('mdd_label')}",
                f"win rate {hizumi.get('win_rate_label')}",
                f"trades {hizumi.get('trade_count')}",
            ],
            required_agents=["value_mispricing", "risk_sentinel"],
            challenger_agents=["daily_striker", "discovery_scout"],
            why_it_matters="Strategy quality is not only measured by rank; error containment can be valuable.",
        ))

    if kaeshi:
        add_topic(topics, Topic(
            topic_id="payoff_ratio_kaeshi",
            topic_type="strategy_clash",
            priority=76,
            headline=f"KAESHI wins often enough but ranks low: {kaeshi.get('win_rate_label')} win rate, {kaeshi.get('return_label')} return",
            editorial_angle="Discuss payoff ratio: snapbacks may be real, but capture may be too small relative to drawdown.",
            evidence_numbers=[
                f"KAESHI return {kaeshi.get('return_label')}",
                f"win rate {kaeshi.get('win_rate_label')}",
                f"MDD {kaeshi.get('mdd_label')}",
                f"trades {kaeshi.get('trade_count')}",
            ],
            required_agents=["reversal_snapback", "risk_sentinel"],
            challenger_agents=["contrarian_monk", "value_mispricing"],
            why_it_matters="A strategy can be directionally right but economically weak if payoff capture is poor.",
        ))

    if risk_best and leader and risk_best.get("agent_id") != leader.get("agent_id"):
        add_topic(topics, Topic(
            topic_id="risk_efficiency_" + slug(risk_best.get("name", "risk")),
            topic_type="risk_council",
            priority=80,
            headline=f"{risk_best.get('name')} has the cleanest drawdown profile at {risk_best.get('mdd_label')}",
            editorial_angle="Compare the leaderboard winner with the agent showing the cleanest drawdown behavior.",
            evidence_numbers=[
                f"{risk_best.get('name')} MDD {risk_best.get('mdd_label')}",
                f"{risk_best.get('name')} return {risk_best.get('return_label')}",
                f"leader {leader.get('name')} return {leader.get('return_label')}",
            ],
            required_agents=[risk_best.get("agent_id", "risk_sentinel"), "risk_sentinel"],
            challenger_agents=[leader.get("agent_id", "weekly_sage"), "daily_striker"],
            why_it_matters="It highlights risk efficiency instead of pure return ranking.",
        ))

    recent_orders = [o for o in orders if o.get("ticker")][:8]
    if recent_orders:
        order = recent_orders[0]
        side = "IN" if "BUY" in order.get("side", "") or "IN" in order.get("badge", "") else "OUT"
        add_topic(topics, Topic(
            topic_id="latest_execution_" + slug(order.get("ticker", "order")),
            topic_type="latest_execution",
            priority=86 if session_type in {"open_council", "close_council"} else 69,
            headline=f"Latest execution: {side} {order.get('ticker')} by {order.get('agent_name')}",
            editorial_angle="Make agents explain why this execution matters, and what would prove it right or wrong. Do not claim future performance.",
            evidence_numbers=[
                f"{side} {order.get('ticker')}",
                f"agent {order.get('agent_name')}",
                f"reason {order.get('reason_code') or 'n/a'}",
            ],
            linked_symbols=[order.get("ticker", "")],
            required_agents=[order.get("agent_id") or "daily_striker", "risk_sentinel"],
            challenger_agents=["weekly_sage", "value_mispricing"],
            why_it_matters="Executions turn abstract strategy into concrete portfolio changes.",
        ))

    memory = read_memory()
    active_watch = [w for w in memory.get("watch_items", []) if w.get("status") in {"pending", "active"}]
    if active_watch:
        w = active_watch[0]
        add_topic(topics, Topic(
            topic_id="memory_review_" + slug(w.get("symbol", "watch")),
            topic_type="memory_review",
            priority=90,
            headline=f"Memory review: {w.get('owner')} is still tracking {w.get('symbol')}",
            editorial_angle="Review a previous hypothesis honestly. If evidence has not arrived, say the hypothesis is decaying.",
            evidence_numbers=[
                f"prior owner {w.get('owner')}",
                f"symbol {w.get('symbol')}",
                f"hypothesis {w.get('hypothesis')}",
            ],
            linked_symbols=[w.get("symbol", "")],
            required_agents=[w.get("agent_id", "value_mispricing"), "risk_sentinel"],
            challenger_agents=["daily_striker", "contrarian_monk"],
            why_it_matters="Memory makes the page feel like a persistent thinking system rather than a random text generator.",
        ))

    # Session-specific boosts.
    boost_by_type = {
        "open_council": {"latest_execution": 12, "dead_capital": 8, "best_open_alpha": 5},
        "midday_council": {"best_open_alpha": 10, "dead_capital": 8, "risk_council": 6},
        "close_council": {"leaderboard_battle": 10, "performance_attribution": 12, "latest_execution": 8},
        "night_strategy_lab": {"strategy_clash": 12, "memory_review": 12, "performance_attribution": 8},
        "weekly_arena_review": {"performance_attribution": 15, "leaderboard_battle": 10, "risk_council": 10},
    }.get(session_type, {})
    for topic in topics:
        topic.priority += boost_by_type.get(topic.topic_type, 0)

    # Deduplicate by topic_id and keep the most useful evidence-rich topics.
    seen: dict[str, Topic] = {}
    for topic in topics:
        existing = seen.get(topic.topic_id)
        if not existing or topic.priority > existing.priority:
            seen[topic.topic_id] = topic
    ranked = sorted(seen.values(), key=lambda t: (t.priority, len(t.evidence_numbers)), reverse=True)
    return ranked[:9]


def read_memory() -> dict[str, Any]:
    memory_path = WAR_ROOM_DIR / "memory.json"
    payload = read_json(memory_path, {"watch_items": [], "hypotheses": [], "last_sessions": []})
    if not isinstance(payload, dict):
        return {"watch_items": [], "hypotheses": [], "last_sessions": []}
    payload.setdefault("watch_items", [])
    payload.setdefault("hypotheses", [])
    payload.setdefault("last_sessions", [])
    return payload


def prune_memory(memory: dict[str, Any]) -> dict[str, Any]:
    cutoff = now_jst() - timedelta(days=10)
    def keep(item: dict[str, Any]) -> bool:
        created = str(item.get("created_at", ""))
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(JST)
        except Exception:
            return True
        return dt >= cutoff
    memory["watch_items"] = [x for x in memory.get("watch_items", []) if isinstance(x, dict) and keep(x)][-30:]
    memory["hypotheses"] = [x for x in memory.get("hypotheses", []) if isinstance(x, dict) and keep(x)][-30:]
    memory["last_sessions"] = [x for x in memory.get("last_sessions", []) if isinstance(x, dict)][-20:]
    return memory


def session_message_count(session_type: str) -> int:
    override = os.getenv("AI_ARENA_WAR_ROOM_MESSAGES")
    if override:
        return int(override)
    return int(SESSION_DEFAULTS[session_type]["target_messages"])


def select_topics_for_session(topics: list[Topic], session_type: str) -> list[Topic]:
    target = {
        "open_council": 4,
        "midday_council": 5,
        "close_council": 7,
        "night_strategy_lab": 5,
        "weekly_arena_review": 8,
    }.get(session_type, 5)
    selected = topics[:target]
    # Ensure at least one risk and one strategy/attribution topic if available.
    for required_type in ["risk_council", "strategy_clash", "performance_attribution"]:
        if len(selected) >= target + 1:
            break
        if not any(t.topic_type == required_type for t in selected):
            extra = next((t for t in topics if t.topic_type == required_type), None)
            if extra and extra.topic_id not in {t.topic_id for t in selected}:
                selected.append(extra)
    return selected[:target + 1]


def build_debate_plan(topics: list[Topic], session_type: str, ctx: dict[str, Any]) -> dict[str, Any]:
    settings = SESSION_DEFAULTS[session_type]
    target_messages = session_message_count(session_type)
    # Allocate 2-4 messages per topic. GPT may adjust but must stay near target.
    topic_plans = []
    remaining = target_messages
    for idx, topic in enumerate(topics):
        left = len(topics) - idx
        count = max(2, min(4, remaining - max(0, left - 1) * 2))
        if idx == 0 and session_type in {"close_council", "weekly_arena_review"}:
            count = min(5, count + 1)
        remaining -= count
        topic_plans.append({
            **topic.as_dict(),
            "message_target": count,
            "required_beats": [
                "specific evidence interpretation",
                "one challenge or qualification",
                "why it matters to the Arena",
            ],
        })
    return {
        "session_type": session_type,
        "session_title": settings["title"],
        "market_phase": settings["market_phase"],
        "purpose": settings["purpose"],
        "tone": settings["tone"],
        "target_messages": target_messages,
        "topics": topic_plans,
        "global_requirements": {
            "minimum_messages_with_numbers_pct": 55,
            "minimum_challenge_or_rebuttal_pct": 35,
            "must_include": ["risk audit", "opportunity cost", "watch item", "hypothesis to revisit"],
            "must_not_include": GENERIC_BLACKLIST + BANNED_INVESTMENT_PHRASES,
        },
    }


def compact_context(ctx: dict[str, Any]) -> dict[str, Any]:
    positions = sorted(ctx["open_positions"], key=lambda p: (to_float(p.get("pnl_pct")), to_float(p.get("weight_pct"))), reverse=True)[:18]
    portfolio = ctx.get("portfolio") or {}
    return {
        "agents": ctx["agents"],
        "ranking": ctx["ranking"],
        "open_positions": positions,
        "portfolio": {
            "open_position_count": portfolio.get("open_position_count") or len(ctx["open_positions"]),
            "total_market_value_jpy": portfolio.get("total_market_value_jpy"),
            "allocation_by_agent": portfolio.get("allocation_by_agent", [])[:8] if isinstance(portfolio.get("allocation_by_agent"), list) else [],
            "best_ticker_contribution": portfolio.get("best_ticker_contribution", [])[:8] if isinstance(portfolio.get("best_ticker_contribution"), list) else [],
            "worst_ticker_contribution": portfolio.get("worst_ticker_contribution", [])[:8] if isinstance(portfolio.get("worst_ticker_contribution"), list) else [],
        },
        "recent_orders": ctx.get("orders", [])[:12],
    }


def system_prompt() -> str:
    return """You are the showrunner and senior market editor for Neon Tokyo Signals' AI Arena Live Council.
You are not writing generic chatbot lines. You are creating a compelling, evidence-bound English live conversation between seven AI trading agents.

Core standard:
- Every message must either interpret a concrete number, challenge another agent, explain a risk, compare two strategies, identify a watch item, or revise a prior hypothesis.
- The agents should sound like distinct professional trading personas, not inspirational mascots.
- The conversation should feel alive: direct replies, disagreements, corrections, and evolving conclusions.
- Do not invent external news, target prices, analyst ratings, real fundamentals, or future performance.
- Do not give investment advice. This is a simulation commentary product.
- Avoid generic filler and banned phrases.
- Use concise, sharp English. 1-3 sentences per message.
- Mention Japanese tickers when evidence supports it.
- At least half of messages should include a number from the supplied evidence.

Agent voices:
KYOU: fast momentum striker; direct, impatient, price pressure first.
NAGARE: calm trend sage; structure and persistence over noise.
MAMORU: risk sentinel; drawdown, sizing, concentration, opportunity cost.
SAGURI: discovery scout; overlooked early signals and uncrowded alpha.
MATSU: patient pullback monk; terse, skeptical of bad entries, waiting as discipline.
KAESHI: snapback hunter; energetic but precise, compression and rebound capture.
HIZUMI: value distortion analyst; exacting, trap-aware, separates price from value.
"""


def user_prompt(ctx: dict[str, Any], topics: list[Topic], debate_plan: dict[str, Any], memory: dict[str, Any]) -> str:
    required_schema = {
        "daily_brief": {
            "headline": "specific editorial headline, not generic",
            "summary": "one concise paragraph explaining today's Arena tension",
            "bullets": ["4-6 evidence-rich bullets"],
        },
        "session": {
            "session_type": debate_plan["session_type"],
            "headline": "session-specific headline",
            "summary": "why this session matters",
            "messages": [
                {
                    "agent_id": "one canonical agent_id",
                    "message_type": "opening_observation | challenge | rebuttal | risk_audit | evidence_drop | position_review | leaderboard_read | exit_watch | watch_item | hypothesis_review | closing_signal",
                    "reply_to_agent": "agent name or empty string",
                    "state": "2-4 word uppercase state",
                    "mood": "calm | alert | challenging | analytical | protective | excited | skeptical",
                    "body": "1-3 sentence natural English message",
                    "evidence_label": "short label",
                    "evidence_numbers": ["specific numbers used"],
                    "linked_symbol": "ticker or empty string",
                    "linked_name": "company name or empty string",
                    "why_it_matters": "one sentence explaining reader value",
                }
            ],
            "watch_items": [
                {
                    "owner": "agent name",
                    "agent_id": "canonical agent id",
                    "symbol": "ticker or empty string",
                    "hypothesis": "specific hypothesis to revisit",
                    "check_next": "open | midday | close | next_session | next_week",
                }
            ],
            "hypotheses": [
                {
                    "owner": "agent name",
                    "agent_id": "canonical agent id",
                    "claim": "specific claim made by the council",
                    "evidence": ["numbers"],
                    "check_next": "when to revisit",
                }
            ],
        },
    }
    packed = {
        "generation_goal": "Create a high-quality live council session. The page should feel like seven AI agents genuinely thinking through the Arena state.",
        "debate_plan": debate_plan,
        "topics": [t.as_dict() for t in topics],
        "arena_context": compact_context(ctx),
        "memory_to_review": {
            "watch_items": memory.get("watch_items", [])[-8:],
            "hypotheses": memory.get("hypotheses", [])[-8:],
            "last_sessions": memory.get("last_sessions", [])[-4:],
        },
        "output_schema": required_schema,
        "hard_rules": [
            "Return valid JSON only.",
            "Do not include markdown.",
            "Do not invent facts outside the supplied Arena context.",
            "Do not use banned filler phrases.",
            "No target prices, no explicit buy/sell recommendations.",
            "Every message must include at least one evidence_number unless it directly replies to a message that did.",
            "At least 35% of messages must be challenge/rebuttal/risk_audit.",
            "Do not make every agent speak in a fixed rotation. Choose agents based on topics.",
        ],
    }
    return json.dumps(packed, ensure_ascii=False, indent=2)


def _pace_openai_request() -> None:
    """Throttle OpenAI requests so topic-by-topic generation does not burst into 429s.

    GitHub Actions can afford waiting. Publishing fewer or deterministic messages is worse
    for this product than spending a few extra minutes generating high-quality GPT-4o text.
    """
    global LAST_OPENAI_CALL_TS
    if OPENAI_MIN_INTERVAL_SECONDS <= 0:
        return
    now = time.monotonic()
    elapsed = now - LAST_OPENAI_CALL_TS
    wait = OPENAI_MIN_INTERVAL_SECONDS - elapsed
    if wait > 0:
        print(f"OpenAI pacing: waiting {wait:.1f}s before next GPT request")
        time.sleep(wait)


def _parse_retry_after_seconds(exc: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(OPENAI_429_BASE_SLEEP_SECONDS, float(retry_after))
        except Exception:
            pass
    # Long cooldowns are intentional: they protect conversation quality by avoiding fallback text.
    return min(420.0, OPENAI_429_BASE_SLEEP_SECONDS * (1.65 ** attempt))


def call_openai_json(system: str, user: str, *, temperature: float = TEMPERATURE) -> dict[str, Any]:
    """Call OpenAI and return JSON.

    Production behavior:
    - GPT-4o is the source of truth for conversation quality.
    - Requests are spaced by AI_ARENA_WAR_ROOM_OPENAI_MIN_INTERVAL_SECONDS.
    - 429 responses trigger long cooldown retries instead of deterministic fallback.
    - If OpenAI remains unavailable after retries, the workflow fails. That is intentional:
      the previous published latest.json should remain live rather than publishing weak text.
    - AI_ARENA_WAR_ROOM_ALLOW_FALLBACK=true remains available only for local emergency tests.
    """
    global RATE_LIMIT_HIT, LAST_OPENAI_CALL_TS
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if ALLOW_FALLBACK:
            print("WARN OPENAI_API_KEY missing; using local emergency fallback because AI_ARENA_WAR_ROOM_ALLOW_FALLBACK=true")
            return {"__fallback__": "missing_api_key"}
        raise SystemExit("OPENAI_API_KEY is required for AI Arena War Room GPT-4o generation.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    last_error: Exception | None = None
    for attempt in range(OPENAI_MAX_RETRIES):
        _pace_openai_request()
        LAST_OPENAI_CALL_TS = time.monotonic()
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - intentional HTTPS API call.
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                RATE_LIMIT_HIT = True
                sleep = _parse_retry_after_seconds(exc, attempt)
                if attempt < OPENAI_MAX_RETRIES - 1:
                    print(
                        f"WARN OpenAI rate limit 429 on attempt {attempt + 1}/{OPENAI_MAX_RETRIES}; "
                        f"cooling down for {sleep:.0f}s before retry"
                    )
                    time.sleep(sleep)
                    continue
                if ALLOW_FALLBACK:
                    print("WARN OpenAI rate limit persisted; using local emergency fallback because ALLOW_FALLBACK=true")
                    return {"__fallback__": "rate_limit_emergency"}
                raise SystemExit(
                    "OpenAI GPT-4o generation failed after repeated HTTP 429 rate limits. "
                    "Increase AI_ARENA_WAR_ROOM_OPENAI_MIN_INTERVAL_SECONDS or wait before rerunning; "
                    "not publishing deterministic fallback text."
                ) from exc
            if ALLOW_FALLBACK:
                print(f"WARN OpenAI HTTP error; using local emergency fallback: {exc}")
                return {"__fallback__": f"http_{exc.code}"}
            raise SystemExit(f"OpenAI GPT-4o generation failed with HTTP {exc.code}: {exc}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            sleep = min(120.0, 10.0 + attempt * 15.0)
            if attempt < OPENAI_MAX_RETRIES - 1:
                print(f"WARN OpenAI attempt {attempt + 1}/{OPENAI_MAX_RETRIES} failed: {exc}; retrying in {sleep:.0f}s")
                time.sleep(sleep)
                continue

    if ALLOW_FALLBACK:
        print(f"WARN OpenAI failed after retries; using local emergency fallback: {last_error}")
        return {"__fallback__": "generic_api_failure"}
    raise SystemExit(f"OpenAI GPT-4o generation failed after retries: {last_error}")


def fallback_dialogue(ctx: dict[str, Any], topics: list[Topic], debate_plan: dict[str, Any]) -> dict[str, Any]:
    # Emergency only. It is intentionally better than the old template fallback,
    # but production should use GPT-4o.
    messages: list[dict[str, Any]] = []
    for topic in topics:
        agents = [a for a in topic.required_agents + topic.challenger_agents if a in AGENT_PERSONAS]
        for idx, agent_id in enumerate(agents[:3]):
            persona = AGENT_PERSONAS[agent_id]
            nums = topic.evidence_numbers[:3]
            if agent_id == "risk_sentinel":
                body = f"{topic.headline}. The risk question is not whether it is interesting; it is whether the evidence survives sizing, drawdown, and time cost."
                mtype = "risk_audit"
            elif idx == 1:
                body = f"I challenge that reading. {topic.editorial_angle} Evidence: {', '.join(nums)}."
                mtype = "challenge"
            else:
                body = f"{topic.headline}. {topic.why_it_matters}"
                mtype = "evidence_drop"
            messages.append({
                "agent_id": agent_id,
                "message_type": mtype,
                "reply_to_agent": "",
                "state": persona["state"],
                "mood": "analytical",
                "body": body,
                "evidence_label": topic.headline[:80],
                "evidence_numbers": nums,
                "linked_symbol": topic.linked_symbols[0] if topic.linked_symbols else "",
                "linked_name": "",
                "why_it_matters": topic.why_it_matters,
            })
    return {
        "daily_brief": {
            "headline": topics[0].headline if topics else "AI Arena Live Council",
            "summary": debate_plan["purpose"],
            "bullets": [t.headline for t in topics[:6]],
        },
        "session": {
            "session_type": debate_plan["session_type"],
            "headline": topics[0].headline if topics else debate_plan["session_title"],
            "summary": debate_plan["purpose"],
            "messages": messages[: debate_plan["target_messages"]],
            "watch_items": [],
            "hypotheses": [],
        },
    }


def message_has_number(text: str) -> bool:
    return bool(re.search(r"[+\-]?\d+(?:\.\d+)?\s?%|¥|#\d+|\d+\s+days?", text))


def normalize_message(raw: dict[str, Any], seq: int, scheduled_base: datetime, cumulative_delay: int) -> dict[str, Any]:
    agent_id = str(raw.get("agent_id") or "").strip()
    if agent_id not in AGENT_PERSONAS:
        # Try to map by name if GPT returned a display name.
        raw_name = str(raw.get("agent_name") or raw.get("name") or "").upper()
        agent_id = next((aid for aid, name in CANONICAL_NAMES.items() if name == raw_name), "risk_sentinel")
    persona = AGENT_PERSONAS[agent_id]
    body = clean_text(raw.get("body"), 520)
    message_type = str(raw.get("message_type") or "evidence_drop").strip()
    if message_type not in MESSAGE_TYPES:
        message_type = "evidence_drop"
    evidence_numbers = raw.get("evidence_numbers")
    if not isinstance(evidence_numbers, list):
        evidence_numbers = []
    evidence_numbers = [clean_text(x, 80) for x in evidence_numbers if clean_text(x, 80)][:5]
    linked_symbol = clean_text(raw.get("linked_symbol"), 20)
    linked_name = clean_text(raw.get("linked_name"), 80)
    delay = 0 if seq == 1 else random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
    reveal_after = cumulative_delay + delay
    return {
        "message_id": f"live-msg-{seq:03d}",
        "sequence": seq,
        "agent_id": agent_id,
        "agent_name": persona["name"],
        "avatar_image": agent_image(agent_id),
        "color": CANONICAL_COLORS[agent_id],
        "state": clean_text(raw.get("state") or persona["state"], 36).upper(),
        "mood": clean_text(raw.get("mood") or "analytical", 24).lower(),
        "message_type": message_type,
        "reply_to_agent": clean_text(raw.get("reply_to_agent"), 24),
        "body": body,
        "evidence_label": clean_text(raw.get("evidence_label"), 120),
        "evidence_numbers": evidence_numbers,
        "linked_symbol": linked_symbol,
        "linked_name": linked_name,
        "why_it_matters": clean_text(raw.get("why_it_matters"), 180),
        "delay_seconds": delay,
        "reveal_after_seconds": reveal_after,
        "scheduled_at": iso_jst(scheduled_base + timedelta(seconds=reveal_after)),
    }


def validate_messages(messages: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], list[str]]:
    valid: list[dict[str, Any]] = []
    issues: list[str] = []
    seen_bodies: set[str] = set()
    agent_counts: dict[str, int] = {}
    symbol_streak = 0
    last_symbol = ""
    for msg in messages:
        body = msg.get("body", "")
        body_l = body.lower()
        if len(body) < 40:
            issues.append(f"drop short message: {body}")
            continue
        if any(p in body_l for p in GENERIC_BLACKLIST):
            issues.append(f"drop generic phrase: {body}")
            continue
        if any(p in body_l for p in BANNED_INVESTMENT_PHRASES):
            issues.append(f"drop investment advice phrase: {body}")
            continue
        dedupe_key = re.sub(r"[^a-z0-9]+", " ", body_l).strip()[:110]
        if dedupe_key in seen_bodies:
            issues.append(f"drop duplicate: {body}")
            continue
        seen_bodies.add(dedupe_key)
        symbol = msg.get("linked_symbol") or ""
        if symbol and symbol == last_symbol:
            symbol_streak += 1
        else:
            symbol_streak = 1
            last_symbol = symbol
        if symbol_streak > 4:
            issues.append(f"drop symbol streak {symbol}: {body}")
            continue
        agent_id = msg.get("agent_id", "")
        agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        if agent_counts[agent_id] > max(5, target // 3):
            issues.append(f"drop agent overuse {agent_id}: {body}")
            continue
        valid.append(msg)

    if valid:
        with_number = sum(1 for m in valid if message_has_number(m.get("body", "")) or m.get("evidence_numbers"))
        challenge = sum(1 for m in valid if m.get("message_type") in {"challenge", "rebuttal", "risk_audit"})
        if with_number / len(valid) < 0.45:
            issues.append("low number/evidence density")
        if challenge / len(valid) < 0.25:
            issues.append("low challenge/rebuttal/risk density")
    return valid, issues


def repair_if_needed(generated: dict[str, Any], ctx: dict[str, Any], topics: list[Topic], debate_plan: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    session = generated.get("session", {}) if isinstance(generated, dict) else {}
    messages = session.get("messages", []) if isinstance(session, dict) else []
    target = debate_plan["target_messages"]
    if len(messages) >= max(6, int(target * 0.75)) and not any("low" in issue for issue in issues):
        return generated
    if not os.getenv("OPENAI_API_KEY"):
        return generated
    repair_prompt = {
        "problem": "The first AI Arena Live Council draft failed validation or was too thin.",
        "issues": issues[:20],
        "requirement": "Regenerate only the session.messages array with sharper, evidence-rich, less generic dialogue. Keep daily_brief if already useful.",
        "debate_plan": debate_plan,
        "topics": [t.as_dict() for t in topics],
        "arena_context": compact_context(ctx),
        "output_schema": {"session": {"messages": "same schema as before"}},
    }
    repaired = call_openai_json(system_prompt(), json.dumps(repair_prompt, ensure_ascii=False, indent=2), temperature=max(0.55, TEMPERATURE - 0.15))
    if isinstance(repaired, dict):
        repaired_messages = (repaired.get("session") or {}).get("messages") or repaired.get("messages")
        if isinstance(repaired_messages, list) and repaired_messages:
            generated.setdefault("session", {})["messages"] = repaired_messages
    return generated



# ---------------------------------------------------------------------------
# V4 topic-by-topic dialogue generation
# ---------------------------------------------------------------------------
# V3 proved that topic extraction and debate planning worked, but one global
# GPT call often returned only one message per agent. The functions below make
# dialogue generation deterministic at the orchestration layer: every selected
# topic gets its own GPT-4o request, every request has an explicit message count,
# and the validator/repair loop runs per topic before all messages are merged.

MIN_WORDS_PER_MESSAGE = int(os.getenv("AI_ARENA_WAR_ROOM_MIN_WORDS", "22"))
MAX_WORDS_PER_MESSAGE = int(os.getenv("AI_ARENA_WAR_ROOM_MAX_WORDS", "68"))
TOPIC_REPAIR_ATTEMPTS = int(os.getenv("AI_ARENA_WAR_ROOM_TOPIC_REPAIR_ATTEMPTS", "2"))
MIN_SESSION_FILL_RATIO = float(os.getenv("AI_ARENA_WAR_ROOM_MIN_SESSION_FILL_RATIO", "0.82"))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9%¥#.+\-]+", text or ""))


def topic_plan_lookup(debate_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(t.get("topic_id")): t for t in debate_plan.get("topics", []) if isinstance(t, dict)}


def topic_message_target(topic_plan: dict[str, Any], session_type: str) -> int:
    raw = int(topic_plan.get("message_target") or 3)
    if session_type in {"close_council", "weekly_arena_review"} and topic_plan.get("priority", 0) >= 95:
        return max(raw, 4)
    return max(2, raw)


def topic_dialogue_prompt(
    ctx: dict[str, Any],
    topic: Topic,
    topic_plan: dict[str, Any],
    debate_plan: dict[str, Any],
    memory: dict[str, Any],
    previous_messages: list[dict[str, Any]],
    attempt: int = 1,
    previous_issues: list[str] | None = None,
) -> str:
    """Build a compact, strict prompt for one topic.

    The prompt is intentionally editorial rather than permissive. GPT must write
    a mini-debate, not isolated comments. We give it the topic thesis, required
    agents, challengers, exact evidence numbers, and the last few messages to
    preserve conversational continuity.
    """
    target = topic_message_target(topic_plan, debate_plan["session_type"])
    agents_to_use: list[str] = []
    for aid in list(topic.required_agents) + list(topic.challenger_agents):
        if aid in AGENT_PERSONAS and aid not in agents_to_use:
            agents_to_use.append(aid)
    if not agents_to_use:
        agents_to_use = ["risk_sentinel", "weekly_sage", "value_mispricing"]

    schema = {
        "topic_id": topic.topic_id,
        "messages": [
            {
                "agent_id": "one canonical agent id from allowed_agents",
                "message_type": "opening_observation | challenge | rebuttal | risk_audit | evidence_drop | position_review | leaderboard_read | exit_watch | watch_item | hypothesis_review | closing_signal",
                "reply_to_agent": "canonical display name being answered, or empty only for the first message",
                "state": "2-4 word uppercase state",
                "mood": "calm | alert | challenging | analytical | protective | excited | skeptical",
                "body": f"{MIN_WORDS_PER_MESSAGE}-{MAX_WORDS_PER_MESSAGE} words, 1-3 sentences, natural English, evidence-bound",
                "evidence_label": "short evidence label",
                "evidence_numbers": ["must include at least one supplied number unless no numerical evidence exists"],
                "linked_symbol": "ticker from linked_symbols or empty string",
                "linked_name": "company name if known or empty string",
                "why_it_matters": "one concrete reader-value sentence",
            }
        ],
        "watch_items": [
            {
                "owner": "agent name",
                "agent_id": "canonical agent id",
                "symbol": "ticker or empty string",
                "hypothesis": "specific hypothesis to revisit, not generic",
                "check_next": "open | midday | close | next_session | next_week",
            }
        ],
        "hypotheses": [
            {
                "owner": "agent name",
                "agent_id": "canonical agent id",
                "claim": "specific claim made by this topic debate",
                "evidence": ["numbers"],
                "check_next": "when to revisit",
            }
        ],
    }

    packed = {
        "task": "Write one sharp mini-debate for this single AI Arena topic. Do not summarize. Make the agents react to each other.",
        "attempt": attempt,
        "previous_issues_to_fix": previous_issues or [],
        "session": {
            "session_type": debate_plan["session_type"],
            "session_title": debate_plan["session_title"],
            "market_phase": debate_plan["market_phase"],
            "tone": debate_plan["tone"],
        },
        "topic": topic_plan,
        "target_message_count": target,
        "allowed_agents": [
            {
                "agent_id": aid,
                "name": AGENT_PERSONAS[aid]["name"],
                "voice": AGENT_PERSONAS[aid]["voice"],
                "edge": AGENT_PERSONAS[aid]["edge"],
                "weakness": AGENT_PERSONAS[aid]["weakness"],
            }
            for aid in agents_to_use
        ],
        "conversation_continuity": [
            {
                "agent_name": m.get("agent_name"),
                "body": m.get("body"),
                "topic_id": m.get("topic_id", ""),
            }
            for m in previous_messages[-5:]
        ],
        "arena_context": compact_context(ctx),
        "memory_to_use_if_relevant": {
            "watch_items": memory.get("watch_items", [])[-6:],
            "hypotheses": memory.get("hypotheses", [])[-6:],
        },
        "hard_requirements": [
            f"Return exactly {target} messages for this topic.",
            f"Each body must be {MIN_WORDS_PER_MESSAGE}-{MAX_WORDS_PER_MESSAGE} words.",
            "At least one message must challenge, qualify, or correct another agent.",
            "At least one message must explain why the evidence could be misleading.",
            "At least one message must say what would confirm or weaken the hypothesis later.",
            "Use the supplied evidence_numbers directly; do not invent new numbers.",
            "Do not use generic filler, motivational language, target prices, or buy/sell advice.",
            "Do not say 'luck' unless you define what evidence would separate luck from process.",
            "Avoid repeating the same sentence structure across messages.",
            "Return valid JSON only.",
        ],
        "banned_phrases": GENERIC_BLACKLIST + BANNED_INVESTMENT_PHRASES,
        "output_schema": schema,
    }
    return json.dumps(packed, ensure_ascii=False, indent=2)


def extract_topic_generation(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Accept several JSON shapes to make GPT output robust."""
    if not isinstance(payload, dict):
        return [], [], []
    messages = payload.get("messages")
    watch_items = payload.get("watch_items")
    hypotheses = payload.get("hypotheses")
    if not isinstance(messages, list):
        topic_obj = payload.get("topic") if isinstance(payload.get("topic"), dict) else {}
        messages = topic_obj.get("messages")
        watch_items = topic_obj.get("watch_items") or watch_items
        hypotheses = topic_obj.get("hypotheses") or hypotheses
    if not isinstance(messages, list):
        session_obj = payload.get("session") if isinstance(payload.get("session"), dict) else {}
        messages = session_obj.get("messages")
        watch_items = session_obj.get("watch_items") or watch_items
        hypotheses = session_obj.get("hypotheses") or hypotheses
    return (
        [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else [],
        [w for w in watch_items if isinstance(w, dict)] if isinstance(watch_items, list) else [],
        [h for h in hypotheses if isinstance(h, dict)] if isinstance(hypotheses, list) else [],
    )


def backfill_message_from_topic(msg: dict[str, Any], topic: Topic, topic_plan: dict[str, Any]) -> dict[str, Any]:
    """Fill missing evidence fields before validation.

    This is not content generation. It only copies known topic evidence into
    empty metadata fields so high-quality text does not get dropped because GPT
    omitted a duplicated JSON field.
    """
    out = dict(msg)
    if not out.get("evidence_numbers"):
        out["evidence_numbers"] = list(topic.evidence_numbers[:3])
    if not out.get("evidence_label"):
        out["evidence_label"] = topic.headline[:100]
    if not out.get("linked_symbol") and topic.linked_symbols:
        out["linked_symbol"] = topic.linked_symbols[0]
    if not out.get("why_it_matters"):
        out["why_it_matters"] = topic.why_it_matters
    out["topic_id"] = topic.topic_id
    out["topic_type"] = topic.topic_type
    return out


def validate_topic_messages(messages: list[dict[str, Any]], topic: Topic, expected: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate one topic mini-debate more strictly than the session-level pass."""
    valid: list[dict[str, Any]] = []
    issues: list[str] = []
    seen_bodies: set[str] = set()
    challenge_count = 0
    for msg in messages:
        body = clean_text(msg.get("body"), 700)
        lower = body.lower()
        wc = word_count(body)
        if wc < MIN_WORDS_PER_MESSAGE:
            issues.append(f"topic {topic.topic_id}: drop short body ({wc} words): {body}")
            continue
        if wc > MAX_WORDS_PER_MESSAGE + 20:
            issues.append(f"topic {topic.topic_id}: body too long ({wc} words): {body[:120]}")
            continue
        if any(p in lower for p in GENERIC_BLACKLIST):
            issues.append(f"topic {topic.topic_id}: generic phrase: {body}")
            continue
        if any(p in lower for p in BANNED_INVESTMENT_PHRASES):
            issues.append(f"topic {topic.topic_id}: banned advice phrase: {body}")
            continue
        key = re.sub(r"[^a-z0-9]+", " ", lower).strip()[:140]
        if key in seen_bodies:
            issues.append(f"topic {topic.topic_id}: duplicate body: {body}")
            continue
        seen_bodies.add(key)
        mtype = str(msg.get("message_type") or "")
        if mtype in {"challenge", "rebuttal", "risk_audit"}:
            challenge_count += 1
        ev = msg.get("evidence_numbers") if isinstance(msg.get("evidence_numbers"), list) else []
        if not ev and not message_has_number(body):
            issues.append(f"topic {topic.topic_id}: no evidence number: {body}")
            continue
        valid.append(msg)
    if len(valid) < expected:
        issues.append(f"topic {topic.topic_id}: expected {expected}, got {len(valid)}")
    if expected >= 3 and challenge_count == 0:
        issues.append(f"topic {topic.topic_id}: missing challenge/rebuttal/risk audit")
    return valid, issues


def generate_topic_dialogue(
    ctx: dict[str, Any],
    topic: Topic,
    topic_plan: dict[str, Any],
    debate_plan: dict[str, Any],
    memory: dict[str, Any],
    previous_messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Generate and validate a mini-debate for one topic.

    The function retries per topic instead of regenerating the whole session.
    This is the core fix for the previous failure mode where a target 22-message
    session collapsed into seven shallow comments.
    """
    expected = topic_message_target(topic_plan, debate_plan["session_type"])
    all_issues: list[str] = []
    best_valid: list[dict[str, Any]] = []
    best_watch: list[dict[str, Any]] = []
    best_hypotheses: list[dict[str, Any]] = []

    global GPT_TOPIC_CALLS
    for attempt in range(1, TOPIC_REPAIR_ATTEMPTS + 2):
        if not os.getenv("OPENAI_API_KEY") and not ALLOW_FALLBACK:
            raise SystemExit("OPENAI_API_KEY is required for AI Arena Live Council generation")
        if GPT_TOPIC_CALLS >= MAX_GPT_TOPIC_CALLS and not ALLOW_FALLBACK:
            raise SystemExit(
                f"AI_ARENA_WAR_ROOM_MAX_GPT_TOPIC_CALLS={MAX_GPT_TOPIC_CALLS} was reached before the session was complete. "
                "Raise the cap or reduce AI_ARENA_WAR_ROOM_MESSAGES; not publishing deterministic fallback text."
            )

        GPT_TOPIC_CALLS += 1
        raw_payload = call_openai_json(
            system_prompt(),
            topic_dialogue_prompt(ctx, topic, topic_plan, debate_plan, memory, previous_messages, attempt, all_issues[-10:]),
            temperature=max(0.52, TEMPERATURE - 0.06 * (attempt - 1)),
        )
        if raw_payload.get("__fallback__"):
            raw_payload = {"messages": fallback_dialogue(ctx, [topic], {**debate_plan, "target_messages": expected}).get("session", {}).get("messages", [])}

        raw_messages, watch_items, hypotheses = extract_topic_generation(raw_payload)
        prepared = [backfill_message_from_topic(m, topic, topic_plan) for m in raw_messages]
        valid, issues = validate_topic_messages(prepared, topic, expected)
        if len(valid) > len(best_valid):
            best_valid, best_watch, best_hypotheses = valid, watch_items, hypotheses
        all_issues.extend(issues)
        if len(valid) >= expected and not any("generic phrase" in x or "banned" in x for x in issues):
            return valid[:expected], watch_items, hypotheses, all_issues

    # If GPT still under-fills, return the best partial result. The session-level
    # top-up pass will fill the remaining slots with another GPT call.
    return best_valid[:expected], best_watch, best_hypotheses, all_issues


def topup_dialogue_prompt(
    ctx: dict[str, Any],
    debate_plan: dict[str, Any],
    selected_topics: list[Topic],
    existing_messages: list[dict[str, Any]],
    missing: int,
    issues: list[str],
) -> str:
    packed = {
        "task": "The Live Council session is under-filled. Add missing high-quality messages that continue the existing conversation.",
        "missing_message_count": missing,
        "session": {
            "session_type": debate_plan["session_type"],
            "session_title": debate_plan["session_title"],
            "target_messages": debate_plan["target_messages"],
        },
        "selected_topics": [t.as_dict() for t in selected_topics],
        "existing_messages": [
            {
                "agent_name": m.get("agent_name"),
                "message_type": m.get("message_type"),
                "body": m.get("body"),
                "topic_id": m.get("topic_id", ""),
            }
            for m in existing_messages[-12:]
        ],
        "known_validation_issues": issues[-20:],
        "arena_context": compact_context(ctx),
        "requirements": [
            f"Return exactly {missing} additional messages.",
            f"Each body must be {MIN_WORDS_PER_MESSAGE}-{MAX_WORDS_PER_MESSAGE} words.",
            "Do not repeat any existing message or topic angle verbatim.",
            "Prioritize unserved topics, performance attribution, dead capital, payoff ratio, and memory/watch items.",
            "Each message must include evidence_numbers and why_it_matters.",
            "Return JSON only with key messages.",
        ],
        "output_schema": {"messages": ["same message schema as topic generation"]},
    }
    return json.dumps(packed, ensure_ascii=False, indent=2)


def generate_topup_messages(
    ctx: dict[str, Any],
    debate_plan: dict[str, Any],
    selected_topics: list[Topic],
    existing_messages: list[dict[str, Any]],
    missing: int,
    issues: list[str],
) -> list[dict[str, Any]]:
    if missing <= 0:
        return []

    default_topic = selected_topics[0] if selected_topics else Topic("general", "strategy_clash", 1, "Arena Council", "", [], [], ["risk_sentinel"], [], "")

    if not os.getenv("OPENAI_API_KEY") and not ALLOW_FALLBACK:
        raise SystemExit("OPENAI_API_KEY is required for top-up generation; not publishing deterministic fallback text.")

    raw_payload = call_openai_json(
        system_prompt(),
        topup_dialogue_prompt(ctx, debate_plan, selected_topics, existing_messages, missing, issues),
        temperature=max(0.55, TEMPERATURE - 0.12),
    )
    if raw_payload.get("__fallback__"):
        fb = fallback_dialogue(ctx, selected_topics, {**debate_plan, "target_messages": missing})
        raw_messages = (fb.get("session") or {}).get("messages", [])
        prepared = [backfill_message_from_topic(m, default_topic, default_topic.as_dict()) for m in raw_messages]
        valid, _ = validate_topic_messages(prepared, default_topic, min(missing, len(prepared)))
        return valid[:missing]
    raw_messages, _, _ = extract_topic_generation(raw_payload)
    prepared = [backfill_message_from_topic(m, default_topic, default_topic.as_dict()) for m in raw_messages]
    valid, _ = validate_topic_messages(prepared, default_topic, min(missing, len(prepared)))
    return valid[:missing]


def build_daily_brief_prompt(ctx: dict[str, Any], selected_topics: list[Topic], messages: list[dict[str, Any]], debate_plan: dict[str, Any]) -> str:
    packed = {
        "task": "Write a sharp editorial daily_brief for the AI Arena Live Council page.",
        "session": debate_plan,
        "topics": [t.as_dict() for t in selected_topics[:8]],
        "messages": [{"agent": m.get("agent_name"), "body": m.get("body"), "evidence_numbers": m.get("evidence_numbers")} for m in messages[:24]],
        "arena_context": compact_context(ctx),
        "requirements": [
            "Headline must be specific and based on the strongest tension, not generic.",
            "Summary must explain what changed or what the reader should understand.",
            "Bullets must be evidence-rich and interpretive, not table readouts.",
            "Return JSON only with headline, summary, bullets.",
        ],
        "output_schema": {"daily_brief": {"headline": "", "summary": "", "bullets": [""]}},
    }
    return json.dumps(packed, ensure_ascii=False, indent=2)


def generate_daily_brief(ctx: dict[str, Any], selected_topics: list[Topic], messages: list[dict[str, Any]], debate_plan: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("OPENAI_API_KEY"):
        try:
            payload = call_openai_json(system_prompt(), build_daily_brief_prompt(ctx, selected_topics, messages, debate_plan), temperature=max(0.45, TEMPERATURE - 0.2))
            brief = payload.get("daily_brief") if isinstance(payload, dict) else None
            if isinstance(brief, dict):
                headline = clean_text(brief.get("headline"), 160)
                summary = clean_text(brief.get("summary"), 300)
                bullets = [clean_text(x, 170) for x in brief.get("bullets", []) if clean_text(x, 170)] if isinstance(brief.get("bullets"), list) else []
                if headline and summary and len(bullets) >= 3:
                    return {"headline": headline, "summary": summary, "bullets": bullets[:6]}
        except Exception as exc:  # pragma: no cover - non-critical editorial fallback.
            print(f"WARN daily brief generation failed: {exc}")
    return {
        "headline": selected_topics[0].headline if selected_topics else "AI Arena Live Council",
        "summary": debate_plan["purpose"],
        "bullets": [t.headline for t in selected_topics[:6]],
    }


def normalize_generated_messages(raw_messages: list[dict[str, Any]], generated_at: datetime) -> list[dict[str, Any]]:
    cumulative = 0
    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_messages, start=1):
        if not isinstance(raw, dict):
            continue
        msg = normalize_message(raw, idx, generated_at, cumulative)
        # Preserve topic metadata added during topic generation.
        if raw.get("topic_id"):
            msg["topic_id"] = raw.get("topic_id")
        if raw.get("topic_type"):
            msg["topic_type"] = raw.get("topic_type")
        cumulative = msg["reveal_after_seconds"]
        normalized.append(msg)
    return normalized

def build_evidence_tape(ctx: dict[str, Any], topics: list[Topic]) -> list[dict[str, Any]]:
    tape: list[dict[str, Any]] = []
    for r in ctx["ranking"]:
        agent_id = r.get("agent_id", "")
        tape.append({
            "label": f"#{r.get('rank')} {r.get('name')}",
            "value": f"{r.get('return_label')} / MDD {r.get('mdd_label')}",
            "agent_id": agent_id,
            "color": CANONICAL_COLORS.get(agent_id, "#7DF9FF"),
        })
    for topic in topics[:8]:
        tape.append({
            "label": topic.topic_type.replace("_", " ").upper(),
            "value": topic.headline,
            "agent_id": topic.required_agents[0] if topic.required_agents else "risk_sentinel",
            "color": CANONICAL_COLORS.get(topic.required_agents[0], "#7DF9FF") if topic.required_agents else "#7DF9FF",
        })
    for p in sorted(ctx["open_positions"], key=lambda x: abs(to_float(x.get("pnl_pct"))), reverse=True)[:8]:
        agent_id = p.get("agent_id", "")
        tape.append({
            "label": f"OPEN {p.get('ticker')}",
            "value": f"{p.get('agent_name')} {p.get('pnl_label')} / {p.get('holding_days')}d",
            "agent_id": agent_id,
            "color": CANONICAL_COLORS.get(agent_id, "#7DF9FF"),
        })
    return tape[:32]


def build_pulse(messages: list[dict[str, Any]], agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_agent: dict[str, dict[str, Any]] = {a["agent_id"]: a for a in agents}
    used: set[str] = set()
    pulse: list[dict[str, Any]] = []
    for msg in messages:
        agent_id = msg.get("agent_id")
        if agent_id in used or agent_id not in by_agent:
            continue
        used.add(agent_id)
        pulse.append({
            "agent_id": agent_id,
            "agent_name": by_agent[agent_id]["name"],
            "color": by_agent[agent_id]["color"],
            "body": clean_text(msg.get("why_it_matters") or msg.get("body"), 96),
            "state": msg.get("state") or by_agent[agent_id]["state"],
        })
    for agent in agents:
        if agent["agent_id"] not in used:
            pulse.append({
                "agent_id": agent["agent_id"],
                "agent_name": agent["name"],
                "color": agent["color"],
                "body": agent.get("edge", agent.get("description", ""))[:96],
                "state": agent["state"],
            })
    return pulse[:7]


def merge_sessions(existing: dict[str, Any], new_session: dict[str, Any]) -> list[dict[str, Any]]:
    today = now_jst().date().isoformat()
    existing_sessions = existing.get("sessions", []) if isinstance(existing, dict) else []
    sessions: list[dict[str, Any]] = []
    for s in existing_sessions:
        if not isinstance(s, dict):
            continue
        generated_at = str(s.get("generated_at", ""))
        # Keep today and yesterday only in latest payload; full history is archived.
        if generated_at[:10] >= (now_jst().date() - timedelta(days=1)).isoformat():
            if s.get("session_id") != new_session.get("session_id"):
                sessions.append(s)
    sessions.append(new_session)
    sessions.sort(key=lambda x: str(x.get("generated_at", "")))
    return sessions[-8:]


def flatten_live_messages(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    seq = 1
    base = now_jst()
    cumulative = 0
    for session in sessions:
        for raw in session.get("messages", []):
            msg = dict(raw)
            msg["session_id"] = session.get("session_id", "")
            msg["session_type"] = session.get("session_type", "")
            msg["session_title"] = session.get("session_title", "")
            msg["global_sequence"] = seq
            # Preserve per-session delay but re-schedule flattened feed from page generation time.
            msg["sequence"] = seq
            msg["message_id"] = f"live-msg-{seq:03d}"
            delay = 0 if seq == 1 else random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            cumulative += delay
            msg["delay_seconds"] = delay
            msg["reveal_after_seconds"] = 0 if seq == 1 else cumulative
            msg["scheduled_at"] = iso_jst(base + timedelta(seconds=msg["reveal_after_seconds"]))
            messages.append(msg)
            seq += 1
    return messages


def update_memory(memory: dict[str, Any], session: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    session_payload = generated.get("session", {}) if isinstance(generated, dict) else {}
    for item in session_payload.get("watch_items", []) or []:
        if isinstance(item, dict):
            memory.setdefault("watch_items", []).append({
                "created_at": iso_jst(),
                "owner": clean_text(item.get("owner"), 40),
                "agent_id": clean_text(item.get("agent_id"), 40),
                "symbol": clean_text(item.get("symbol"), 20),
                "hypothesis": clean_text(item.get("hypothesis"), 260),
                "check_next": clean_text(item.get("check_next"), 40),
                "status": "pending",
            })
    for item in session_payload.get("hypotheses", []) or []:
        if isinstance(item, dict):
            memory.setdefault("hypotheses", []).append({
                "created_at": iso_jst(),
                "owner": clean_text(item.get("owner"), 40),
                "agent_id": clean_text(item.get("agent_id"), 40),
                "claim": clean_text(item.get("claim"), 260),
                "evidence": [clean_text(x, 80) for x in item.get("evidence", []) if clean_text(x, 80)][:5] if isinstance(item.get("evidence"), list) else [],
                "check_next": clean_text(item.get("check_next"), 40),
                "status": "pending",
            })
    memory.setdefault("last_sessions", []).append({
        "session_id": session.get("session_id"),
        "session_type": session.get("session_type"),
        "generated_at": session.get("generated_at"),
        "headline": session.get("headline"),
        "message_count": len(session.get("messages", [])),
    })
    return prune_memory(memory)


def prune_history() -> None:
    history = WAR_ROOM_DIR / "history"
    if not history.exists():
        return
    cutoff = now_jst().date() - timedelta(days=HISTORY_DAYS)
    for path in history.glob("*.json"):
        try:
            day = datetime.strptime(path.stem[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if day < cutoff:
            path.unlink(missing_ok=True)
            print(f"Pruned old war-room history {path.name}")


def build_payload() -> dict[str, Any]:
    """Build the full War Room payload.

    V4 generation deliberately avoids a single monolithic dialogue call. It
    extracts topics once, plans the session once, then asks GPT-4o to generate a
    validated mini-debate for every selected topic. This guarantees that a
    `close_council` target of 22 messages does not collapse into seven shallow
    one-liners.
    """
    ctx = load_inputs()
    generated_at = now_jst()
    session_type = pick_session_type(generated_at)
    all_topics = extract_topics(ctx, session_type)
    selected_topics = select_topics_for_session(all_topics, session_type)
    debate_plan = build_debate_plan(selected_topics, session_type, ctx)
    memory = read_memory()
    plan_by_id = topic_plan_lookup(debate_plan)

    # Generate dialogue per topic. The orchestration layer, not GPT, owns the
    # message count guarantee.
    topic_messages: list[dict[str, Any]] = []
    watch_items: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    validation_issues: list[str] = []

    for topic in selected_topics:
        plan = plan_by_id.get(topic.topic_id, topic.as_dict())
        generated_for_topic, topic_watch, topic_hypotheses, issues = generate_topic_dialogue(
            ctx=ctx,
            topic=topic,
            topic_plan=plan,
            debate_plan=debate_plan,
            memory=memory,
            previous_messages=topic_messages,
        )
        topic_messages.extend(generated_for_topic)
        watch_items.extend(topic_watch)
        hypotheses.extend(topic_hypotheses)
        validation_issues.extend(issues)
        print(
            f"Topic {topic.topic_id}: target={topic_message_target(plan, session_type)} "
            f"accepted={len(generated_for_topic)} issues={len(issues)}"
        )

    target = debate_plan["target_messages"]
    minimum_required = max(8, int(target * MIN_SESSION_FILL_RATIO))
    if len(topic_messages) < minimum_required:
        missing = minimum_required - len(topic_messages)
        print(f"Session under-filled after topic generation: {len(topic_messages)}/{target}; requesting {missing} top-up messages")
        topups = generate_topup_messages(ctx, debate_plan, selected_topics, topic_messages, missing, validation_issues)
        topic_messages.extend(topups)

    # Normalize, validate at session level, and trim to the intended target.
    normalized = normalize_generated_messages(topic_messages, generated_at)
    valid, session_issues = validate_messages(normalized, target)
    validation_issues.extend(session_issues)

    if len(valid) < minimum_required:
        missing = minimum_required - len(valid)
        print(f"Session still under-filled after validation: {len(valid)}/{target}; requesting repair top-up={missing}")
        repaired = generate_topup_messages(ctx, debate_plan, selected_topics, valid, missing, validation_issues)
        normalized_repaired = normalize_generated_messages([*valid, *repaired], generated_at)
        valid, more_issues = validate_messages(normalized_repaired, target)
        validation_issues.extend(more_issues)

    if len(valid) < minimum_required:
        if not (ALLOW_FALLBACK or ALLOW_RATE_LIMIT_FALLBACK or RATE_LIMIT_HIT):
            raise SystemExit(
                f"GPT output failed minimum message requirement: got {len(valid)}, "
                f"required {minimum_required}. Issues: {validation_issues[:12]}"
            )
        print(
            f"WARN Session remains under-filled ({len(valid)}/{minimum_required}); "
            "using deterministic evidence fallback to complete the page"
        )
        fallback = fallback_dialogue(ctx, selected_topics, debate_plan)
        fallback_messages = (fallback.get("session") or {}).get("messages", [])
        normalized_fallback = normalize_generated_messages(fallback_messages, generated_at)
        valid = [*valid, *normalized_fallback][:target]

    valid = valid[:target]
    # Re-number after validation and trimming so the UI has clean sequencing.
    cumulative = 0
    renumbered: list[dict[str, Any]] = []
    for idx, msg in enumerate(valid, start=1):
        raw = dict(msg)
        raw["message_id"] = f"live-msg-{idx:03d}"
        raw["sequence"] = idx
        delay = 0 if idx == 1 else random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
        cumulative += delay
        raw["delay_seconds"] = delay
        raw["reveal_after_seconds"] = 0 if idx == 1 else cumulative
        raw["scheduled_at"] = iso_jst(generated_at + timedelta(seconds=raw["reveal_after_seconds"]))
        renumbered.append(raw)
    valid = renumbered

    daily_brief = generate_daily_brief(ctx, selected_topics, valid, debate_plan)

    # If GPT did not provide watch items/hypotheses, create high-quality memory
    # from the strongest topics and accepted messages. This preserves continuity
    # for the next session without inventing external facts.
    if not watch_items:
        for topic in selected_topics[:3]:
            owner_id = topic.required_agents[0] if topic.required_agents else "risk_sentinel"
            watch_items.append({
                "owner": CANONICAL_NAMES.get(owner_id, owner_id),
                "agent_id": owner_id,
                "symbol": topic.linked_symbols[0] if topic.linked_symbols else "",
                "hypothesis": f"{topic.headline}. Revisit whether this remains true as new Arena evidence arrives.",
                "check_next": "next_session",
            })
    if not hypotheses:
        for topic in selected_topics[:2]:
            owner_id = topic.required_agents[0] if topic.required_agents else "risk_sentinel"
            hypotheses.append({
                "owner": CANONICAL_NAMES.get(owner_id, owner_id),
                "agent_id": owner_id,
                "claim": topic.editorial_angle or topic.headline,
                "evidence": topic.evidence_numbers[:4],
                "check_next": "next_session",
            })

    session_id = f"{generated_at.date().isoformat()}-{session_type}"
    session = {
        "session_id": session_id,
        "session_type": session_type,
        "session_title": debate_plan["session_title"],
        "market_phase": debate_plan["market_phase"],
        "generated_at": iso_jst(generated_at),
        "headline": clean_text(daily_brief.get("headline") or (selected_topics[0].headline if selected_topics else debate_plan["session_title"]), 160),
        "summary": clean_text(daily_brief.get("summary") or debate_plan["purpose"], 300),
        "topics": [t.as_dict() for t in selected_topics],
        "debate_plan": debate_plan,
        "messages": valid,
        "quality": {
            "messages_generated": len(valid),
            "target_messages": target,
            "minimum_required": minimum_required,
            "messages_with_numbers": sum(1 for m in valid if message_has_number(m.get("body", "")) or m.get("evidence_numbers")),
            "challenge_or_risk_messages": sum(1 for m in valid if m.get("message_type") in {"challenge", "rebuttal", "risk_audit"}),
            "validation_issues": validation_issues[:40],
        },
    }

    generated_memory_source = {"session": {"watch_items": watch_items, "hypotheses": hypotheses}}
    memory = update_memory(memory, session, generated_memory_source)
    write_json(WAR_ROOM_DIR / "memory.json", memory)

    existing = read_json(WAR_ROOM_DIR / "latest.json", {})
    sessions = merge_sessions(existing, session)
    live_messages = flatten_live_messages(sessions)

    bullets = daily_brief.get("bullets") if isinstance(daily_brief.get("bullets"), list) else []
    bullets = [clean_text(x, 170) for x in bullets if clean_text(x, 170)]
    if len(bullets) < 3:
        bullets = [t.headline for t in selected_topics[:6]]

    payload = {
        "schema_version": "ai_arena_live_council_v4_topic_by_topic_dialogue_validator",
        "generated_at": iso_jst(generated_at),
        "page": {
            "title": "AI Arena Live Council",
            "subtitle": "Seven trading agents debate Japanese equities through live simulation evidence, memory, and GPT-4o reasoning.",
        },
        "daily_brief": {
            "headline": clean_text(daily_brief.get("headline"), 160),
            "summary": clean_text(daily_brief.get("summary"), 300),
            "bullets": bullets[:6],
        },
        "current_session": session,
        "sessions": sessions,
        "agents": ctx["agents"],
        "ranking": ctx["ranking"],
        "open_positions": ctx["open_positions"],
        "portfolio": ctx["portfolio"],
        "topics": [t.as_dict() for t in all_topics],
        "live_messages": live_messages,
        "feed": live_messages,
        "threads": [],
        "evidence_tape": build_evidence_tape(ctx, selected_topics),
        "pulse": build_pulse(live_messages, ctx["agents"]),
        "memory": {
            "watch_items": memory.get("watch_items", [])[-10:],
            "hypotheses": memory.get("hypotheses", [])[-10:],
        },
        "live_config": {
            "mode": "browser_reveal_queue",
            "min_delay_seconds": MIN_DELAY_SECONDS,
            "max_delay_seconds": MAX_DELAY_SECONDS,
            "message_count": len(live_messages),
            "schedule_note": "The static page reveals pre-generated GPT-4o messages at random 3-5 minute intervals in the browser. GitHub Actions should generate new sessions at Open +30m, Midday, Close, and Night.",
        },
        "metrics": {
            "agent_count": len(ctx["agents"]),
            "message_count": len(live_messages),
            "session_message_count": len(valid),
            "session_target_messages": target,
            "session_count": len(sessions),
            "open_position_count": len(ctx["open_positions"]),
            "ranking_count": len(ctx["ranking"]),
            "topic_count": len(all_topics),
            "selected_topic_count": len(selected_topics),
            "evidence_items": len(build_evidence_tape(ctx, selected_topics)),
        },
        "ai": {
            "required": True,
            "enabled": bool(os.getenv("OPENAI_API_KEY")),
            "status": (
                "gpt-4o_topic_by_topic_generated_after_429_cooldown" if RATE_LIMIT_HIT
                else ("gpt-4o_topic_by_topic_generated" if os.getenv("OPENAI_API_KEY") else "emergency_fallback")
            ),
            "rate_limit_encountered": RATE_LIMIT_HIT,
            "rate_limit_fallback_used": False,
            "openai_min_interval_seconds": OPENAI_MIN_INTERVAL_SECONDS,
            "openai_max_retries": OPENAI_MAX_RETRIES,
            "openai_429_base_sleep_seconds": OPENAI_429_BASE_SLEEP_SECONDS,
            "max_gpt_topic_calls": MAX_GPT_TOPIC_CALLS,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "pipeline": "Evidence -> Topic -> Debate Plan -> Topic Dialogue -> Validator -> Repair -> Memory",
        },
        "disclaimer": "AI Arena is a quantitative simulation and generative discussion interface. Informational only. Not investment advice.",
    }
    return payload

def main() -> int:
    random.seed(f"{iso_jst()}:{MODEL}:{SESSION_TYPE_ENV}")
    payload = build_payload()
    write_json(WAR_ROOM_DIR / "latest.json", payload)
    history_name = f"{now_jst().date().isoformat()}-{payload['current_session']['session_type']}.json"
    write_json(WAR_ROOM_DIR / "history" / history_name, payload)
    prune_history()
    print(
        "Built AI Arena Live Council "
        f"session={payload['current_session']['session_type']} "
        f"messages={payload['metrics']['message_count']} "
        f"topics={payload['metrics']['topic_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
