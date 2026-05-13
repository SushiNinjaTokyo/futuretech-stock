#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import rebuild_signal_outcomes as core  # noqa: E402


def pick_report_date() -> str:
    if core.os.getenv("REPORT_DATE"):
        return core.os.getenv("REPORT_DATE", "")
    latest = core.read_json(core.OUT_DIR / "data" / "top10" / "latest.json")
    if isinstance(latest, dict) and latest.get("date"):
        return str(latest["date"])
    data_dir = core.OUT_DIR / "data"
    dates = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and core.valid_date_dir_name(d.name) and (d / "top10.json").exists()], reverse=True)
    if not dates:
        raise SystemExit("No daily top10 data found")
    return dates[0]


def add_today_signals(registry: Dict[str, Any], date: str, top10: List[Dict[str, Any]]) -> int:
    existing = {str(s.get("id")) for s in registry.get("signals", []) if isinstance(s, dict)}
    added = 0
    for fallback_rank, item in enumerate(top10[:10], start=1):
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym:
            continue
        rank = core.get_item_rank(item, fallback_rank)
        eligible, rule, quality = core.evaluate_signal_eligibility(item, rank)
        if not eligible:
            continue
        sid = core.signal_id(date, sym)
        if sid in existing:
            continue
        registry.setdefault("signals", []).append(core.make_new_signal(item, date, rank, rule, quality))
        existing.add(sid)
        added += 1
    return added


def main() -> None:
    date = pick_report_date()
    core.log("INFO", f"Track Daily Event Lab outcomes for REPORT_DATE={date}")
    registry = core.load_registry()
    top10 = core.load_top10_for_date(date)
    added = add_today_signals(registry, date, top10)
    signals = [core.enrich_signal_defaults(s) for s in registry.get("signals", []) if isinstance(s, dict)]
    changed = core.update_all_signal_outcomes(signals)
    registry["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    registry["policy"] = core.current_policy()
    registry["signals"] = signals
    summary = core.build_summary(signals)
    recent = core.flatten_recent_outcomes(signals, limit=300)
    core.write_json(core.REGISTRY_PATH, registry)
    core.write_json(core.OUTCOMES_PATH, {"items": recent, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")})
    core.write_json(core.SUMMARY_PATH, summary)
    core.log("INFO", f"Added={added}, outcome_updated={changed}, reportable={summary.get('total_signals')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        core.log("ERROR", f"FATAL in track_signal_outcomes: {e}")
        sys.exit(1)
