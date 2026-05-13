#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
if not OUT_DIR.is_absolute():
    OUT_DIR = (ROOT / OUT_DIR).resolve()
else:
    OUT_DIR = OUT_DIR.resolve()

START_DATE = os.getenv("DAILY_V2_START_DATE", os.getenv("START_DATE", "")).strip()
END_DATE = os.getenv("DAILY_V2_END_DATE", os.getenv("END_DATE", "")).strip()
MODE = os.getenv("DAILY_V2_RANGE_MODE", "missing_only").strip().lower()
MAX_DAYS = int(os.getenv("DAILY_V2_MAX_DAYS_PER_RUN", "10") or "10")
SLEEP = float(os.getenv("DAILY_V2_RANGE_SLEEP_SECONDS", "5") or "5")

DAILY_V2_DIR = OUT_DIR / "data" / "daily-v2"
LATEST_JSON = DAILY_V2_DIR / "latest.json"
MANIFEST_JSON = DAILY_V2_DIR / "manifest.json"


def log(msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [INFO] {msg}",
        flush=True,
    )


def warn(msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [WARN] {msg}",
        flush=True,
    )


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
    except Exception as exc:
        warn(f"read_json failed: {path}: {exc}")
        return None


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def is_yyyy_mm_dd(s: str) -> bool:
    if len(s) != 10:
        return False
    try:
        pd.Timestamp(s)
        return s[4] == "-" and s[7] == "-"
    except Exception:
        return False


def trading_dates(start: str, end: str) -> List[str]:
    try:
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize() if end else pd.Timestamp.utcnow().normalize()
    except Exception as exc:
        raise SystemExit(f"Invalid date input: {exc}")

    if e < s:
        raise SystemExit(f"end_date must be >= start_date. start={s.date()}, end={e.date()}")

    return [d.strftime("%Y-%m-%d") for d in pd.date_range(s, e, freq="B")]


def existing_daily_v2_dates() -> List[str]:
    if not DAILY_V2_DIR.exists():
        return []

    dates: List[str] = []
    for d in DAILY_V2_DIR.iterdir():
        if not d.is_dir():
            continue
        if not is_yyyy_mm_dd(d.name):
            continue
        if (d / "top10.json").exists():
            dates.append(d.name)

    return sorted(dates)


def normalize_manifest(manifest: Any) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        manifest = {}

    raw_dates = manifest.get("dates")
    date_records = manifest.get("date_records")

    normalized_dates: Dict[str, Any] = {}

    if isinstance(raw_dates, dict):
        normalized_dates.update(raw_dates)
    elif isinstance(raw_dates, list):
        for d in raw_dates:
            if isinstance(d, str) and is_yyyy_mm_dd(d):
                normalized_dates[d] = {"status": "ok"}

    if isinstance(date_records, dict):
        for d, rec in date_records.items():
            if isinstance(d, str) and is_yyyy_mm_dd(d):
                normalized_dates[d] = rec if isinstance(rec, dict) else {"status": "ok"}

    manifest["dates"] = normalized_dates
    manifest["date_records"] = normalized_dates

    return manifest


def update_manifest(latest_date: str, available_dates: List[str]) -> None:
    manifest = normalize_manifest(read_json(MANIFEST_JSON))

    manifest["version"] = manifest.get("version", "daily_event_score")
    manifest["latest_date"] = latest_date
    manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    manifest["available_dates"] = available_dates
    manifest["date_count"] = len(available_dates)

    records = manifest.setdefault("date_records", {})
    dates_dict = manifest.setdefault("dates", {})

    for d in available_dates:
        records.setdefault(d, {"status": "ok"})
        dates_dict.setdefault(d, {"status": "ok"})

    if latest_date in records:
        records[latest_date]["status"] = "ok"
        records[latest_date]["latest"] = True

    if latest_date in dates_dict:
        dates_dict[latest_date]["status"] = "ok"
        dates_dict[latest_date]["latest"] = True

    write_json(MANIFEST_JSON, manifest)


def sync_latest_to_max_existing_date() -> Optional[str]:
    """
    API safety cap limits only newly built dates.
    /daily/ must always point to the max existing daily-v2 date,
    not the max date processed in this run.
    """
    dates = existing_daily_v2_dates()
    if not dates:
        warn("No daily-v2 date folders with top10.json found; latest.json not updated.")
        return None

    latest_date = dates[-1]
    src = DAILY_V2_DIR / latest_date / "top10.json"

    payload = read_json(src)
    if not isinstance(payload, dict):
        warn(f"Latest source is not a valid JSON object: {src}")
        return None

    payload["date"] = payload.get("date") or latest_date
    payload["latest_synced_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    payload["latest_source"] = safe_relative(src)

    write_json(LATEST_JSON, payload)
    update_manifest(latest_date, dates)

    log(f"Synced daily-v2 latest.json -> {latest_date}")
    log(f"latest_source={safe_relative(src)}")
    return latest_date


def run_script(script: str, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(
        [sys.executable, script],
        cwd=str(ROOT),
        env=env or os.environ.copy(),
        check=True,
    )


def render_daily_and_index() -> None:
    log("Render /daily/ from daily-v2 latest.json")
    run_script("scripts/render_daily_v2.py")

    render_index = ROOT / "scripts" / "render_index.py"
    if render_index.exists():
        log("Render / index page")
        run_script("scripts/render_index.py")
    else:
        warn("scripts/render_index.py not found; index page not rendered.")


def main() -> None:
    if not START_DATE:
        raise SystemExit("DAILY_V2_START_DATE is required")

    if MODE not in {"missing_only", "overwrite"}:
        raise SystemExit(f"Unsupported DAILY_V2_RANGE_MODE: {MODE}. Use missing_only or overwrite.")

    dates = trading_dates(START_DATE, END_DATE)

    selected: List[str] = []
    for d in dates:
        out = DAILY_V2_DIR / d / "top10.json"
        if MODE == "missing_only" and out.exists():
            continue
        selected.append(d)

    if MAX_DAYS > 0:
        selected = selected[:MAX_DAYS]

    log(
        "Daily v2 range build "
        f"requested={len(dates)}, selected={len(selected)}, mode={MODE}, max_days={MAX_DAYS}"
    )

    if selected:
        log(f"Selected dates={selected}")
    else:
        log("No new daily-v2 dates to build.")

    for i, d in enumerate(selected, 1):
        env = os.environ.copy()
        env["REPORT_DATE"] = d

        log(f"Build daily-v2 {i}/{len(selected)} {d}")
        run_script("scripts/build_daily_v2.py", env=env)

        if SLEEP > 0 and i < len(selected):
            time.sleep(SLEEP)

    latest_date = sync_latest_to_max_existing_date()
    if latest_date:
        render_daily_and_index()

    log("Daily v2 range build completed.")


if __name__ == "__main__":
    main()
