#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("OUT_DIR", str(ROOT / "site")))
DAILY_V2_DIR = OUT_DIR / "data" / "daily-v2"
LATEST_JSON = DAILY_V2_DIR / "latest.json"
MANIFEST_JSON = DAILY_V2_DIR / "manifest.json"


def log(level: str, msg: str) -> None:
    print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}",
        flush=True,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log("WARN", f"read_json failed: {path}: {exc}")
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


def existing_daily_v2_dates() -> List[str]:
    if not DAILY_V2_DIR.exists():
        return []

    dates: List[str] = []
    for d in DAILY_V2_DIR.iterdir():
        if d.is_dir() and is_yyyy_mm_dd(d.name) and (d / "top10.json").exists():
            dates.append(d.name)

    return sorted(dates)


def main() -> None:
    dates = existing_daily_v2_dates()
    if not dates:
        raise SystemExit("No daily-v2 top10.json files found. Cannot sync latest.")

    latest_date = dates[-1]
    src = DAILY_V2_DIR / latest_date / "top10.json"
    payload = read_json(src)

    if not isinstance(payload, dict):
      raise SystemExit(f"Invalid latest source JSON: {src}")

    payload["date"] = payload.get("date") or latest_date
    payload["latest_synced_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    payload["latest_source"] = str(src.relative_to(ROOT))

    write_json(LATEST_JSON, payload)

    manifest = read_json(MANIFEST_JSON)
    if not isinstance(manifest, dict):
        manifest = {}

    manifest["version"] = manifest.get("version", "daily_event_score")
    manifest["latest_date"] = latest_date
    manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    manifest["date_count"] = len(dates)
    manifest["dates"] = dates

    write_json(MANIFEST_JSON, manifest)
    log("INFO", f"Synced daily-v2 latest.json -> {latest_date}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR", f"FATAL in sync_daily_v2_latest: {exc}")
        sys.exit(1)
