#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE")
DII_SOURCE = os.getenv("DII_SOURCE", "skip").strip().lower()


def log(level: str, msg: str) -> None:
    from datetime import datetime, timezone
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return None


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_universe() -> List[Dict[str, str]]:
    if not UNIVERSE_CSV.exists():
        log("WARN", f"Universe CSV missing: {UNIVERSE_CSV}")
        return []
    df = pd.read_csv(UNIVERSE_CSV)
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol", list(df.columns)[0])
    name_col = cols.get("name")
    items: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip().upper()
        if not sym:
            continue
        nm = str(row.get(name_col, "")).strip() if name_col else ""
        items.append({"symbol": sym, "name": nm})
    return items


def extract_form4_map() -> Dict[str, float]:
    """
    fetch_form4.py の latest.json があればそれを DII に流用する。
    無ければ空。
    """
    latest = OUT_DIR / "data" / "form4" / "latest.json"
    j = read_json(latest)
    if not j:
        return {}
    payload = j.get("items", j) if isinstance(j, dict) else j
    out: Dict[str, float] = {}
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol", "")).upper()
            if not sym:
                continue
            try:
                score = float(row.get("score_0_1", 0.0))
            except Exception:
                score = 0.0
            out[sym] = max(0.0, min(1.0, score))
    return out


def build_items() -> List[Dict[str, Any]]:
    universe = load_universe()
    form4_map = extract_form4_map() if DII_SOURCE in {"form4", "auto", "finra_api"} else {}

    items: List[Dict[str, Any]] = []
    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")
        score = float(form4_map.get(sym, 0.0))
        items.append({
            "symbol": sym,
            "name": nm,
            "score_0_1": round(score, 6),
            "source": "form4" if sym in form4_map else "neutral",
        })
    return items


def main() -> None:
    if not REPORT_DATE:
        raise SystemExit("REPORT_DATE is required")

    items = build_items()
    payload = {"date": REPORT_DATE, "items": items}

    out_day = OUT_DIR / "data" / REPORT_DATE / "dii.json"
    out_latest = OUT_DIR / "data" / "dii" / "latest.json"
    write_json(out_day, payload)
    write_json(out_latest, payload)
    log("INFO", f"Wrote DII: {out_day} ({len(items)} items)")


if __name__ == "__main__":
    main()
