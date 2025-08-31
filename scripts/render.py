#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render the daily HTML from Jinja2 template and JSON data.

- Finds 'daily.html.j2' robustly:
  1) env TEMPLATE_PATH (file or directory; comma-separated supported)
  2) scripts/templates/daily.html.j2
  3) scripts/daily.html.j2
  4) templates/daily.html.j2 (repo root)
  5) daily.html.j2 (cwd)

- Fails fast if the template file is not found (prevents publishing a broken plain page).
- Loads data from site/data/{kind}/{REPORT_DATE}.json, falling back to site/data/{kind}/latest.json.
- Writes to site/daily/{REPORT_DATE}.html

Usage:
  REPORT_DATE=2025-08-29 python scripts/render.py
  python scripts/render.py --date 2025-08-29
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
from typing import Any, Dict, Optional, Tuple, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---------- paths / constants ----------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = REPO_ROOT / "site" / "daily"

TEMPLATE_NAME = "daily.html.j2"

# Data kinds we try to load. Adjust keys to what your template expects.
DATA_SOURCES: List[Tuple[str, str]] = [
    ("trends", "trends"),   # site/data/trends/{date}.json (or latest.json)
    ("news", "news"),       # site/data/news/{date}.json
    ("dii", "dii"),         # site/data/dii/{date}.json
    ("top10", "top10"),     # site/data/top10/{date}.json  (if your fetch_daily.py writes this)
]

# ---------- utils ----------

def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", dest="report_date", type=str, default=os.environ.get("REPORT_DATE", "").strip(),
                   help="Report date YYYY-MM-DD (or via env REPORT_DATE)")
    p.add_argument("--out", dest="out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="Output directory for HTML (default: site/daily)")
    p.add_argument("--template", dest="template_override", type=str, default=os.environ.get("TEMPLATE_PATH", "").strip(),
                   help="Template file or directory (or comma-separated list). Env TEMPLATE_PATH also supported.")
    return p.parse_args()

def validate_date_str(date_str: str) -> None:
    # Very light validation: YYYY-MM-DD
    import re
    if not date_str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        raise SystemExit(f"[FATAL] REPORT_DATE is invalid or missing: '{date_str}'. Provide via env REPORT_DATE or --date YYYY-MM-DD.")

def split_paths(env_value: str) -> List[pathlib.Path]:
    if not env_value:
        return []
    parts = [s.strip() for s in env_value.split(",") if s.strip()]
    return [pathlib.Path(p).resolve() for p in parts]

def candidate_template_locations(template_override: str) -> List[pathlib.Path]:
    """
    Returns candidate full file paths to the template.
    Accepts files or directories in template_override (comma-separated).
    """
    candidates: List[pathlib.Path] = []

    # 1) Overrides from CLI/env: can be a file OR directory (or list)
    for p in split_paths(template_override):
        if p.is_file():
            candidates.append(p)
        elif p.is_dir():
            candidates.append(p / TEMPLATE_NAME)

    # 2) Conventional locations (in order)
    candidates += [
        SCRIPT_DIR / "templates" / TEMPLATE_NAME,
        SCRIPT_DIR / TEMPLATE_NAME,
        REPO_ROOT / "templates" / TEMPLATE_NAME,
        pathlib.Path.cwd() / TEMPLATE_NAME,
    ]

    # De-duplicate keeping order
    seen = set()
    uniq: List[pathlib.Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq

def find_template_file(template_override: str) -> pathlib.Path:
    for cand in candidate_template_locations(template_override):
        if cand.exists():
            print(f"[INFO] Using template: {cand}")
            return cand
    msg_lines = ["[FATAL] Jinja2 template not found. Searched:"]
    for cand in candidate_template_locations(template_override):
        msg_lines.append(f" - {cand}")
    msg_lines.append("")
    msg_lines.append("Hint:")
    msg_lines.append(" - Ensure 'scripts/daily.html.j2' is committed to the repo")
    msg_lines.append(" - Or set env TEMPLATE_PATH to the file or directory containing the template")
    raise SystemExit("\n".join(msg_lines))

def load_json_with_fallback(kind: str, date_str: str) -> Any:
    """
    Try site/data/{kind}/{date}.json, then site/data/{kind}/latest.json.
    Returns parsed JSON or an empty structure if both miss.
    """
    base_dir = REPO_ROOT / "site" / "data" / kind
    primary = base_dir / f"{date_str}.json"
    fallback = base_dir / "latest.json"

    def load(path: pathlib.Path) -> Optional[Any]:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as ex:
                eprint(f"[WARN] Failed to parse JSON: {path} :: {ex}")
        return None

    data = load(primary)
    if data is not None:
        print(f"[INFO] Loaded {kind} from {primary}")
        return data

    data = load(fallback)
    if data is not None:
        print(f"[INFO] Loaded {kind} from {fallback}")
        return data

    eprint(f"[WARN] Missing data for '{kind}': tried {primary} and {fallback}. Using empty.")
    # Reasonable empty defaults (adjust to your template’s expectations)
    if kind == "trends":
        return {"items": []}
    if kind == "news":
        return {"items": []}
    if kind == "dii":
        return {"items": []}
    if kind == "top10":
        return {"symbols": [], "detail": {}}
    return {}

def build_jinja_env(template_file: pathlib.Path) -> Environment:
    # Loader must point to the directory containing the template
    loader_dir = template_file.parent
    env = Environment(
        loader=FileSystemLoader([loader_dir]),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # You can add filters here if your template uses them.
    return env

def ensure_out_dir(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

# ---------- main ----------

def main() -> None:
    args = parse_args()
    report_date = args.report_date
    validate_date_str(report_date)

    # Resolve template
    template_file = find_template_file(args.template_override)
    env = build_jinja_env(template_file)
    try:
        tmpl = env.get_template(template_file.name)
    except Exception as ex:
        raise SystemExit(f"[FATAL] Failed to load template '{template_file}': {ex}")

    # Load all data kinds with fallback
    context: Dict[str, Any] = {"report_date": report_date}
    for key, kind in DATA_SOURCES:
        context[key] = load_json_with_fallback(kind, report_date)

    # The template may rely on convenience keys — provide some sane defaults here
    # If your template expects flattened arrays, derive them here (non-breaking).
    # Example: extract top10 list if structure is {symbols: [...]}
    if isinstance(context.get("top10"), dict):
        top10_dict = context["top10"]
        context["top10_list"] = top10_dict.get("symbols", [])  # for {% for s in top10_list %}
        context["top10_detail"] = top10_dict.get("detail", {}) # map symbol->metrics

    out_dir = pathlib.Path(args.out_dir).resolve()
    ensure_out_dir(out_dir)
    out_file = out_dir / f"{report_date}.html"

    # Render
    try:
        html = tmpl.render(**context)
    except Exception as ex:
        raise SystemExit(f"[FATAL] Jinja2 render failed: {ex}")

    # Write
    try:
        with out_file.open("w", encoding="utf-8") as f:
            f.write(html)
    except Exception as ex:
        raise SystemExit(f"[FATAL] Failed to write HTML to {out_file}: {ex}")

    print(f"[INFO] Rendered daily HTML: {out_file}")

if __name__ == "__main__":
    main()
