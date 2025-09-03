#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, logging, shutil, time
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("render")

ROOT   = Path(".")
OUT    = Path("site")
REPORT = os.getenv("REPORT_DATE") or "latest"

def resolve_template() -> tuple[Path, Environment]:
    cand_dirs = [Path("templates"), ROOT]
    tpl_name = "daily.html.j2"
    for d in cand_dirs:
        p = d / tpl_name
        if p.exists():
            env = Environment(
                loader=FileSystemLoader(str(d)),
                autoescape=select_autoescape(['html','xml']),
                enable_async=False, trim_blocks=True, lstrip_blocks=True,
            )
            log.info("[TPL] using file: %s", p)
            return p, env
    raise FileNotFoundError("'daily.html.j2' not found in search path: %s" % ", ".join(map(str,cand_dirs)))

def copy_assets():
    src_css = Path("templates/daily.css")
    dst_dir = OUT / "assets"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_css.exists():
        shutil.copy2(src_css, dst_dir / "daily.css")
        log.info("[ASSET] copied CSS -> %s", dst_dir / "daily.css")
    else:
        log.warning("[ASSET] templates/daily.css not found; page will be unstyled")

def load_top10(date: str) -> dict:
    """
    Returns unified dict: { 'date': str, 'top10': list[dict] }
    Accepts historical path (site/data/<date>/top10.json) and 'latest' alias.
    Tolerates both a list JSON and an object JSON with {date, top10|items}.
    """
    if date == "latest":
        # prefer symlink/alias
        guess = OUT / "data" / "top10" / "latest.json"
        if guess.exists():
            p = guess
        else:
            # fallback: latest daily folder by mtime
            data_dir = OUT / "data"
            dated = sorted(
                [d for d in data_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if not dated:
                raise FileNotFoundError("No dated folders under site/data/")
            p = dated[0] / "top10.json"
    else:
        p = OUT / "data" / date / "top10.json"

    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")

    raw = json.load(open(p, "r", encoding="utf-8"))

    # Normalize shapes:
    #  - list -> assume it's already the items
    #  - dict -> try keys: 'top10' or 'items'
    if isinstance(raw, list):
        top10 = raw
        dstr = date
    elif isinstance(raw, dict):
        if "top10" in raw and isinstance(raw["top10"], list):
            top10 = raw["top10"]
        elif "items" in raw and isinstance(raw["items"], list):
            top10 = raw["items"]
        else:
            # maybe it's already a flat array stored under another key; try best
            top10 = next((v for v in raw.values() if isinstance(v, list)), [])
        dstr = str(raw.get("date") or date)
    else:
        raise TypeError(f"Unsupported JSON root type for {p}: {type(raw)}")

    # guarantee ints for score_pts if present
    for it in top10:
        if "score_pts" in it and isinstance(it["score_pts"], (float, int)):
            it["score_pts"] = int(round(it["score_pts"]))
        # alias compatibilities
        if "rank_points" in it and "score_pts" not in it:
            it["score_pts"] = int(round(it["rank_points"]))

    return {"date": dstr, "top10": top10}

def build_schema(data: dict) -> str:
    # defensive: itemListElement must align with present items
    items = data.get("top10", [])
    date  = data.get("date") or ""
    schema = {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "name": f"Daily Top 10 — {date}",
        "itemListElement": [
            {
                "@type":"ListItem",
                "position": i+1,
                "name": it.get("symbol") or it.get("name") or f"rank{i+1}",
                "url": f"/daily/{date}.html#card-{i+1}"
            } for i, it in enumerate(items)
        ]
    }
    return json.dumps(schema, ensure_ascii=False)

def main():
    t0 = time.time()
    tpl_path, env = resolve_template()
    data = load_top10(REPORT)
    schema_json = build_schema(data)
    tpl  = env.get_template(tpl_path.name)

    html = tpl.render(
        date=data["date"],
        top10=data["top10"],
        schema_json=schema_json
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"daily").mkdir(parents=True, exist_ok=True)
    outp = OUT/"daily"/f"{data['date']}.html"
    outp.write_text(html, encoding="utf-8")
    log.info("Rendered daily HTML: %s (template=%s)", outp, tpl_path)

    copy_assets()
    log.info("[TIME] render_total: %.3fs", time.time()-t0)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log.error("FATAL in render: %s", e, exc_info=True)
        sys.exit(1)
