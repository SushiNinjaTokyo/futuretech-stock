#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render daily Top10 HTML.
- Input JSONs (robust to slight schema differences):
    site/data/top10/{DATE}.json
    site/data/{DATE}/trends.json
    site/data/{DATE}/news.json
    site/data/{DATE}/dii.json  or site/data/dii/latest.json (ページ側でfetch)
- Output:
    site/daily/{DATE}.html
- Jinja template:
    templates/daily.html.j2
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "site" / "data"
DAILY_DIR = ROOT / "site" / "daily"
TEMPLATES_DIR = ROOT / "templates"

# -----------------------------
# Helpers
# -----------------------------
def read_json(p: Path) -> Any:
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def clamp01(x: Any) -> float:
    v = to_float(x, 0.0)
    return max(0.0, min(1.0, v))

def first(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def norm_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0:
        # fallback: equal weights (avoid zero division)
        keys = list(w.keys()) or ["vol_anomaly", "dii", "trends", "news"]
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    return {k: max(0.0, float(v)) / s for k, v in w.items()}

def read_env_weights() -> Dict[str, float]:
    # read from env (logs show these are set)
    return {
        "vol_anomaly": to_float(os.getenv("WEIGHT_VOL_ANOM"), 0.25),
        "dii":         to_float(os.getenv("WEIGHT_DII"), 0.25),
        "trends":      to_float(os.getenv("WEIGHT_TRENDS"), 0.30),
        "news":        to_float(os.getenv("WEIGHT_NEWS"), 0.20),
    }

def guess_date_from_env() -> str:
    d = os.getenv("REPORT_DATE")
    if d:
        return d
    # fallback: latest folder name under site/data/YYYY-MM-DD
    candidates = sorted((DATA_DIR.glob("20??-??-??")), reverse=True)
    return candidates[0].name if candidates else "1970-01-01"

# -----------------------------
# Unify a single item to template schema
# -----------------------------
def unify_item(raw: Dict[str, Any], idx: int, date_str: str, default_weights: Dict[str, float]) -> Dict[str, Any]:
    sym = raw.get("symbol") or raw.get("ticker") or raw.get("sym") or ""
    name = raw.get("name") or raw.get("company") or raw.get("title") or ""

    # final score (0..1) and points
    score01 = first(
        raw.get("final_score_0_1"),
        raw.get("score_norm"),
        (raw.get("final") or {}).get("score_0_1"),
        default=0.0
    )
    score01 = clamp01(score01)
    score_pts = int(round(score01 * 1000))

    # rank
    rank = int(first(raw.get("rank"), idx + 1))

    # components (0..1)
    comps_candidates = raw.get("score_components") or raw.get("components") or {}
    comps: Dict[str, float] = {
        # volume anomaly
        "vol_anomaly": clamp01(first(
            comps_candidates.get("vol_anomaly"),
            raw.get("vol_anomaly_score"),
            raw.get("volume_anomaly"),
            (raw.get("detail") or {}).get("vol_anomaly_score"),
            default=0.0
        )),
        # DII (replacing Insider)
        "dii": clamp01(first(
            comps_candidates.get("dii"),
            raw.get("dii_score"),
            raw.get("insider_momo"),
            raw.get("form4_score"),
            default=0.0
        )),
        # trends
        "trends": clamp01(first(
            comps_candidates.get("trends"),
            raw.get("trends_breakout"),
            default=0.0
        )),
        # news
        "news": clamp01(first(
            comps_candidates.get("news"),
            raw.get("news_score"),
            default=0.0
        )),
    }

    # weights
    weights_raw = raw.get("score_weights") or raw.get("weights") or {}
    weights: Dict[str, float] = {
        "vol_anomaly": to_float(first(weights_raw.get("vol_anomaly"),
                                      weights_raw.get("vol_anomaly_score"),
                                      default_weights.get("vol_anomaly"),)),
        "dii":         to_float(first(weights_raw.get("dii"),
                                      weights_raw.get("insider"),
                                      default_weights.get("dii"),)),
        "trends":      to_float(first(weights_raw.get("trends"),
                                      weights_raw.get("trends_breakout"),
                                      default_weights.get("trends"),)),
        "news":        to_float(first(weights_raw.get("news"),
                                      weights_raw.get("news_score"),
                                      default_weights.get("news"),)),
    }
    weights = norm_weights(weights)

    # price deltas
    d1  = first(raw.get("price_delta_1d"),  raw.get("delta_1d"),  raw.get("d1"),  default=None)
    d5  = first(raw.get("price_delta_1w"),  raw.get("delta_1w"),  raw.get("d5"),  raw.get("price_delta_5d"),  default=None)
    d20 = first(raw.get("price_delta_1m"),  raw.get("delta_1m"),  raw.get("d20"), raw.get("price_delta_20d"), default=None)

    # detail.vol_anomaly
    vol_detail = None
    det = raw.get("detail") or {}
    if "vol_anomaly" in det:
        vol_detail = det.get("vol_anomaly")
    else:
        # build minimal struct if pieces exist
        rvol20 = first(det.get("rvol20"), det.get("rvol_20"))
        z60    = first(det.get("z60"), det.get("zscore_60"))
        pr90   = first(det.get("pct_rank_90"), det.get("pr90"))
        dv     = first(det.get("dollar_vol"), det.get("dvol"))
        elig   = first(det.get("eligible"), det.get("eligibility"), None)
        if any(v is not None for v in [rvol20, z60, pr90, dv, elig]):
            vol_detail = {"rvol20": rvol20, "z60": z60, "pct_rank_90": pr90, "dollar_vol": dv, "eligible": elig}

    # chart url (relative from site root)
    chart_url = raw.get("chart_url") or f"/charts/{date_str}/{sym}.png"

    # expose top-level fields the template expects (plus fallbacks)
    out = {
        "symbol": sym,
        "name": name,
        "rank": rank,
        "final_score_0_1": score01,
        "score_pts": score_pts,
        "score_components": comps,
        "score_weights": weights,
        "trends_breakout": comps["trends"],
        "news_score": comps["news"],
        "dii_score": comps["dii"],
        "vol_anomaly_score": comps["vol_anomaly"],
        "chart_url": chart_url,
        "price_delta_1d": d1,
        "price_delta_1w": d5,
        "price_delta_1m": d20,
        "detail": {"vol_anomaly": vol_detail} if vol_detail else {},
    }
    return out

# -----------------------------
# Load Top10 source
# -----------------------------
def load_top10(date_str: str) -> List[Dict[str, Any]]:
    p = DATA_DIR / "top10" / f"{date_str}.json"
    j = read_json(p)
    if j is None:
        return []
    if isinstance(j, dict):
        # possible keys: items / top10 / data
        items = j.get("items") or j.get("top10") or j.get("data") or []
    elif isinstance(j, list):
        items = j
    else:
        items = []
    return items

# -----------------------------
# Main
# -----------------------------
def main():
    date_str = guess_date_from_env()

    raw_items = load_top10(date_str)
    if not raw_items:
        print(f"[WARN] No top10 data for {date_str} ({DATA_DIR/'top10'/f'{date_str}.json'})", file=sys.stderr)

    env_weights = read_env_weights()
    unified: List[Dict[str, Any]] = [
        unify_item(raw, i, date_str, env_weights) for i, raw in enumerate(raw_items)
    ]

    # Prepare Jinja environment
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("daily.html.j2")

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DAILY_DIR / f"{date_str}.html"

    html = tmpl.render(
        date=date_str,
        top10=unified,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()
