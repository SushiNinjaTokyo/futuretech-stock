#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DII (dark/ATS-like) placeholder fetcher with graceful fallback

- DII_SOURCE:
  - "skip": 収集スキップ。latest.json があれば踏襲、なければ空として0スコア化
  - "finra_api": 将来の実装用。現状は noisy になるAPI直叩きを避け、latest.json にフォールバック
  - "latest_json": 既存 latest.json をそのまま採用
- 出力: site/data/dii/latest.json と site/data/<DATE>/dii.json
"""

import os, sys, json, pathlib, datetime

OUT_DIR      = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
DATE         = os.getenv("REPORT_DATE") or datetime.date.today().isoformat()
SRC          = (os.getenv("DII_SOURCE") or "skip").lower()

def load_json_safe(p, default):
    try:
        q = pathlib.Path(p)
        if q.exists():
            return json.loads(q.read_text())
    except Exception:
        pass
    return default

def save_json(p, obj):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(p).write_text(json.dumps(obj, indent=2))

def main():
    out_latest = pathlib.Path(OUT_DIR)/"data"/"dii"/"latest.json"
    out_daily  = pathlib.Path(OUT_DIR)/"data"/DATE/"dii.json"

    prev = load_json_safe(out_latest, {"items":{}}).get("items", {})
    items = {}

    if SRC in ("skip", "latest_json"):
        items = prev
        source_used = "latest_json" if prev else "empty"
    elif SRC == "finra_api":
        # 現時点: 安定ソース未提供のため noisy API を回避し、即時フォールバック
        items = prev
        source_used = "latest_json" if prev else "empty"
    else:
        items = prev
        source_used = "latest_json" if prev else "empty"

    # フォーマット保証（score_0_1 のみで十分／なければ0）
    norm = {}
    for sym, d in (items or {}).items():
        try:
            s = float(d.get("score_0_1", 0.0))
        except Exception:
            s = 0.0
        norm[sym] = {"score_0_1": max(0.0, min(1.0, s))}
        # 任意情報はあってもよい: ATS share ratioなど
        if "ats_share_ratio" in d:
            try:
                norm[sym]["ats_share_ratio"] = float(d["ats_share_ratio"])
            except Exception:
                pass

    payload = {"items": norm, "source": source_used, "date": DATE}
    save_json(out_latest, payload)
    save_json(out_daily, payload)
    print(f"[DII] saved: {out_latest} and {out_daily} (symbols={len(norm)}) source={source_used}")

if __name__ == "__main__":
    main()
