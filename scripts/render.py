#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render daily report HTML.

- 入力:
    ENV REPORT_DATE=YYYY-MM-DD   … 任意（未指定時は site/data/top10/*.json の最新日付）
    templates/daily.html.j2      … Jinja2 テンプレート
    site/data/top10/<DATE>.json  … メイン（fetch_daily.py の成果物）
    site/data/<DATE>/trends.json … 任意
    site/data/<DATE>/news.json   … 任意
    site/data/<DATE>/dii.json    … 任意

- 出力:
    site/daily/<DATE>.html

- 方針:
    * テンプレートの “ranking” / “top10” どちらでも動くように alias を渡す
    * 欠損フィールドはサイレントに既定値で補完（テンプレート側で KeyError を起こさない）
    * デザインはテンプレートに委譲（ここでは構造を崩さない）
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape


ROOT = Path(__file__).resolve().parents[1]  # repo root
SITE_DIR = ROOT / "site"
DATA_DIR = SITE_DIR / "data"
TOP10_DIR = DATA_DIR / "top10"
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = SITE_DIR / "daily"


# ---------- Utilities ----------

def load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def latest_date_from_top10() -> Optional[str]:
    if not TOP10_DIR.exists():
        return None
    dates: List[Tuple[str, Path]] = []
    for p in TOP10_DIR.glob("*.json"):
        m = re.match(r"(\d{4}-\d{2}-\d{2})\.json$", p.name)
        if m:
            dates.append((m.group(1), p))
    if not dates:
        return None
    dates.sort(key=lambda t: t[0])
    return dates[-1][0]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return v if v is not None else default


# ---------- Data shaping ----------

CORE_KEYS = ["vol_anomaly_score", "insider_momo", "trends_breakout", "news_score"]

def normalize_item(item: Dict[str, Any], rank_fallback: int, date_str: str) -> Dict[str, Any]:
    """テンプレートが必要とする最低限の形に正規化。未知キーは触らない。"""
    sym = safe_get(item, "symbol", "")
    name = safe_get(item, "name", "")
    rank = int(safe_get(item, "rank", rank_fallback))

    # スコア系
    final_0_1 = float(safe_get(item, "final_score_0_1", 0.0))
    score_pts = int(safe_get(item, "score_pts", round(final_0_1 * 1000)))

    # score_components が dict でなければ空 dict
    sc = safe_get(item, "score_components", {}) or {}
    if not isinstance(sc, dict):
        sc = {}

    # 代表キーが numeric になるようにキャスト（壊れてても落とさない）
    def to_num(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    for k in CORE_KEYS:
        if k in sc and sc[k] is not None:
            sc[k] = to_num(sc[k])

    # detail（出来高異常など）
    detail = safe_get(item, "detail", {}) or {}
    vol = detail.get("vol_anomaly") if isinstance(detail, dict) else None
    if not isinstance(detail, dict):
        detail = {}
    # 期待する形: detail.vol_anomaly.{is_anomaly, ratio, zscore ...}
    if vol is not None and not isinstance(vol, dict):
        detail["vol_anomaly"] = {}

    # チャート URL（テンプレート互換）
    chart_url = safe_get(item, "chart_url", f"/charts/{date_str}/{sym}.png")

    # rank 1..n に合わせて返す
    shaped = {
        **item,  # 既存のフィールドは温存
        "symbol": sym,
        "name": name,
        "rank": rank,
        "final_score_0_1": final_0_1,
        "score_pts": score_pts,
        "score_components": sc,
        "detail": detail,
        "chart_url": chart_url,
    }
    return shaped


# ---------- Main ----------

def main() -> None:
    # 1) 日付解決
    report_date = os.environ.get("REPORT_DATE", "").strip()
    if not report_date:
        report_date = latest_date_from_top10() or ""
    if not report_date:
        raise SystemExit("[ERROR] REPORT_DATE が解決できませんでした（site/data/top10/*.json も見つかりません）")

    # 2) データ読み取り（存在しなくても落とさない）
    top10_path = TOP10_DIR / f"{report_date}.json"
    top10_raw = load_json(top10_path, default={"items": []})
    if isinstance(top10_raw, dict) and "items" in top10_raw:
        items_in = top10_raw["items"]
    elif isinstance(top10_raw, list):
        items_in = top10_raw
    else:
        items_in = []

    # trends/news/dii は “存在すれば” 渡す（テンプレートでは使わなくてもOK）
    base_day_dir = DATA_DIR / report_date
    trends = load_json(base_day_dir / "trends.json", default={})
    news = load_json(base_day_dir / "news.json", default={})
    dii = load_json(base_day_dir / "dii.json", default={})

    # 3) 形を整える（rank, score, components, detail, chart_url）
    items_out: List[Dict[str, Any]] = []
    for idx, it in enumerate(items_in, start=1):
        try:
            items_out.append(normalize_item(it, rank_fallback=idx, date_str=report_date))
        except Exception:
            # 1件壊れていても全体は落とさない
            items_out.append(normalize_item({}, rank_fallback=idx, date_str=report_date))

    # 4) Jinja2 で出力
    ensure_dir(OUT_DIR)

    # テンプレートは “厳密すぎると欠損で落ちる” ので StrictUndefined は使わない
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("daily.html.j2")

    # スコアの説明文（デザイン文言はテンプレ側にもあるが、念のため渡す）
    scoring = "volume anomaly + DII + trends + news"

    html = tmpl.render(
        date=report_date,
        scoring=scoring,
        top10=items_out,     # 既存呼び名
        ranking=items_out,   # 互換エイリアス（テンプレがどちらでも動く）
        trends=trends,
        news=news,
        dii=dii,
    )

    out_path = OUT_DIR / f"{report_date}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
