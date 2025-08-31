#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, ChoiceLoader, Template

OUT_DIR = os.getenv("OUT_DIR", "site")
REPORT_DATE = os.getenv("REPORT_DATE")  # 例: 2025-08-29
TEMPLATE_NAME = "daily.html.j2"
# 明示的にテンプレート配置を変えたい場合は TEMPLATE_DIR を指定（例: templates）
TEMPLATE_DIR = os.getenv("TEMPLATE_DIR")

# ---- 内蔵フォールバック用テンプレート（最後の砦） ----
BUILTIN_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Daily Report — {{ report_date }}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root { --fg:#111; --muted:#666; --bg:#fff; --card:#f7f7f9; }
    html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"; }
    header{ padding:24px 16px; border-bottom:1px solid #eee;}
    .container{ max-width:1100px; margin:0 auto; padding:0 16px;}
    h1{font-size:28px; margin:0 0 6px;}
    .meta{color:var(--muted); font-size:14px;}
    .grid{ display:grid; grid-template-columns: repeat(auto-fill,minmax(280px,1fr)); gap:16px; margin:24px 0 40px;}
    .card{ background:var(--card); border:1px solid #eee; border-radius:14px; padding:14px;}
    .rank{font-weight:700; font-size:13px; color:#444;}
    .sym{font-weight:800; font-size:20px; letter-spacing:.3px;}
    .name{color:var(--muted); font-size:13px;}
    .score{margin-top:6px; font-size:14px;}
    .comp{margin-top:8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:12px;}
    img{width:100%; height:auto; border-radius:10px; display:block; margin-top:8px;}
    footer{border-top:1px solid #eee; padding:16px; color:var(--muted); font-size:12px; text-align:center;}
  </style>
</head>
<body>
  <header>
    <div class="container">
      <h1>Daily Report</h1>
      <div class="meta">Report date (ET market date): <strong>{{ report_date }}</strong> — Universe size: {{ universe_size }}</div>
    </div>
  </header>

  <main class="container">
    {% if top10|length == 0 %}
      <p>No Top10 items were generated.</p>
    {% else %}
      <div class="grid">
        {% for r in top10 %}
        <article class="card">
          <div class="rank">#{{ r.rank }}</div>
          <div class="sym">{{ r.symbol }}</div>
          {% if r.name %}<div class="name">{{ r.name }}</div>{% endif %}
          <div class="score">Score: <strong>{{ r.score_pts }}</strong> ({{ "%.3f"|format(r.final_score_0_1) }})</div>
          <div class="comp">
            vol={{ "%.2f"|format(r.vol_anomaly_score) }},
            trends={{ "%.2f"|format(r.trends_breakout) }},
            news={{ "%.2f"|format(r.news_score) }},
            dii={{ "%.2f"|format(r.dii_score) }}
          </div>
          {% if r.chart_url %}
            <img loading="lazy" src="{{ r.chart_url }}" alt="{{ r.symbol }} weekly chart">
          {% endif %}
          <div class="comp">Δ1D={{ r.price_delta_1d if r.price_delta_1d is not none else "-" }}%
            / Δ1W={{ r.price_delta_1w if r.price_delta_1w is not none else "-" }}%
            / Δ1M={{ r.price_delta_1m if r.price_delta_1m is not none else "-" }}%</div>
        </article>
        {% endfor %}
      </div>
    {% endif %}
  </main>

  <footer>
    Generated {{ generated_at }} (server time)
  </footer>
</body>
</html>
"""

def _resolve_report_date():
    if REPORT_DATE:
        return REPORT_DATE
    # 念のため fallback（ET 18:00 前は前営業日）
    now_et = datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    if now_et.hour < 18:
        from datetime import timedelta
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
    return d.isoformat()

def _load_top10(date_iso: str):
    p = pathlib.Path(OUT_DIR) / "data" / date_iso / "top10.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []

def _guess_universe_size(date_iso: str, top10_len: int):
    # universe.csv を読みにいってみる
    uni_csv = pathlib.Path("data/universe.csv")
    if uni_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(uni_csv)
            if "symbol" in df.columns:
                return int(df["symbol"].nunique())
        except Exception:
            pass
    # 分からなければ top10 長さを返す（最低限）
    return top10_len

def _build_jinja_env():
    search_paths = []
    # 優先: ユーザーが指定した TEMPLATE_DIR
    if TEMPLATE_DIR:
        search_paths.append(TEMPLATE_DIR)
    # よくある場所
    search_paths += [
        "templates",
        ".",               # リポジトリ直下
        "scripts",         # 以前の実装互換
    ]
    existing = [p for p in search_paths if pathlib.Path(p).exists()]
    loaders = [FileSystemLoader(p) for p in existing]
    if not loaders:
        # 物理テンプレートが無い場合でも env 自体は返す
        return Environment(loader=ChoiceLoader([]), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return Environment(loader=ChoiceLoader(loaders), autoescape=False, trim_blocks=True, lstrip_blocks=True)

def main():
    date_iso = _resolve_report_date()

    # 入力
    top10 = _load_top10(date_iso)
    universe_size = _guess_universe_size(date_iso, len(top10))

    # 出力先準備
    out_dir = pathlib.Path(OUT_DIR) / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_iso}.html"

    # テンプレート解決
    env = _build_jinja_env()
    html = None
    try:
        tmpl = env.get_template(TEMPLATE_NAME)
        html = tmpl.render(
            report_date=date_iso,
            top10=top10,
            universe_size=universe_size,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )
    except Exception as e:
        # 最後の砦：内蔵テンプレートで描画（TemplateNotFound でも確実に成功する）
        sys.stderr.write(f"[WARN] Falling back to builtin template: {e}\n")
        tmpl = Template(BUILTIN_TEMPLATE)
        html = tmpl.render(
            report_date=date_iso,
            top10=top10,
            universe_size=universe_size,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )

    out_file.write_text(html, encoding="utf-8")
    print(f"[RENDER] wrote: {out_file}")

if __name__ == "__main__":
    main()
