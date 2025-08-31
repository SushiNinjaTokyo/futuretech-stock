#!/usr/bin/env python3
import os, json
from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUT_DIR    = os.path.join(BASE_DIR, "site")
REPORT_DATE = os.environ.get("REPORT_DATE")  # e.g. 2025-08-29

def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_first_existing(paths):
    for p in paths:
        d = read_json(p)
        if d is not None:
            return d, p
    return {}, None

def normalize_top10(payload):
    """
    期待形式：
      { "items": [ {symbol, name, final_score_0_1, ...} ] }
    それ以外でも最低限動くように防御的に整形。
    """
    if not payload:
        return []
    items = payload.get("items")
    if isinstance(items, list):
        src = items
    elif isinstance(payload, list):
        src = payload
    else:
        return []

    norm = []
    for it in src:
        if isinstance(it, dict):
            # 数値は float 化（None は 0.0 に）
            def f(v, default=0.0):
                try:
                    return float(v)
                except Exception:
                    return default
            norm.append({
                "symbol": it.get("symbol",""),
                "name": it.get("name", it.get("symbol","")),
                "rank": it.get("rank"),  # なければテンプレ側で loop.index
                "final_score_0_1": f(it.get("final_score_0_1"), 0.0),
                "score_pts": int(round(f(it.get("final_score_0_1"),0.0) * 1000)),
                "vol_anomaly_score": f(it.get("vol_anom_0_1") or it.get("vol_anomaly_score"), 0.0),
                "trends_breakout": f(it.get("trends_0_1") or it.get("trends_breakout"), 0.0),
                "news_score": f(it.get("news_0_1") or it.get("news_score"), 0.0),
                "dii_0_1": f(it.get("dii_0_1"), 0.0),
                "price_delta_1d": it.get("pct_change_1d"),
                "price_delta_1w": it.get("pct_change_1w"),
                "price_delta_1m": it.get("pct_change_1m"),
                "chart_url": it.get("chart_url"),
                # Breakdown 用（なければ空で OK）
                "score_components": it.get("score_components", {}),
                "score_weights": it.get("score_weights", {}),
                # 詳細（ボリュームなど）
                "detail": it.get("detail", {}),
                "insider_momo": f(it.get("insider_momo"), 0.0),
                "news_recent_count": it.get("news_recent_count", 0),
            })
        elif isinstance(it, str):
            norm.append({
                "symbol": it, "name": it, "final_score_0_1": 0.0, "score_pts": 0,
                "vol_anomaly_score": 0.0, "trends_breakout": 0.0, "news_score": 0.0, "dii_0_1": 0.0,
                "price_delta_1d": None, "price_delta_1w": None, "price_delta_1m": None,
                "chart_url": None, "score_components": {}, "score_weights": {},
                "detail": {}, "insider_momo": 0.0, "news_recent_count": 0,
            })
    return norm

def main():
    os.makedirs(os.path.join(OUT_DIR, "daily"), exist_ok=True)

    # Top10 の探索場所を増やして確実に拾う
    top10_payload, top10_src = load_first_existing([
        os.path.join(OUT_DIR, f"data/top10/{REPORT_DATE}.json"),
        os.path.join(OUT_DIR, f"data/{REPORT_DATE}/top10.json"),
        os.path.join(OUT_DIR, "data/top10/latest.json"),
    ])
    if not top10_payload:
        print(f"[WARN] Missing data for 'top10': tried 3 paths. Using empty.")
    else:
        print(f"[INFO] Loaded top10 from {top10_src}")

    trends, trends_src = load_first_existing([
        os.path.join(OUT_DIR, f"data/{REPORT_DATE}/trends.json"),
        os.path.join(OUT_DIR, "data/trends/latest.json"),
    ])
    if trends_src: print(f"[INFO] Loaded trends from {trends_src}")

    news, news_src = load_first_existing([
        os.path.join(OUT_DIR, f"data/{REPORT_DATE}/news.json"),
        os.path.join(OUT_DIR, "data/news/latest.json"),
    ])
    if news_src: print(f"[INFO] Loaded news from {news_src}")

    dii, dii_src = load_first_existing([
        os.path.join(OUT_DIR, f"data/{REPORT_DATE}/dii.json"),
        os.path.join(OUT_DIR, "data/dii/latest.json"),
    ])
    if dii_src: print(f"[INFO] Loaded dii from {dii_src}")

    top10_items = normalize_top10(top10_payload)

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    # 誤ってテンプレに enumerate を書いても落ちない保険
    env.globals["enumerate"] = enumerate

    tmpl = env.get_template("daily.html.j2")
    try:
        html = tmpl.render(
            date=REPORT_DATE,    # ← テンプレは date を使う
            top10=top10_items,
            trends=trends or {},
            news=news or {},
            dii=dii or {},
        )
    except Exception as e:
        print(f"[FATAL] Jinja2 render failed: {e}")
        raise

    out_path = os.path.join(OUT_DIR, "daily", f"{REPORT_DATE}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()
