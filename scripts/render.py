#!/usr/bin/env python3
import os, json
from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUT_DIR = os.path.join(BASE_DIR, "site")
REPORT_DATE = os.environ.get("REPORT_DATE")

def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_or_empty(paths):
    for p in paths:
        d = read_json(p)
        if d is not None:
            return d
    return {}

def normalize_top10(payload):
    # 期待形式: {"items":[{...}]}
    if not payload or "items" not in payload or not isinstance(payload["items"], list):
        return []
    items = []
    for it in payload["items"]:
        if isinstance(it, dict):
            items.append({
                "symbol": it.get("symbol",""),
                "name": it.get("name", it.get("symbol","")),
                "final_score_0_1": float(it.get("final_score_0_1", 0.0)),
                "vol_anom_0_1": float(it.get("vol_anom_0_1", 0.0)),
                "trends_0_1": float(it.get("trends_0_1", 0.0)),
                "news_0_1": float(it.get("news_0_1", 0.0)),
                "dii_0_1": float(it.get("dii_0_1", 0.0)),
                "price": it.get("price"),
                "pct_change_1d": it.get("pct_change_1d"),
            })
        elif isinstance(it, str):
            items.append({
                "symbol": it, "name": it,
                "final_score_0_1": 0.0,
                "vol_anom_0_1": 0.0, "trends_0_1": 0.0, "news_0_1": 0.0, "dii_0_1": 0.0,
                "price": None, "pct_change_1d": None
            })
    return items

def main():
    os.makedirs(os.path.join(OUT_DIR,"daily"), exist_ok=True)

    top10_payload = load_or_empty([
        os.path.join(OUT_DIR, f"data/top10/{REPORT_DATE}.json"),
        os.path.join(OUT_DIR, "data/top10/latest.json"),
    ])
    if not top10_payload:
        print(f"[WARN] Missing data for 'top10': tried {OUT_DIR}/data/top10/{REPORT_DATE}.json and {OUT_DIR}/data/top10/latest.json. Using empty.")

    trends = load_or_empty([
        os.path.join(OUT_DIR, f"data/{REPORT_DATE}/trends.json"),
        os.path.join(OUT_DIR, "data/trends/latest.json"),
    ])
    news = load_or_empty([
        os.path.join(OUT_DIR, f"data/{REPORT_DATE}/news.json"),
        os.path.join(OUT_DIR, "data/news/latest.json"),
    ])
    dii = load_or_empty([
        os.path.join(OUT_DIR, f"data/{REPORT_DATE}/dii.json"),
        os.path.join(OUT_DIR, "data/dii/latest.json"),
    ])

    top10_items = normalize_top10(top10_payload)

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"])
    )
    # 再発防止の保険（テンプレ側で誤って enumerate を書いても動く）
    env.globals["enumerate"] = enumerate

    print(f"[INFO] Using template: {os.path.join(TEMPLATE_DIR,'daily.html.j2')}")
    tmpl = env.get_template("daily.html.j2")

    try:
        html = tmpl.render(
            report_date=REPORT_DATE,
            top10=top10_items,
            trends=trends or {},
            news=news or {},
            dii=dii or {},
        )
    except Exception as e:
        print(f"[FATAL] Jinja2 render failed: {e}")
        raise

    out_path = os.path.join(OUT_DIR, "daily", f"{REPORT_DATE}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()
