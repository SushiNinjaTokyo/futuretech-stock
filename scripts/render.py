#!/usr/bin/env python3
import os, json, pathlib, datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

# 日付ET化
import datetime
from zoneinfo import ZoneInfo

def usa_market_date_now():
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    d = now_et.date()
    if now_et.hour < 18:
        d = d - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d = d - datetime.timedelta(days=1)
    return d

DATE = os.getenv("REPORT_DATE") or usa_market_date_now().isoformat()


OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = os.getenv("REPORT_DATE") or datetime.date.today().isoformat()

def main():
    env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape())
    tpl = env.get_template("daily.html.j2")
    data_path = pathlib.Path(OUT_DIR) / "data" / DATE / "top10.json"
    top10 = json.loads(data_path.read_text()) if data_path.exists() else []
    schema_json = {
      "@context":"https://schema.org",
      "@type":"NewsArticle",
      "headline": f"Top 10 — {DATE}",
      "datePublished": DATE,
      "about":["stocks","AI","robotics","space"]
    }
    html = tpl.render(date=DATE, top10=top10, schema_json=json.dumps(schema_json))
    daily_dir = pathlib.Path(OUT_DIR) / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / f"{DATE}.html").write_text(html)
    print("Rendered daily HTML:", (daily_dir / f"{DATE}.html"))

if __name__ == "__main__":
    main()
