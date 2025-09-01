#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, pathlib, datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    from et_market_date import get_effective_market_date
except Exception:
    from zoneinfo import ZoneInfo
    def get_effective_market_date():
        now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
        d = now_et.date()
        if now_et.hour < 18:
            d -= datetime.timedelta(days=1)
        while d.weekday() >= 5:
            d -= datetime.timedelta(days=1)
        return d

OUT_DIR = os.getenv("OUT_DIR", "site")
DATE = (os.getenv("REPORT_DATE") and datetime.date.fromisoformat(os.getenv("REPORT_DATE"))) or get_effective_market_date()
DATE_S = DATE.isoformat()

def main():
    env = Environment(
        loader=FileSystemLoader("."),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        enable_async=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("daily.html.j2")

    # load data
    data_path = pathlib.Path(OUT_DIR)/"data"/DATE_S/"top10.json"
    if not data_path.exists():
        raise SystemExit(f"Top10 json not found: {data_path}")
    top10 = json.loads(data_path.read_text())

    # schema.org
    schema_json = {
      "@context":"https://schema.org",
      "@type":"NewsArticle",
      "headline": f"Top 10 — {DATE_S}",
      "datePublished": DATE_S,
      "about":["stocks","AI","robotics","space"]
    }

    html = tpl.render(date=DATE_S, top10=top10, schema_json=json.dumps(schema_json))
    daily_dir = pathlib.Path(OUT_DIR) / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / f"{DATE_S}.html").write_text(html)
    print("Rendered daily HTML:", (daily_dir / f"{DATE_S}.html"))

if __name__ == "__main__":
    main()
