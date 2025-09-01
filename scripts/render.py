#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, pathlib, datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---- market date resolver (import from scripts/ or CWD) ----
try:
    # when executed as "python scripts/render.py" from repo root
    from scripts.et_market_date import get_effective_market_date  # type: ignore
except Exception:
    try:
        # when executed from scripts/ directly
        from et_market_date import get_effective_market_date  # type: ignore
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
TEMPLATE_NAME = os.getenv("DAILY_TEMPLATE", "daily.html.j2")

def _find_template_dir() -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    here = pathlib.Path(__file__).resolve().parent
    repo = here.parent
    candidates = [cwd, repo, repo / "templates", here, here / "templates"]
    for p in candidates:
        if (p / TEMPLATE_NAME).exists():
            return p
    tried = "\n".join(str(p / TEMPLATE_NAME) for p in candidates)
    raise SystemExit(f"TemplateNotFound: '{TEMPLATE_NAME}'\nTried:\n{tried}")

def main():
    tpl_dir = _find_template_dir()
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        enable_async=False, trim_blocks=True, lstrip_blocks=True,
    )
    tpl = env.get_template(TEMPLATE_NAME)

    data_path = pathlib.Path(OUT_DIR)/"data"/DATE_S/"top10.json"
    if not data_path.exists():
        raise SystemExit(f"Top10 json not found: {data_path}")
    top10 = json.loads(data_path.read_text())

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
    (daily_dir / f"{DATE_S}.html").write_text(html, encoding="utf-8")
    print("Rendered daily HTML:", (daily_dir / f"{DATE_S}.html"))

if __name__ == "__main__":
    main()
