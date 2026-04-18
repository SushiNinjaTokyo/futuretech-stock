#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import pandas as pd
import requests


OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE") or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
SEC_UA = os.getenv("SEC_USER_AGENT", "futuretech-stock/1.0 (contact: example@example.com)")
FORM4_MAX_FILINGS = int(os.getenv("FORM4_MAX_FILINGS", "8"))
FORM4_MAX_AGE_DAYS = int(os.getenv("FORM4_MAX_AGE_DAYS", "95"))
SEC_SLEEP = float(os.getenv("SEC_SLEEP", "0.7"))

CACHE_DIR = OUT_DIR / "cache"
CIK_CACHE = CACHE_DIR / "cik_map.json"
COMPANY_CACHE = CACHE_DIR / "sec_company_tickers.json"


def log(level: str, msg: str) -> None:
    from datetime import timezone
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_UA,
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_universe() -> List[Dict[str, str]]:
    df = pd.read_csv(UNIVERSE_CSV)
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol", list(df.columns)[0])
    name_col = cols.get("name")
    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip().upper()
        if not sym:
            continue
        nm = str(row.get(name_col, "")).strip() if name_col else ""
        out.append({"symbol": sym, "name": nm})
    return out


def load_company_tickers() -> Dict[str, int]:
    cache = read_json(COMPANY_CACHE)
    if cache:
        return {str(k).upper(): int(v) for k, v in cache.items()}

    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=headers(), timeout=30)
    r.raise_for_status()
    data = r.json()

    mapping: Dict[str, int] = {}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).strip().upper()
        cik = rec.get("cik_str")
        if t and cik:
            mapping[t] = int(cik)

    write_json(COMPANY_CACHE, mapping)
    return mapping


def search_cik_html(sym: str) -> Optional[int]:
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={quote_plus(sym)}&owner=exclude&action=getcompany"
    try:
        r = requests.get(url, headers=headers(), timeout=30)
        r.raise_for_status()
        m = re.search(r"CIK=0*([0-9]{1,10})", r.text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def resolve_cik(sym: str) -> Optional[str]:
    cache = read_json(CIK_CACHE)
    if sym in cache:
        return str(int(cache[sym])).zfill(10)

    company = load_company_tickers()
    alts = {sym}
    if "." in sym:
        alts.add(sym.replace(".", "-"))
    if "-" in sym:
        alts.add(sym.replace("-", "."))

    cik_num: Optional[int] = None
    for alt in alts:
        if alt in company:
            cik_num = int(company[alt])
            break

    if cik_num is None:
        cik_num = search_cik_html(sym)

    if cik_num is None:
        return None

    cache[sym] = cik_num
    write_json(CIK_CACHE, cache)
    return str(cik_num).zfill(10)


def get_submissions(cik10: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = requests.get(url, headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def filing_xml_candidates(cik10: str, accession_no_dash: str, primary_doc: str) -> List[str]:
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accession_no_dash}"
    cands = []
    if primary_doc and primary_doc.lower().endswith(".xml"):
        cands.append(f"{base}/{primary_doc}")
    cands.append(f"{base}/primary_doc.xml")
    cands.append(f"{base}/form4.xml")
    cands.append(f"{base}/ownership.xml")
    return cands


def parse_text(root: ET.Element, tag_name: str) -> Optional[str]:
    for el in root.iter():
        if el.tag.split("}")[-1].lower() == tag_name.lower():
            txt = (el.text or "").strip()
            if txt:
                return txt
    return None


def to_float(x: Optional[str]) -> float:
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0


def parse_form4_xml(xml_text: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_text)
    out = {
        "date": parse_text(root, "periodOfReport") or parse_text(root, "transactionDate"),
        "buy_shares": 0.0,
        "sell_shares": 0.0,
        "buyers": [],
    }
    buyers = set()

    for tx in root.iter():
        tag = tx.tag.split("}")[-1].lower()
        if tag not in {"nonderivativetransaction", "derivativetransaction"}:
            continue

        code = ""
        shares = 0.0
        for ch in tx.iter():
            nm = ch.tag.split("}")[-1].lower()
            txt = (ch.text or "").strip()
            if nm == "transactioncode" and txt:
                code = txt.upper()
            elif nm in {"transactionshares", "shares"} and txt:
                shares = to_float(txt)

        if code == "P":
            out["buy_shares"] += shares
        elif code == "S":
            out["sell_shares"] += shares

    for ro in root.iter():
        if ro.tag.split("}")[-1].lower() == "rptownername":
            txt = (ro.text or "").strip()
            if txt:
                buyers.add(txt)

    out["buyers"] = sorted(buyers)
    return out


def fetch_recent_form4_events(sym: str, cik10: str, ref_date: date) -> List[Dict[str, Any]]:
    try:
        j = get_submissions(cik10)
    except Exception as e:
        log("WARN", f"{sym}: submissions fetch failed: {e}")
        return []

    recent = j.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    accs = recent.get("accessionNumber", []) or []
    docs = recent.get("primaryDocument", []) or []
    dates = recent.get("filingDate", []) or []

    events: List[Dict[str, Any]] = []
    for form, acc, doc, fdate in zip(forms, accs, docs, dates):
        if str(form).strip() != "4":
            continue
        try:
            filing_date = date.fromisoformat(str(fdate))
        except Exception:
            continue
        if (ref_date - filing_date).days > FORM4_MAX_AGE_DAYS:
            continue

        acc_no_dash = str(acc).replace("-", "").strip()
        xmls = filing_xml_candidates(cik10, acc_no_dash, str(doc))
        parsed: Optional[Dict[str, Any]] = None

        for u in xmls:
            try:
                r = requests.get(u, headers=headers(), timeout=30)
                if not r.ok:
                    continue
                txt = r.text
                if "<ownershipDocument" not in txt:
                    continue
                parsed = parse_form4_xml(txt)
                break
            except Exception:
                continue

        time.sleep(SEC_SLEEP)

        if parsed:
            events.append(parsed)
            if len(events) >= FORM4_MAX_FILINGS:
                break

    return events


def percentile_rank(values: List[float], x: float) -> float:
    pos = sorted(v for v in values if v > 0)
    if not pos or x <= 0:
        return 0.0
    le = sum(1 for v in pos if v <= x)
    return le / len(pos)


def main() -> None:
    ref_date = date.fromisoformat(REPORT_DATE)
    universe = load_universe()

    raw_items: List[Dict[str, Any]] = []
    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")
        cik10 = resolve_cik(sym)
        if not cik10:
            raw_items.append({
                "symbol": sym,
                "name": nm,
                "cik": None,
                "net_buy_shares_30": 0.0,
                "net_buy_shares_90": 0.0,
                "buyers_30": 0,
                "buyers_90": 0,
                "score_0_1": 0.0,
            })
            continue

        events = fetch_recent_form4_events(sym, cik10, ref_date)

        net30 = 0.0
        net90 = 0.0
        buyers30 = set()
        buyers90 = set()

        for ev in events:
            try:
                d = date.fromisoformat(str(ev.get("date")))
            except Exception:
                continue
            age = (ref_date - d).days
            net = float(ev.get("buy_shares", 0.0)) - float(ev.get("sell_shares", 0.0))
            if age <= 90:
                net90 += net
                buyers90.update(ev.get("buyers", []))
            if age <= 30:
                net30 += net
                buyers30.update(ev.get("buyers", []))

        raw_items.append({
            "symbol": sym,
            "name": nm,
            "cik": cik10,
            "net_buy_shares_30": round(net30, 2),
            "net_buy_shares_90": round(net90, 2),
            "buyers_30": len(buyers30),
            "buyers_90": len(buyers90),
        })

    vals = [max(0.0, float(x["net_buy_shares_30"])) for x in raw_items]
    items: List[Dict[str, Any]] = []
    for row in raw_items:
        score = percentile_rank(vals, max(0.0, float(row["net_buy_shares_30"])))
        items.append({
            **row,
            "score_0_1": round(score, 6),
        })

    payload = {"date": REPORT_DATE, "items": items}
    day_path = OUT_DIR / "data" / REPORT_DATE / "form4.json"
    latest_path = OUT_DIR / "data" / "form4" / "latest.json"
    write_json(day_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote Form4: {day_path} ({len(items)} items)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in fetch_form4: {e}")
        sys.exit(1)
