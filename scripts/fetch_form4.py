#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import pandas as pd
import requests

try:
    import numpy as np
except Exception:
    np = None


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
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} [{level}] {msg}", flush=True)


def headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_UA,
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log("WARN", f"read_json failed: {path}: {e}")
        return {}


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def load_universe() -> List[Dict[str, str]]:
    if not UNIVERSE_CSV.exists():
        log("ERROR", f"Universe CSV missing: {UNIVERSE_CSV}")
        return []

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


def session_get_json(sess: requests.Session, url: str, timeout: int = 30) -> Dict[str, Any]:
    r = sess.get(url, headers=headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def session_get_text(sess: requests.Session, url: str, timeout: int = 30) -> str:
    r = sess.get(url, headers=headers(), timeout=timeout)
    r.raise_for_status()
    return r.text


def load_company_tickers(sess: requests.Session) -> Dict[str, int]:
    cache = read_json(COMPANY_CACHE)
    if cache:
        try:
            return {str(k).upper(): int(v) for k, v in cache.items()}
        except Exception:
            pass

    url = "https://www.sec.gov/files/company_tickers.json"
    data = session_get_json(sess, url)

    mapping: Dict[str, int] = {}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).strip().upper()
        cik = rec.get("cik_str")
        if t and cik:
            try:
                mapping[t] = int(cik)
            except Exception:
                continue

    write_json(COMPANY_CACHE, mapping)
    return mapping


def search_cik_html(sess: requests.Session, sym: str) -> Optional[int]:
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={quote_plus(sym)}&owner=exclude&action=getcompany"
    try:
        html = session_get_text(sess, url)
        m = re.search(r"CIK=0*([0-9]{1,10})", html, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception as e:
        log("WARN", f"{sym}: search_cik_html failed: {e}")
    return None


def resolve_cik(sess: requests.Session, sym: str) -> Optional[str]:
    cache = read_json(CIK_CACHE)
    if sym in cache:
        try:
            return str(int(cache[sym])).zfill(10)
        except Exception:
            pass

    company = load_company_tickers(sess)
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
        cik_num = search_cik_html(sess, sym)

    if cik_num is None:
        return None

    cache[sym] = cik_num
    write_json(CIK_CACHE, cache)
    return str(cik_num).zfill(10)


def get_submissions(sess: requests.Session, cik10: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    return session_get_json(sess, url)


def filing_xml_candidates(cik10: str, accession_no_dash: str, primary_doc: str) -> List[str]:
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accession_no_dash}"
    cands: List[str] = []
    if primary_doc and primary_doc.lower().endswith(".xml"):
        cands.append(f"{base}/{primary_doc}")
    cands.append(f"{base}/primary_doc.xml")
    cands.append(f"{base}/form4.xml")
    cands.append(f"{base}/ownership.xml")
    return cands


def local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def element_text(el: ET.Element) -> Optional[str]:
    txt = (el.text or "").strip()
    if txt:
        return txt
    return None


def child_element(parent: ET.Element, name: str) -> Optional[ET.Element]:
    for ch in list(parent):
        if local_name(ch.tag).lower() == name.lower():
            return ch
    return None


def child_text(parent: ET.Element, name: str) -> Optional[str]:
    ch = child_element(parent, name)
    if ch is None:
        return None

    txt = element_text(ch)
    if txt:
        return txt

    val = child_element(ch, "value")
    if val is not None:
        vtxt = element_text(val)
        if vtxt:
            return vtxt

    return None


def nested_element(parent: ET.Element, path: List[str]) -> Optional[ET.Element]:
    cur = parent
    for name in path:
        nxt = child_element(cur, name)
        if nxt is None:
            return None
        cur = nxt
    return cur


def nested_text(parent: ET.Element, path: List[str]) -> Optional[str]:
    el = nested_element(parent, path)
    if el is None:
        return None

    txt = element_text(el)
    if txt:
        return txt

    val = child_element(el, "value")
    if val is not None:
        vtxt = element_text(val)
        if vtxt:
            return vtxt

    return None


def parse_text(root: ET.Element, tag_name: str) -> Optional[str]:
    for el in root.iter():
        if local_name(el.tag).lower() == tag_name.lower():
            txt = element_text(el)
            if txt:
                return txt
            val = child_element(el, "value")
            if val is not None:
                vtxt = element_text(val)
                if vtxt:
                    return vtxt
    return None


def to_float(x: Optional[str]) -> float:
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0


def parse_form4_xml(xml_text: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_text)

    period_of_report = parse_text(root, "periodOfReport")

    out: Dict[str, Any] = {
        "date": period_of_report,
        "buy_shares": 0.0,
        "sell_shares": 0.0,
        "buyers": [],
    }

    buyers = set()

    # owner names
    for ro in root.iter():
        if local_name(ro.tag).lower() == "rptownername":
            txt = element_text(ro)
            if txt:
                buyers.add(txt)

    # transactions
    for tx in root.iter():
        tag = local_name(tx.tag).lower()
        if tag not in {"nonderivativetransaction", "derivativetransaction"}:
            continue

        code = (
            nested_text(tx, ["transactionCoding", "transactionCode"])
            or child_text(tx, "transactionCode")
            or ""
        ).strip().upper()

        shares_txt = (
            nested_text(tx, ["transactionAmounts", "transactionShares", "value"])
            or nested_text(tx, ["transactionShares", "value"])
            or child_text(tx, "transactionShares")
            or child_text(tx, "shares")
        )
        shares = to_float(shares_txt)

        if shares == 0.0:
            underlying_txt = (
                nested_text(tx, ["underlyingSecurity", "underlyingSecurityShares", "value"])
                or nested_text(tx, ["underlyingSecurityShares", "value"])
            )
            shares = to_float(underlying_txt)

        if code == "P":
            out["buy_shares"] += shares
        elif code == "S":
            out["sell_shares"] += shares

        if not out["date"]:
            tx_date = nested_text(tx, ["transactionDate", "value"]) or child_text(tx, "transactionDate")
            if tx_date:
                out["date"] = tx_date

    out["buyers"] = sorted(buyers)
    return out


def fetch_recent_form4_events(sess: requests.Session, sym: str, cik10: str, ref_date: date) -> List[Dict[str, Any]]:
    try:
        j = get_submissions(sess, cik10)
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

        age_days = (ref_date - filing_date).days
        if age_days < 0:
            continue
        if age_days > FORM4_MAX_AGE_DAYS:
            continue

        acc_no_dash = str(acc).replace("-", "").strip()
        xmls = filing_xml_candidates(cik10, acc_no_dash, str(doc))
        parsed: Optional[Dict[str, Any]] = None

        for u in xmls:
            try:
                txt = session_get_text(sess, u)
                if "<ownershipDocument" not in txt:
                    continue
                parsed = parse_form4_xml(txt)
                break
            except Exception:
                continue

        time.sleep(SEC_SLEEP)

        if parsed:
            log(
                "INFO",
                f"{sym}: parsed Form4 event date={parsed.get('date')} "
                f"buy={parsed.get('buy_shares')} sell={parsed.get('sell_shares')} "
                f"buyers={len(parsed.get('buyers', []))}"
            )
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
    if not universe:
        raise SystemExit("Universe is empty")

    sess = requests.Session()

    raw_items: List[Dict[str, Any]] = []

    for u in universe:
        sym = u["symbol"]
        nm = u.get("name", "")

        cik10 = resolve_cik(sess, sym)
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

        events = fetch_recent_form4_events(sess, sym, cik10, ref_date)

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
            if age < 0:
                continue

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
