#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import pandas as pd
import requests


OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE") or datetime.now(ZoneInfo("America/New_York")).date().isoformat()

SEC_UA = os.getenv("SEC_USER_AGENT", "futuretech-stock/1.0 (contact: your-email@example.com)")
FORM4_MAX_FILINGS = int(os.getenv("FORM4_MAX_FILINGS", "12"))
FORM4_MAX_AGE_DAYS = int(os.getenv("FORM4_MAX_AGE_DAYS", "120"))
SEC_SLEEP = float(os.getenv("SEC_SLEEP", "0.7"))

CACHE_DIR = OUT_DIR / "cache"
CIK_CACHE = CACHE_DIR / "cik_map.json"
COMPANY_CACHE = CACHE_DIR / "sec_company_tickers.json"

# DII scoring parameters
DII_LOOKBACK_DAYS = int(os.getenv("DII_LOOKBACK_DAYS", "30"))
DII_LONG_LOOKBACK_DAYS = int(os.getenv("DII_LONG_LOOKBACK_DAYS", "90"))

# score mix
DII_WEIGHT_UNIQUE_BUYERS = float(os.getenv("DII_WEIGHT_UNIQUE_BUYERS", "0.40"))
DII_WEIGHT_BUY_EVENTS = float(os.getenv("DII_WEIGHT_BUY_EVENTS", "0.30"))
DII_WEIGHT_CLUSTER = float(os.getenv("DII_WEIGHT_CLUSTER", "0.20"))
DII_WEIGHT_OFFICER = float(os.getenv("DII_WEIGHT_OFFICER", "0.10"))


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


def extract_transaction_date(tx: ET.Element, fallback_date: Optional[str]) -> Optional[str]:
    for ch in tx.iter():
        nm = ch.tag.split("}")[-1].lower()
        txt = (ch.text or "").strip()
        if nm == "value" and txt:
            parent_tag = ""
            try:
                parent_tag = ch.getparent().tag.split("}")[-1].lower()  # type: ignore[attr-defined]
            except Exception:
                parent_tag = ""
            if parent_tag == "transactiondate":
                return txt

    # ElementTree has no getparent(), so manual scan
    for node in tx.iter():
        nm = node.tag.split("}")[-1].lower()
        if nm == "transactiondate":
            for sub in node.iter():
                sub_nm = sub.tag.split("}")[-1].lower()
                txt = (sub.text or "").strip()
                if sub_nm == "value" and txt:
                    return txt

    return fallback_date


def detect_owner_role(root: ET.Element) -> Tuple[str, float]:
    """
    officer/director/10% owner のざっくり重み。
    CEO/CFO を強く評価。
    """
    owner_titles: List[str] = []
    is_director = False
    is_officer = False
    is_ten_percent = False

    for el in root.iter():
        nm = el.tag.split("}")[-1].lower()
        txt = (el.text or "").strip()

        if nm == "isdirector":
            is_director = txt == "1"
        elif nm == "isofficer":
            is_officer = txt == "1"
        elif nm == "istenpercentowner":
            is_ten_percent = txt == "1"
        elif nm == "officertitle" and txt:
            owner_titles.append(txt.lower())

    title_blob = " | ".join(owner_titles)

    if "chief executive officer" in title_blob or re.search(r"\bceo\b", title_blob):
        return "ceo", 1.0
    if "chief financial officer" in title_blob or re.search(r"\bcfo\b", title_blob):
        return "cfo", 1.0
    if "president" in title_blob or "chief operating officer" in title_blob or re.search(r"\bcoo\b", title_blob):
        return "senior_officer", 0.8
    if is_officer:
        return "officer", 0.7
    if is_director:
        return "director", 0.5
    if is_ten_percent:
        return "ten_percent_owner", 0.3
    return "other", 0.2


def parse_form4_xml(xml_text: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_text)

    report_date = parse_text(root, "periodOfReport")
    owner_role, role_weight = detect_owner_role(root)

    owner_names: Set[str] = set()
    for ro in root.iter():
        if ro.tag.split("}")[-1].lower() == "rptownername":
            txt = (ro.text or "").strip()
            if txt:
                owner_names.add(txt)

    txs: List[Dict[str, Any]] = []

    for tx in root.iter():
        tag = tx.tag.split("}")[-1].lower()
        if tag not in {"nonderivativetransaction", "derivativetransaction"}:
            continue

        code = ""
        shares = 0.0
        price = 0.0
        tx_date = report_date

        for ch in tx.iter():
            nm = ch.tag.split("}")[-1].lower()
            txt = (ch.text or "").strip()

            if nm == "transactioncode" and txt:
                code = txt.upper()
            elif nm in {"transactionshares", "shares"} and txt:
                shares = to_float(txt)
            elif nm == "transactionpricepershare" and txt:
                price = to_float(txt)

        tx_date = extract_transaction_date(tx, report_date)

        if code in {"P", "S"}:
            txs.append({
                "code": code,  # P=buy, S=sell
                "shares": round(shares, 4),
                "price": round(price, 4),
                "date": tx_date,
            })

    return {
        "report_date": report_date,
        "owner_names": sorted(owner_names),
        "owner_role": owner_role,
        "role_weight": role_weight,
        "transactions": txs,
    }


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


def safe_event_date(s: Optional[str], fallback: Optional[str]) -> Optional[date]:
    candidates = [s, fallback]
    for c in candidates:
        if not c:
            continue
        try:
            return date.fromisoformat(str(c))
        except Exception:
            continue
    return None


def percentile_rank(values: List[float], x: float) -> float:
    vals = sorted(float(v) for v in values if v is not None and v >= 0)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return 1.0 if x > 0 else 0.0
    le = sum(1 for v in vals if v <= x)
    return max(0.0, min(1.0, (le - 1) / (len(vals) - 1)))


def normalize_weights() -> Tuple[float, float, float, float]:
    ws = [
        DII_WEIGHT_UNIQUE_BUYERS,
        DII_WEIGHT_BUY_EVENTS,
        DII_WEIGHT_CLUSTER,
        DII_WEIGHT_OFFICER,
    ]
    s = sum(ws)
    if s <= 0:
        return (0.40, 0.30, 0.20, 0.10)
    return tuple(w / s for w in ws)


def main() -> None:
    ref_date = date.fromisoformat(REPORT_DATE)
    universe = load_universe()

    raw_items: List[Dict[str, Any]] = []
    dii_raw_scores: List[float] = []

    w_unique, w_events, w_cluster, w_officer = normalize_weights()

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
                "buy_event_count_30d": 0,
                "buy_event_count_90d": 0,
                "unique_buyers_30d": 0,
                "unique_buyers_90d": 0,
                "cluster_buy_flag_30d": 0,
                "officer_weight_30d": 0.0,
                "dii_raw_score": 0.0,
                "score_0_1": 0.0,
            })
            dii_raw_scores.append(0.0)
            continue

        events = fetch_recent_form4_events(sym, cik10, ref_date)

        net30 = 0.0
        net90 = 0.0

        buyers30: Set[str] = set()
        buyers90: Set[str] = set()

        buy_events_30 = 0
        buy_events_90 = 0

        unique_buyers_30: Set[str] = set()
        unique_buyers_90: Set[str] = set()

        officer_weight_30 = 0.0
        officer_weight_90 = 0.0

        for ev in events:
            fallback_report_date = ev.get("report_date")
            owner_names = ev.get("owner_names", []) or []
            role_weight = float(ev.get("role_weight", 0.0))

            event_has_buy_30 = False
            event_has_buy_90 = False

            for tx in ev.get("transactions", []) or []:
                tx_date = safe_event_date(tx.get("date"), fallback_report_date)
                if tx_date is None:
                    continue

                age = (ref_date - tx_date).days
                if age < 0:
                    continue

                shares = float(tx.get("shares", 0.0))
                code = str(tx.get("code", "")).upper()

                net_effect = shares if code == "P" else (-shares if code == "S" else 0.0)

                if age <= DII_LONG_LOOKBACK_DAYS:
                    net90 += net_effect
                    buyers90.update(owner_names)
                    if code == "P":
                        buy_events_90 += 1
                        unique_buyers_90.update(owner_names)
                        event_has_buy_90 = True

                if age <= DII_LOOKBACK_DAYS:
                    net30 += net_effect
                    buyers30.update(owner_names)
                    if code == "P":
                        buy_events_30 += 1
                        unique_buyers_30.update(owner_names)
                        event_has_buy_30 = True

            if event_has_buy_30:
                officer_weight_30 = max(officer_weight_30, role_weight)
            if event_has_buy_90:
                officer_weight_90 = max(officer_weight_90, role_weight)

        cluster_buy_flag_30 = 1 if len(unique_buyers_30) >= 2 else 0

        # saturation-based raw scoring
        s_unique = min(len(unique_buyers_30), 3) / 3.0
        s_events = min(buy_events_30, 4) / 4.0
        s_cluster = float(cluster_buy_flag_30)
        s_officer = min(officer_weight_30, 1.0)

        dii_raw = (
            w_unique * s_unique
            + w_events * s_events
            + w_cluster * s_cluster
            + w_officer * s_officer
        )
        dii_raw = round(max(0.0, min(1.0, dii_raw)), 8)

        raw_items.append({
            "symbol": sym,
            "name": nm,
            "cik": cik10,
            "net_buy_shares_30": round(net30, 2),
            "net_buy_shares_90": round(net90, 2),
            "buyers_30": len(buyers30),
            "buyers_90": len(buyers90),
            "buy_event_count_30d": buy_events_30,
            "buy_event_count_90d": buy_events_90,
            "unique_buyers_30d": len(unique_buyers_30),
            "unique_buyers_90d": len(unique_buyers_90),
            "cluster_buy_flag_30d": cluster_buy_flag_30,
            "officer_weight_30d": round(officer_weight_30, 4),
            "officer_weight_90d": round(officer_weight_90, 4),
            "dii_raw_score": dii_raw,
        })
        dii_raw_scores.append(dii_raw)

    # percentile normalize across universe
    items: List[Dict[str, Any]] = []
    for row in raw_items:
        score01 = percentile_rank(dii_raw_scores, float(row["dii_raw_score"]))
        items.append({
            **row,
            "score_0_1": round(score01, 6),
        })

    payload = {"date": REPORT_DATE, "items": items}

    day_path = OUT_DIR / "data" / REPORT_DATE / "dii.json"
    latest_path = OUT_DIR / "data" / "dii" / "latest.json"

    write_json(day_path, payload)
    write_json(latest_path, payload)
    log("INFO", f"Wrote DII: {day_path} ({len(items)} items)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR", f"FATAL in fetch_form4: {e}")
        sys.exit(1)
