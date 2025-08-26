#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, time, pathlib, datetime, bisect
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import pandas as pd
import requests

# ---------------- Config ----------------
OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
DATE = os.getenv("REPORT_DATE") or datetime.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
SEC_UA = os.getenv("SEC_USER_AGENT", "futuretech-stock/1.0 (contact@example.com)")

MAX_FILINGS_PER_SYMBOL = int(os.getenv("FORM4_MAX_FILINGS", "12"))
MAX_AGE_DAYS = int(os.getenv("FORM4_MAX_AGE_DAYS", "95"))
SEC_SLEEP = float(os.getenv("SEC_SLEEP", "0.7"))
SEC_XML_SLEEP = float(os.getenv("SEC_XML_SLEEP", "0.25"))
RETRY = int(os.getenv("SEC_RETRY", "3"))
DEBUG_DUMP = os.getenv("FORM4_DEBUG_DUMP", "0") == "1"

WINDOW_SHORT = 30
WINDOW_LONG = 90

# -------------- Caches --------------
CACHE_DIR = pathlib.Path(OUT_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CIK_CACHE = CACHE_DIR / "cik_map.json"
SECTICKERS_CACHE = CACHE_DIR / "sec_company_tickers.json"
DUMP_DIR = CACHE_DIR / "form4_dumps"

def headers():
    return {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate", "Connection": "keep-alive"}

# -------------- Utils --------------
def load_json(p: pathlib.Path):
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: return {}
    return {}

def save_json(p: pathlib.Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def load_universe_symbols():
    df = pd.read_csv(UNIVERSE_CSV)
    return [str(s).strip().upper() for s in df["symbol"].tolist()]

def normalize_symbol(sym: str):
    s = sym.upper().strip()
    alts = {s}
    if "." in s: alts.add(s.replace(".", "-"))
    if "-" in s: alts.add(s.replace("-", "."))
    return list(alts)

# -------------- CIK resolve --------------
def build_cik_map_from_sec(max_retry=3, backoff=0.8):
    url = "https://www.sec.gov/files/company_tickers.json"
    last = None
    for i in range(max_retry):
        try:
            r = requests.get(url, headers=headers(), timeout=30)
            r.raise_for_status()
            data = r.json()
            mapping = {}
            for _, rec in data.items():
                t = str(rec.get("ticker","")).upper().strip()
                cik = int(rec.get("cik_str", 0))
                if t and cik: mapping[t] = cik
            save_json(SECTICKERS_CACHE, mapping)
            return mapping
        except Exception as e:
            last = e; time.sleep(backoff*(i+1))
    raise last

def get_cik_map():
    cache = load_json(SECTICKERS_CACHE)
    if cache: return cache
    try:
        return build_cik_map_from_sec()
    except Exception as e:
        print(f"[WARN] company_tickers fetch failed, fallback to HTML search: {e}", file=sys.stderr)
        return {}

def search_cik_by_ticker_via_html(sym: str) -> str|None:
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={quote_plus(sym)}&owner=exclude&action=getcompany"
    try:
        r = requests.get(url, headers=headers(), timeout=30)
        r.raise_for_status()
        m = re.search(r"CIK=0*([0-9]{1,10})", r.text, flags=re.IGNORECASE)
        if m:
            return str(int(m.group(1))).zfill(10)
    except Exception:
        pass
    return None

def resolve_cik_for_symbol(sym: str) -> str|None:
    cmap = load_json(CIK_CACHE)
    if sym in cmap: return str(cmap[sym]).zfill(10)
    secmap = get_cik_map()
    for alt in normalize_symbol(sym):
        if alt in secmap:
            cmap[sym] = int(secmap[alt]); save_json(CIK_CACHE, cmap)
            return str(secmap[alt]).zfill(10)
    cik = search_cik_by_ticker_via_html(sym)
    if cik:
        cmap[sym] = int(cik); save_json(CIK_CACHE, cmap)
        return cik
    try:
        import yfinance as yf
        info = yf.Ticker(sym).get_info()
        cik2 = info.get("cik")
        if cik2:
            cmap[sym] = int(cik2); save_json(CIK_CACHE, cmap)
            return str(cik2).zfill(10)
    except Exception:
        pass
    return None

# -------------- Submissions + file picking --------------
def get_recent_submissions(cik10: str):
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = requests.get(url, headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def pick_xml_from_dir(base_url: str, primary_doc: str|None) -> list[str]:
    cands: list[str] = []
    # index.json 優先
    try:
        jr = requests.get(f"{base_url}/index.json", headers=headers(), timeout=20)
        if jr.ok:
            j = jr.json()
            items = j.get("directory", {}).get("item", [])
            xmls = [it["name"] for it in items if str(it.get("name","")).lower().endswith(".xml")]
            xmls.sort(key=lambda nm: (
                0 if "form4" in nm.lower() else
                1 if "ownership" in nm.lower() else
                2 if "primary" in nm.lower() else
                9))
            cands.extend([f"{base_url}/{nm}" for nm in xmls])
    except Exception:
        pass
    # -index.html or {acc}-index.html
    try:
        ir = requests.get(f"{base_url}/-index.html", headers=headers(), timeout=20)
        if not ir.ok:
            acc_nodash = base_url.rstrip("/").split("/")[-1]
            acc = f"{acc_nodash[:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}"
            ir = requests.get(f"{base_url}/{acc}-index.html", headers=headers(), timeout=20)
        if ir.ok:
            ms = re.findall(r'href="([^"]+\.xml)"', ir.text, flags=re.IGNORECASE)
            for m in ms:
                cands.append(m if m.startswith("http") else f"{base_url}/{m}")
    except Exception:
        pass
    # primary が .xml
    if primary_doc and str(primary_doc).lower().endswith(".xml"):
        cands.append(f"{base_url}/{primary_doc}")
    # uniq
    uniq, seen = [], set()
    for u in cands:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq

def fetch_text_with_retry(url: str, expect_xml: bool = False, dump_path: pathlib.Path|None = None) -> str|None:
    last = None
    for i in range(RETRY):
        try:
            r = requests.get(url, headers=headers(), timeout=30)
            if not r.ok:
                last = Exception(f"HTTP {r.status_code}")
                time.sleep(SEC_XML_SLEEP*(i+1)); continue
            txt = r.text
            if expect_xml:
                ctype = r.headers.get("Content-Type","").lower()
                if ("xml" not in ctype) and ("text/xml" not in ctype) and not re.search(r"<ownershipDocument", txt, re.IGNORECASE):
                    last = Exception("not-xml"); 
                    if DEBUG_DUMP and dump_path:
                        dump_path.parent.mkdir(parents=True, exist_ok=True)
                        dump_path.write_text(txt[:2000])
                    time.sleep(SEC_XML_SLEEP*(i+1)); continue
            return txt
        except Exception as e:
            last = e
        time.sleep(SEC_XML_SLEEP*(i+1))
    return None

def is_form4_xml(text: str) -> bool:
    return bool(re.search(r"<ownershipDocument", text, re.IGNORECASE))

# -------------- Parse Form4 XML --------------
def find_text(elem: ET.Element, qname: str):
    for x in elem.iter():
        if x.tag.split("}")[-1].lower() == qname.lower():
            if x.text and x.text.strip(): return x.text.strip()
    return None

def find_text_in(elem: ET.Element, qname: str):
    for x in elem.iter():
        if x.tag.split("}")[-1].lower() == qname.lower():
            if x.text and x.text.strip(): return x.text.strip()
    return None

def to_float(x):
    try: return float(str(x).replace(",", ""))
    except Exception: return 0.0

def parse_form4_xml(xml_text: str):
    if not is_form4_xml(xml_text):
        raise ET.ParseError("not a Form4 XML")
    root = ET.fromstring(xml_text)
    tx_date = find_text(root, "transactionDate") or find_text(root, "periodOfReport")
    buys = 0.0; sells = 0.0
    for tx in root.iter():
        tag = tx.tag.split("}")[-1].lower()
        if tag not in ("nonderivativetransaction","derivativetransaction"): continue
        code = (find_text_in(tx, "transactionCode") or "").upper()
        shares = to_float(find_text_in(tx, "transactionShares") or find_text_in(tx, "shares"))
        if code == "P": buys += shares
        elif code == "S": sells += shares
    buyers = set()
    if buys > 0:
        for ro in root.iter():
            if ro.tag.split("}")[-1] == "rptOwnerName":
                if ro.text and ro.text.strip(): buyers.add(ro.text.strip())
    return {"date": tx_date, "buy_shares": buys, "sell_shares": sells, "buyers": sorted(list(buyers))}

def within_days(date_iso: str, days: int, ref_date: datetime.date) -> bool:
    try: d = datetime.date.fromisoformat(date_iso)
    except Exception: return False
    return (ref_date - d).days <= days

def pct_rank_positive(values, x):
    pos = sorted([v for v in values if v > 0])
    if not pos or x <= 0: return 0.0
    k = bisect.bisect_right(pos, x)
    return k / len(pos)

# -------------- Main --------------
def main():
    syms = load_universe_symbols()
    ref_date = datetime.date.fromisoformat(DATE)
    per_symbol = {}
    total_xml = 0
    total_parsed = 0

    for sym in syms:
        cik10 = resolve_cik_for_symbol(sym)
        if not cik10:
            print(f"[WARN] no CIK for {sym}", file=sys.stderr); 
            per_symbol[sym] = {"cik": None, "net_buy_shares_30":0.0,"net_buy_shares_90":0.0,"buyers_30":0,"buyers_90":0,
                               "debug":{"skipped_old":0,"bad_xml":0,"taken":0}}
            continue

        time.sleep(SEC_SLEEP)
        try:
            subs = get_recent_submissions(cik10)
        except Exception as e:
            print(f"[WARN] submissions fetch failed {sym}: {e}", file=sys.stderr)
            per_symbol[sym] = {"cik": cik10, "net_buy_shares_30":0.0,"net_buy_shares_90":0.0,"buyers_30":0,"buyers_90":0,
                               "debug":{"skipped_old":0,"bad_xml":0,"taken":0}}
            continue

        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accs  = recent.get("accessionNumber", [])
        prims = recent.get("primaryDocument", [])
        fdates= recent.get("filingDate", [])

        net30=0.0; net90=0.0; buyers30=set(); buyers90=set()
        taken = 0; skipped_old = 0; bad_xml = 0

        for form, acc, prim, fdate in zip(forms, accs, prims, fdates):
            if str(form).strip().upper() != "4":
                continue
            try:
                fd = datetime.date.fromisoformat(str(fdate))
                if (ref_date - fd).days > MAX_AGE_DAYS:
                    skipped_old += 1
                    break
            except Exception:
                pass

            acc_nodash = str(acc).replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_nodash}"
            xml_candidates = pick_xml_from_dir(base, prim)
            if not xml_candidates:
                bad_xml += 1
                continue

            parsed_one = False
            for xml_url in xml_candidates:
                time.sleep(SEC_XML_SLEEP)
                dump_path = None
                if DEBUG_DUMP:
                    dump_path = DUMP_DIR / sym / f"{acc_nodash}.txt"
                txt = fetch_text_with_retry(xml_url, expect_xml=True, dump_path=dump_path)
                if not txt:
                    continue
                try:
                    parsed = parse_form4_xml(txt)
                except Exception:
                    if DEBUG_DUMP and dump_path and not dump_path.exists():
                        dump_path.parent.mkdir(parents=True, exist_ok=True)
                        dump_path.write_text(txt[:2000])
                    continue

                total_xml += 1
                parsed_one = True
                dt = parsed.get("date")
                if dt:
                    net = float(parsed["buy_shares"]) - float(parsed["sell_shares"])
                    if within_days(dt, WINDOW_LONG, ref_date):
                        net90 += net; buyers90.update(parsed.get("buyers", []))
                    if within_days(dt, WINDOW_SHORT, ref_date):
                        net30 += net; buyers30.update(parsed.get("buyers", []))
                break

            if not parsed_one:
                bad_xml += 1

            taken += 1
            if taken >= MAX_FILINGS_PER_SYMBOL:
                break

        per_symbol[sym] = {
            "cik": cik10,
            "net_buy_shares_30": net30,
            "net_buy_shares_90": net90,
            "buyers_30": len(buyers30),
            "buyers_90": len(buyers90),
            "debug": {"skipped_old": skipped_old, "bad_xml": bad_xml, "taken": taken}
        }

    # 正規化
    nets30 = [v["net_buy_shares_30"] for v in per_symbol.values()]
    nets90 = [v["net_buy_shares_90"] for v in per_symbol.values()]
    b30 =   [v["buyers_30"]          for v in per_symbol.values()]
    b90 =   [v["buyers_90"]          for v in per_symbol.values()]

    for sym, rec in per_symbol.items():
        net30_pct = pct_rank_positive(nets30, rec["net_buy_shares_30"])
        net90_pct = pct_rank_positive(nets90, rec["net_buy_shares_90"])
        b30_pct   = pct_rank_positive(b30,   rec["buyers_30"])
        b90_pct   = pct_rank_positive(b90,   rec["buyers_90"])
        score30 = 0.7*net30_pct + 0.3*b30_pct
        score90 = 0.7*net90_pct + 0.3*b90_pct
        insider_momo = max(score30, 0.9*score90)
        rec.update({
            "score_30": round(score30, 6),
            "score_90": round(score90, 6),
            "insider_momo": round(insider_momo, 6),
        })

    out_latest = (pathlib.Path(OUT_DIR) / "data" / "insider" / "form4_latest.json").resolve()
    out_today  = (pathlib.Path(OUT_DIR) / "data" / DATE / "insider.json").resolve()
    payload = {"as_of": DATE, "window_days": {"short": WINDOW_SHORT, "long": WINDOW_LONG}, "items": per_symbol}
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_today.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_latest, payload)
    save_json(out_today, payload)

    # サマリ
    sym_ok = sum(1 for v in per_symbol.values() if (v["net_buy_shares_30"]>0 or v["net_buy_shares_90"]>0 or v["buyers_30"]>0 or v["buyers_90"]>0))
    print(f"Form4 saved:")
    print(f"  latest: {out_latest}")
    print(f"  today : {out_today}")
    print(f"[INFO] symbols processed: {len(per_symbol)}, with any activity: {sym_ok}")
    # 代表例のデバッグ
    for s, rec in list(per_symbol.items())[:5]:
        dbg = rec.get('debug', {})
        print(f"[DEBUG] {s} -> taken={dbg.get('taken')} bad_xml={dbg.get('bad_xml')} skipped_old={dbg.get('skipped_old')}")

if __name__ == "__main__":
    main()
