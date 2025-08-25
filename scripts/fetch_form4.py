#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, time, pathlib, datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import pandas as pd
import requests

OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
DATE = os.getenv("REPORT_DATE") or datetime.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
SEC_UA = os.getenv("SEC_USER_AGENT", "futuretech-stock/1.0 (contact@example.com)")
MAX_FILINGS_PER_SYMBOL = int(os.getenv("FORM4_MAX_FILINGS", "25"))   # 会社ごと最大処理件数
WINDOW_SHORT = 30
WINDOW_LONG = 90

CACHE_DIR = pathlib.Path(OUT_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CIK_CACHE = CACHE_DIR / "cik_map.json"
SECTICKERS_CACHE = CACHE_DIR / "sec_company_tickers.json"

def headers():
    # Host は付けない。UA は必須。
    return {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate", "Connection": "keep-alive"}

def load_universe_symbols():
    df = pd.read_csv(UNIVERSE_CSV)
    return [str(s).strip().upper() for s in df["symbol"].tolist()]

def normalize_symbol(sym: str):
    s = sym.upper().strip()
    alts = {s}
    if "." in s: alts.add(s.replace(".", "-"))
    if "-" in s: alts.add(s.replace("-", "."))
    return list(alts)

def load_json(p: pathlib.Path):
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: return {}
    return {}

def save_json(p: pathlib.Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

# ---------- CIK 解決 ----------
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
    # 1) ローカルキャッシュ
    cmap = load_json(CIK_CACHE)
    if sym in cmap: return str(cmap[sym]).zfill(10)
    # 2) 公式 JSON
    secmap = get_cik_map()
    for alt in normalize_symbol(sym):
        if alt in secmap:
            cmap[sym] = int(secmap[alt]); save_json(CIK_CACHE, cmap)
            return str(secmap[alt]).zfill(10)
    # 3) HTML 検索
    cik = search_cik_by_ticker_via_html(sym)
    if cik:
        cmap[sym] = int(cik); save_json(CIK_CACHE, cmap)
        return cik
    # 4) yfinance 最後の手段
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

# ---------- Submissions API → Form4 XML 取得 ----------
def get_recent_submissions(cik10: str):
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = requests.get(url, headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def iter_form4_xml_urls(cik10: str, max_items=25):
    j = get_recent_submissions(cik10)
    recent = j.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs  = recent.get("accessionNumber", [])
    prims = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])

    out = []
    for form, acc, prim, fdate in zip(forms, accs, prims, dates):
        if str(form).strip().upper() != "4":  # Form 4 のみ
            continue
        # primaryDocument が .xml ならそれを使う。そうでなければ index から探すフォールバック。
        acc_nodash = str(acc).replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_nodash}"
        if prim and prim.lower().endswith(".xml"):
            out.append({"xml": f"{base}/{prim}", "filingDate": fdate})
        else:
            # index を開いて .xml を拾う（最小限のフォールバック）
            try:
                idx = requests.get(f"{base}/{acc}-index.html", headers=headers(), timeout=30)
                idx.raise_for_status()
                m = re.findall(r'href="([^"]+\.xml)"', idx.text, flags=re.IGNORECASE)
                if m:
                    # 最初のXMLを使用
                    out.append({"xml": f"{base}/{m[0]}", "filingDate": fdate})
            except Exception:
                continue
        if len(out) >= max_items:
            break
    return out

def parse_form4_xml(xml_text: str):
    root = ET.fromstring(xml_text)
    def find_text(q):
        for x in root.iter():
            if x.tag.split("}")[-1].lower() == q.lower():
                if x.text and x.text.strip(): return x.text.strip()
        return None

    # 取引日（なければ periodOfReport）
    tx_date = find_text("transactionDate") or find_text("periodOfReport")
    buys = 0.0; sells = 0.0
    for tx in root.iter():
        tag = tx.tag.split("}")[-1].lower()
        if tag not in ("nonderivativetransaction","derivativetransaction"): continue
        code = find_text_in(tx, "transactionCode")
        shares = to_float(find_text_in(tx, "transactionShares") or find_text_in(tx, "shares"))
        if code.upper() == "P": buys += shares
        elif code.upper() == "S": sells += shares

    buyers = set()
    if buys > 0:
        for ro in root.iter():
            if ro.tag.split("}")[-1] == "rptOwnerName":
                if ro.text and ro.text.strip(): buyers.add(ro.text.strip())

    return {"date": tx_date, "buy_shares": buys, "sell_shares": sells, "buyers": sorted(list(buyers))}

def find_text_in(elem: ET.Element, qname: str):
    for x in elem.iter():
        if x.tag.split("}")[-1].lower() == qname.lower():
            if x.text and x.text.strip(): return x.text.strip()
    return None

def to_float(s):
    try: return float(str(s).replace(",", ""))
    except Exception: return 0.0

def within_days(date_iso: str, days: int, ref_date: datetime.date) -> bool:
    try: d = datetime.date.fromisoformat(date_iso)
    except Exception: return False
    return (ref_date - d).days <= days

def percentile_rank_among_positives(values, x):
    pos = sorted([v for v in values if v > 0])
    if not pos or x <= 0: return 0.0
    import bisect
    k = bisect.bisect_right(pos, x)
    return k / len(pos)

# ---------- メイン ----------
def main():
    syms = load_universe_symbols()
    ref_date = datetime.date.fromisoformat(DATE)
    per_symbol = {}

    for sym in syms:
        cik10 = resolve_cik_for_symbol(sym)
        if not cik10:
            print(f"[WARN] no CIK for {sym}", file=sys.stderr); continue

        time.sleep(0.5)  # SECに優しく
        try:
            items = iter_form4_xml_urls(cik10, max_items=MAX_FILINGS_PER_SYMBOL)
        except Exception as e:
            print(f"[WARN] submissions fetch failed {sym}: {e}", file=sys.stderr); continue

        net30=0.0; net90=0.0
        buyers30=set(); buyers90=set()

        for it in items:
            xml_url = it["xml"]
            # XML 取得
            try:
                xr = requests.get(xml_url, headers=headers(), timeout=30)
                xr.raise_for_status()
                parsed = parse_form4_xml(xr.text)
            except Exception as e:
                print(f"[WARN] xml parse failed {sym}: {e}", file=sys.stderr); continue

            dt = parsed.get("date")
            if not dt: continue
            net = float(parsed["buy_shares"]) - float(parsed["sell_shares"])
            if within_days(dt, WINDOW_LONG, ref_date):
                net90 += net; buyers90.update(parsed.get("buyers", []))
            if within_days(dt, WINDOW_SHORT, ref_date):
                net30 += net; buyers30.update(parsed.get("buyers", []))

        per_symbol[sym] = {
            "cik": cik10,
            "net_buy_shares_30": net30,
            "net_buy_shares_90": net90,
            "buyers_30": len(buyers30),
            "buyers_90": len(buyers90),
        }

    # 正規化（宇宙内パーセンタイル）
    nets30 = [v["net_buy_shares_30"] for v in per_symbol.values()]
    nets90 = [v["net_buy_shares_90"] for v in per_symbol.values()]
    b30 =   [v["buyers_30"]          for v in per_symbol.values()]
    b90 =   [v["buyers_90"]          for v in per_symbol.values()]

    for sym, rec in per_symbol.items():
        net30_pct = percentile_rank_among_positives(nets30, rec["net_buy_shares_30"])
        net90_pct = percentile_rank_among_positives(nets90, rec["net_buy_shares_90"])
        b30_pct   = percentile_rank_among_positives(b30,   rec["buyers_30"])
        b90_pct   = percentile_rank_among_positives(b90,   rec["buyers_90"])
        score30 = 0.7*net30_pct + 0.3*b30_pct
        score90 = 0.7*net90_pct + 0.3*b90_pct
        insider_momo = max(score30, 0.9*score90)
        rec.update({
            "score_30": round(score30, 6),
            "score_90": round(score90, 6),
            "insider_momo": round(insider_momo, 6),
        })

    out_latest = pathlib.Path(OUT_DIR) / "data" / "insider" / "form4_latest.json"
    out_today  = pathlib.Path(OUT_DIR) / "data" / DATE / "insider.json"
    payload = {"as_of": DATE, "window_days": {"short": WINDOW_SHORT, "long": WINDOW_LONG}, "items": per_symbol}
    save_json(out_latest, payload); save_json(out_today, payload)
    print(f"Form4 saved: {out_latest} & {out_today} ({len(per_symbol)} symbols)")

if __name__ == "__main__":
    main()
