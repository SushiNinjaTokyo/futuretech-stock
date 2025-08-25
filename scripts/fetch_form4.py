#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, time, math, pathlib, datetime
from zoneinfo import ZoneInfo
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import pandas as pd
import requests

OUT_DIR = os.getenv("OUT_DIR", "site")
UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
DATE = os.getenv("REPORT_DATE") or datetime.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
SEC_UA = os.getenv("SEC_USER_AGENT", "futuretech-stock/1.0 (contact@example.com)")
MAX_FILINGS_PER_SYMBOL = int(os.getenv("FORM4_MAX_FILINGS", "25"))   # 会社ごとに処理する最大件数
WINDOW_SHORT = 30
WINDOW_LONG = 90

CACHE_DIR = pathlib.Path(OUT_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CIK_CACHE = CACHE_DIR / "cik_map.json"
SECTICKERS_CACHE = CACHE_DIR / "sec_company_tickers.json"

def sec_headers():
    return {
        "User-Agent": SEC_UA,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

def load_universe_symbols():
    df = pd.read_csv(UNIVERSE_CSV)
    syms = [str(s).strip().upper() for s in df["symbol"].tolist()]
    return syms

def normalize_symbol(sym: str) -> list[str]:
    """SECのティッカー表記差異（- と .）を両方試す"""
    s = sym.upper().strip()
    alts = {s}
    if "." in s:
        alts.add(s.replace(".", "-"))
    if "-" in s:
        alts.add(s.replace("-", "."))
    return list(alts)

def load_json(path: pathlib.Path):
    if path.exists():
        try: return json.loads(path.read_text())
        except Exception: return {}
    return {}

def save_json(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def build_cik_map_from_sec():
    # https://www.sec.gov/files/company_tickers.json
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=sec_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()  # { "0": { "cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc." }, ... }
    mapping = {}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).upper().strip()
        cik = int(rec.get("cik_str", 0))
        if t and cik:
            mapping[t] = cik
    save_json(SECTICKERS_CACHE, mapping)
    return mapping

def get_cik_map():
    # 優先：ローカルキャッシュ → SEC公式JSON → 空
    cache = load_json(SECTICKERS_CACHE)
    if cache: return cache
    try:
        return build_cik_map_from_sec()
    except Exception as e:
        print(f"[WARN] SEC company_tickers fetch failed: {e}", file=sys.stderr)
        return {}

def resolve_cik_for_symbol(sym: str) -> str | None:
    # 既存キャッシュ
    cik_map = load_json(CIK_CACHE)
    if sym in cik_map:
        return str(cik_map[sym]).zfill(10)
    # SEC公式マップから探す（表記ゆれも試す）
    secmap = get_cik_map()
    for alt in normalize_symbol(sym):
        if alt in secmap:
            cik_map[sym] = int(secmap[alt])
            save_json(CIK_CACHE, cik_map)
            return str(secmap[alt]).zfill(10)
    # yfinance フォールバック（なくてもOK）
    try:
        import yfinance as yf
        info = yf.Ticker(sym).get_info()
        cik = info.get("cik")
        if cik:
            cik_map[sym] = int(cik)
            save_json(CIK_CACHE, cik_map)
            return str(cik).zfill(10)
    except Exception:
        pass
    return None

def fetch_atom_entries_for_cik(cik: str):
    url = f"https://data.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&owner=only&count=100&output=atom"
    r = requests.get(url, headers=sec_headers(), timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = root.findall("a:entry", ns)
    out = []
    for e in entries[:MAX_FILINGS_PER_SYMBOL]:
        link = e.find("a:link", ns)
        href = link.get("href") if link is not None else None
        updated = e.findtext("a:updated", default=None, namespaces=ns)
        title = e.findtext("a:title", default="", namespaces=ns)
        out.append({"href": href, "updated": updated, "title": title})
    return out

def extract_first_xml_url_from_index(index_html_url: str) -> str | None:
    r = requests.get(index_html_url, headers=sec_headers(), timeout=30)
    r.raise_for_status()
    html = r.text
    # index ページ内の .xml リンク（Form 4のXML）
    # よくあるパターン: .../xslF345X03/XXXX.xml など
    m = re.findall(r'href="([^"]+\.xml)"', html, flags=re.IGNORECASE)
    if not m:
        # -index.htm → .xml トライ
        if index_html_url.endswith("-index.htm"):
            return index_html_url.replace("-index.htm", ".xml")
        return None
    # 相対→絶対
    for href in m:
        url = urljoin(index_html_url, href)
        if re.search(r'(form4|f345|xslf345)', url, flags=re.IGNORECASE):
            return url
    return urljoin(index_html_url, m[0])

def _find_text(elem: ET.Element, qname: str):
    # 名前空間無視でローカル名一致を探す
    for x in elem.iter():
        tag = x.tag.split("}")[-1]
        if tag.lower() == qname.lower():
            if x.text and x.text.strip():
                return x.text.strip()
    return None

def parse_form4_xml(xml_text: str):
    """
    戻り値: dict(date, buy_shares, sell_shares, buyers) — buyersは文書内の報告者名セット
    """
    root = ET.fromstring(xml_text)
    # 取引日
    date = _find_text(root, "transactionDate") or _find_text(root, "periodOfReport")
    # 取引テーブル（非デリバ/デリバ両方）
    buys = 0.0
    sells = 0.0
    # すべての transaction 要素を探索
    for tx in root.iter():
        tag = tx.tag.split("}")[-1].lower()
        if tag not in ("nonderivativetransaction", "derivativetransaction"):
            continue
        code = _find_text(tx, "transactionCode") or ""
        shares_txt = _find_text(tx, "transactionShares") or _find_text(tx, "shares")
        try:
            shares = float(shares_txt.replace(",", "")) if shares_txt else 0.0
        except Exception:
            shares = 0.0
        if code.upper() == "P":  # open market purchase
            buys += shares
        elif code.upper() == "S":  # sale
            sells += shares

    buyers = set()
    has_buy = buys > 0
    if has_buy:
        for ro in root.iter():
            if ro.tag.split("}")[-1] == "rptOwnerName":
                if ro.text and ro.text.strip():
                    buyers.add(ro.text.strip())

    return {"date": date, "buy_shares": buys, "sell_shares": sells, "buyers": sorted(list(buyers))}

def within_days(date_iso: str, days: int, ref_date: datetime.date) -> bool:
    try:
        d = datetime.date.fromisoformat(date_iso)
    except Exception:
        return False
    return (ref_date - d).days <= days

def percentile_rank_among_positives(values: list[float], x: float) -> float:
    pos = sorted([v for v in values if v > 0])
    if not pos or x <= 0:
        return 0.0
    import bisect
    k = bisect.bisect_right(pos, x)
    return k / len(pos)

def main():
    symbols = load_universe_symbols()
    ref_date = datetime.date.fromisoformat(DATE)

    per_symbol = {}
    for sym in symbols:
        cik = resolve_cik_for_symbol(sym)
        if not cik:
            print(f"[WARN] no CIK for {sym}", file=sys.stderr)
            continue

        time.sleep(0.6)  # SECに優しく
        try:
            entries = fetch_atom_entries_for_cik(cik)
        except Exception as e:
            print(f"[WARN] fetch entries failed {sym}: {e}", file=sys.stderr)
            continue

        net30 = 0.0; net90 = 0.0
        buyers30 = set(); buyers90 = set()

        for ent in entries:
            href = ent.get("href")
            if not href:
                continue
            # index → XML
            try:
                xml_url = extract_first_xml_url_from_index(href)
            except Exception as e:
                print(f"[WARN] index fetch failed {sym}: {e}", file=sys.stderr)
                continue
            if not xml_url:
                continue
            time.sleep(0.4)
            try:
                xr = requests.get(xml_url, headers=sec_headers(), timeout=30)
                xr.raise_for_status()
                parsed = parse_form4_xml(xr.text)
            except Exception as e:
                print(f"[WARN] xml parse failed {sym}: {e}", file=sys.stderr)
                continue

            dt = parsed.get("date")
            if not dt:
                continue
            net = float(parsed["buy_shares"]) - float(parsed["sell_shares"])
            if within_days(dt, WINDOW_LONG, ref_date):
                net90 += net
                buyers90.update(parsed.get("buyers", []))
            if within_days(dt, WINDOW_SHORT, ref_date):
                net30 += net
                buyers30.update(parsed.get("buyers", []))

        per_symbol[sym] = {
            "cik": cik,
            "net_buy_shares_30": net30,
            "net_buy_shares_90": net90,
            "buyers_30": len(buyers30),
            "buyers_90": len(buyers90),
        }

    # 正規化（宇宙内パーセンタイル）
    nets30 = [v["net_buy_shares_30"] for v in per_symbol.values()]
    nets90 = [v["net_buy_shares_90"] for v in per_symbol.values()]
    b30 = [v["buyers_30"] for v in per_symbol.values()]
    b90 = [v["buyers_90"] for v in per_symbol.values()]

    for sym, rec in per_symbol.items():
        net30_pct = percentile_rank_among_positives(nets30, rec["net_buy_shares_30"])
        net90_pct = percentile_rank_among_positives(nets90, rec["net_buy_shares_90"])
        b30_pct = percentile_rank_among_positives(b30, rec["buyers_30"])
        b90_pct = percentile_rank_among_positives(b90, rec["buyers_90"])

        score30 = 0.7 * net30_pct + 0.3 * b30_pct
        score90 = 0.7 * net90_pct + 0.3 * b90_pct
        insider_momo = max(score30, 0.9 * score90)

        rec.update({
            "score_30": round(score30, 6),
            "score_90": round(score90, 6),
            "insider_momo": round(insider_momo, 6),
        })

    out_latest = pathlib.Path(OUT_DIR) / "data" / "insider" / "form4_latest.json"
    out_today  = pathlib.Path(OUT_DIR) / "data" / DATE / "insider.json"
    payload = {
        "as_of": DATE,
        "window_days": {"short": WINDOW_SHORT, "long": WINDOW_LONG},
        "items": per_symbol,
    }
    save_json(out_latest, payload)
    save_json(out_today, payload)
    print(f"Form4 saved: {out_latest} & {out_today} ({len(per_symbol)} symbols)")

if __name__ == "__main__":
    main()
