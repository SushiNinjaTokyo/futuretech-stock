#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, re, csv, math, time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from urllib.parse import quote_plus, urlparse, urlunparse, parse_qsl, urlencode

import feedparser

# ===================== Env & Const =====================

OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE") or datetime.utcnow().date().isoformat()

LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
MAX_PER = int(os.getenv("NEWS_MAX_PER_SYMBOL", "20"))
SLEEP_SEC = float(os.getenv("NEWS_SLEEP_SEC", "1.0"))

# 新しさの半減期（例: 3日で重みが1/2）
DECAY_HALF_LIFE_DAYS = float(os.getenv("NEWS_DECAY_HALF_LIFE_DAYS", "3"))

# Google News RSS（q= はエンコード済みの検索式を入れる）
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

# 曖昧ティッカーなどに備え、社名も含めて検索（stock/company を付加）
# 例: ("NVDA" OR "NVIDIA Corporation") (stock OR company)
def build_query(symbol: str, name: Optional[str]) -> str:
    terms = [f'"{symbol}"']
    if name:
        terms.append(f'"{name}"')
    base = "(" + " OR ".join(terms) + ") (stock OR company)"
    return quote_plus(base)

# ===================== Helpers =====================

def load_universe(csv_path: Path) -> List[Tuple[str, str]]:
    """CSV から (symbol, name) のリストを返す。区切りはカンマ/タブ等を自動判定。"""
    if not csv_path.exists():
        # フォールバック（20銘柄）
        return [
            ("NVDA","NVIDIA Corporation"), ("MSFT","Microsoft Corporation"),
            ("PLTR","Palantir Technologies Inc."), ("AI","C3.ai, Inc."),
            ("ISRG","Intuitive Surgical, Inc."), ("TER","Teradyne, Inc."),
            ("SYM","Symbotic Inc."), ("RKLB","Rocket Lab USA, Inc."),
            ("IRDM","Iridium Communications Inc."), ("VSAT","Viasat, Inc."),
            ("INOD","Innodata Inc."), ("SOUN","SoundHound AI, Inc."),
            ("MNDY","Monday.com Ltd."), ("AVAV","AeroVironment, Inc."),
            ("PERF","Perfect Corp."), ("GDRX","GoodRx Holdings, Inc."),
            ("ABCL","AbCellera Biologics Inc."), ("U","Unity Software Inc."),
            ("TEM","Tempus AI, Inc."), ("VRT","Vertiv Holdings Co"),
        ]
    txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    # Sniffer で区切り推定
    try:
        dialect = csv.Sniffer().sniff(txt[:4096], delimiters=",\t;|")
    except Exception:
        dialect = csv.excel
    rows: List[Tuple[str, str]] = []
    r = csv.DictReader(txt.splitlines(), dialect=dialect)
    # ヘッダ想定: symbol,name,(theme…)
    for row in r:
        sym = (row.get("symbol") or row.get("Symbol") or "").strip()
        if not sym or sym.startswith("#"):
            continue
        name = (row.get("name") or row.get("Name") or "").strip()
        rows.append((sym, name))
        if len(rows) >= 20:
            # 仕様上 20銘柄に丸める
            break
    return rows

def sanitize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_url(u: str) -> str:
    """utm 等のクエリノイズを落として重複検出を安定化。"""
    try:
        p = urlparse(u)
        # 許すクエリのみ残す（ほぼ全部消す）
        whitelist = set(["id"])
        q = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True) if k in whitelist]
        return urlunparse((p.scheme, p.netloc.lower(), p.path, "", urlencode(q), ""))
    except Exception:
        return u.strip()

def parse_entry_time(e) -> Optional[datetime]:
    """feedparser entry から日時を取得。UTZ naive に寄せる。"""
    tm = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
    if not tm:
        return None
    try:
        dt = datetime(*tm[:6])
        # UTCに寄せる（タイムゾーンがあれば消えるので“近似”扱い）
        return dt.replace(tzinfo=None)
    except Exception:
        return None

def decay_weight(age_days: float, half_life_days: float) -> float:
    if age_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return 0.0
    # 2^(-age/half_life)
    return 2.0 ** (-age_days / half_life_days)

def safe_minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        # 全部同じ or 無効なら 0 で返す
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

# ===================== Main =====================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    day_dir = OUT_DIR / "data" / REPORT_DATE
    day_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(UNIVERSE_CSV)  # List[(symbol, name)]
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)

    per_symbol_recent_count: Dict[str, int] = {}
    per_symbol_weighted: Dict[str, float] = {}
    articles_by_sym: Dict[str, List[Dict]] = {}

    for (sym, name) in universe:
        q = build_query(sym, name or None)
        url = GOOGLE_NEWS_RSS.format(q=q)

        recent_count = 0
        weighted_sum = 0.0
        seen_keys = set()
        arts: List[Dict] = []

        try:
            feed = feedparser.parse(url)
            entries = list(getattr(feed, "entries", []) or [])
        except Exception:
            entries = []

        # 新しい順に来る保証はないので、日付あり→降順に
        def sort_key(e):
            dt = parse_entry_time(e)
            return dt or datetime(1970, 1, 1)
        entries.sort(key=sort_key, reverse=True)

        for e in entries:
            title = sanitize_text(getattr(e, "title", ""))
            link_raw = sanitize_text(getattr(e, "link", ""))
            if not title or not link_raw:
                continue
            link = normalize_url(link_raw)

            # タイトル+ドメインで緩い重複除外
            dom = urlparse(link).netloc.lower()
            dup_key = (title.lower(), dom)
            if dup_key in seen_keys:
                continue
            seen_keys.add(dup_key)

            dt = parse_entry_time(e)
            if not dt or dt < cutoff:
                # 期間外（または日付不明）はスコアには寄与させない（記事自体は残す）
                arts.append({"title": title, "link": link, "published": None})
                continue

            age_days = (datetime.utcnow() - dt).total_seconds() / 86400.0
            w = decay_weight(age_days, DECAY_HALF_LIFE_DAYS)

            recent_count += 1
            weighted_sum += w

            arts.append({
                "title": title,
                "link": link,
                "published": dt.isoformat(timespec="seconds") + "Z"
            })

            if recent_count >= MAX_PER:
                break

        per_symbol_recent_count[sym] = recent_count
        per_symbol_weighted[sym] = round(weighted_sum, 6)
        articles_by_sym[sym] = arts

        # レート制御
        if SLEEP_SEC > 0:
            time.sleep(SLEEP_SEC)

    # 正規化（weighted_sum ベース）
    weighted_list = [per_symbol_weighted.get(sym, 0.0) for (sym, _name) in universe]
    normalized = safe_minmax_normalize(weighted_list)

    # items を組み立て
    items = []
    for (i, (sym, name)) in enumerate(universe):
        score01 = round(float(normalized[i]), 12) if i < len(normalized) else 0.0
        items.append({
            "symbol": sym,
            "name": name,
            "recent_count": int(per_symbol_recent_count.get(sym, 0)),
            "weighted_count": float(per_symbol_weighted.get(sym, 0.0)),
            "score_0_1": score01,
            "articles": articles_by_sym.get(sym, [])[:MAX_PER],
            "components": { "news": score01 },  # 互換用（fetch_daily 側で参照してもOK）
        })

    payload = {
        "date": REPORT_DATE,
        "items": items,
        "meta": {
            "lookback_days": LOOKBACK_DAYS,
            "max_per_symbol": MAX_PER,
            "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
            "source": "google_news_rss",
            "query_mode": '("SYMBOL" OR "NAME") AND (stock OR company)',
        }
    }

    # 保存
    (OUT_DIR / "data" / "news").mkdir(parents=True, exist_ok=True)
    latest = OUT_DIR / "data" / "news" / "latest.json"
    byday = day_dir / "news.json"
    for p in (latest, byday):
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"[NEWS] saved: {latest} and {byday} (symbols={len(items)})")

if __name__ == "__main__":
    main()
