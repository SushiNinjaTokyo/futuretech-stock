#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yfinance as yf

OUT_DIR = Path(os.getenv("OUT_DIR", "site"))
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))
REPORT_DATE = os.getenv("REPORT_DATE") or datetime.utcnow().date().isoformat()

WEIGHT_VOL_ANOM = float(os.getenv("WEIGHT_VOL_ANOM", "0.25"))
WEIGHT_DII      = float(os.getenv("WEIGHT_DII", "0.25"))
WEIGHT_TRENDS   = float(os.getenv("WEIGHT_TRENDS", "0.30"))
WEIGHT_NEWS     = float(os.getenv("WEIGHT_NEWS", "0.20"))
TOTAL_W = max(1e-9, WEIGHT_VOL_ANOM + WEIGHT_DII + WEIGHT_TRENDS + WEIGHT_NEWS)

def load_universe(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return ["NVDA","MSFT","PLTR","AI","ISRG","TER","SYM","RKLB","IRDM","VSAT",
                "INOD","SOUN","MNDY","AVAV","PERF","GDRX","ABCL","U","TEM","VRT"]
    out: List[str] = []
    for line in csv_path.read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#"): continue
        out.append(t.split(",")[0].strip())
    return out[:20]

def load_payload(p: Path) -> Dict[str, Any]:
    """
    JSONが list だったり dict だったりする不整合を吸収。
    期待形: {"date":..., "items":[{"symbol":...,"score_0_1":..., "components":{...}}]}
    """
    if not p.exists():
        return {"date": REPORT_DATE, "items": []}
    obj = json.loads(p.read_text() or "{}")
    # もし list なら、古い形式: items = list, date=REPORT_DATE に矯正
    if isinstance(obj, list):
        return {"date": REPORT_DATE, "items": obj}
    if isinstance(obj, dict):
        # items がない/不正 -> 矯正
        items = obj.get("items", [])
        if isinstance(items, dict):
            # 万一 dict なら values を使う
            items = list(items.values())
        if not isinstance(items, list):
            items = []
        obj["items"] = items
        if not obj.get("date"):
            obj["date"] = REPORT_DATE
        return obj
    # それ以外も矯正
    return {"date": REPORT_DATE, "items": []}

def rank_percentiles(values: List[float]) -> List[float]:
    if not values:
        return []
    order = sorted(values, reverse=True)
    mp, i, n = {}, 0, len(order)
    while i < n:
        j = i
        while j < n and order[j] == order[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        pct = 1.0 - (avg_rank / max(1, n - 1)) if n > 1 else 1.0
        mp[order[i]] = pct
        i = j
    return [mp[v] for v in values]

def price_volume_anomaly(symbols: List[str]) -> Dict[str, float]:
    """
    近3か月の出来高から近似アノマリーを作る。
    - 無効ティッカーは静かに0点扱い（ログ1行に集約）
    - yfinanceの雑多なstderr出力を抑制
    - 単一銘柄戻り/複数銘柄MultiIndexの両対応
    """
    import io
    import sys
    import contextlib
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from datetime import datetime, timedelta

    if not symbols:
        return {}

    # yfinanceの冗長メッセージを抑制
    fake_out, fake_err = io.StringIO(), io.StringIO()
    end = datetime.utcnow().date()
    start = end - timedelta(days=150)

    with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_err):
        data = yf.download(
            tickers=symbols,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )

    scores: Dict[str, float] = {}
    missing: List[str] = []

    # data の形状に応じて各銘柄の Series を取り出すヘルパ
    def pick_df(sym: str):
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # 複数銘柄: 列が MultiIndex (('NVDA','Open'),...)
                if sym in data.columns.get_level_values(0):
                    return data[sym]
                raise KeyError(sym)
            else:
                # 単一銘柄: そのままDataFrame
                if len(symbols) == 1:
                    return data
                # 単一銘柄しか返ってこなかったが、symがそれ以外 → 欠損扱い
                raise KeyError(sym)
        except Exception:
            raise KeyError(sym)

    for s in symbols:
        try:
            df = pick_df(s)
            vol = df.get("Volume")
            if vol is None:
                raise KeyError("Volume")
            vol = vol.dropna()
            if len(vol) < 10:
                scores[s] = 0.0
                continue
            recent = float(vol.tail(5).mean())
            base = float(vol.mean())
            r = (recent / base) if base > 0 else 0.0
            # 2倍で満点に正規化し、[0,1]へクリップ
            scores[s] = max(0.0, min(r / 2.0, 1.0))
        except Exception:
            scores[s] = 0.0
            missing.append(s)

    # パーセンタイル（降順）
    vals = [scores.get(s, 0.0) for s in symbols]
    order = sorted(vals, reverse=True)
    mp: Dict[float, float] = {}
    i, n = 0, len(order)
    while i < n:
        j = i
        while j < n and order[j] == order[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        pct = 1.0 - (avg_rank / max(1, n - 1)) if n > 1 else 1.0
        mp[order[i]] = pct
        i = j
    pcts = {s: mp[v] for s, v in zip(symbols, vals)}

    # まとめて1行だけ情報を出す（CIログがうるさくならないように）
    if missing:
        print(f"[fetch_daily] yfinance missing/invalid tickers treated as 0: {sorted(set(missing))}")

    return pcts

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    day_dir = OUT_DIR / "data" / REPORT_DATE
    day_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(UNIVERSE_CSV)

    dii = load_payload(day_dir / "dii.json")            # volume アノマリー（別名DII）
    trends = load_payload(day_dir / "trends.json")
    news = load_payload(day_dir / "news.json")

    score_by = {
        "dii": {x["symbol"]: float(x.get("score_0_1", 0.0)) for x in dii.get("items", [])},
        "trends": {x["symbol"]: float(x.get("score_0_1", 0.0)) for x in trends.get("items", [])},
        "news": {x["symbol"]: float(x.get("score_0_1", 0.0)) for x in news.get("items", [])},
    }

    # 価格由来の出来高アノマリー（ネットワーク等で失敗しても 0 埋め）
    try:
        import pandas as pd  # 遅延インポート：yfinance の戻り処理で使う
        pva = price_volume_anomaly(universe)
    except Exception:
        pva = {s: 0.0 for s in universe}

    items = []
    for s in universe:
        sv = float(score_by["dii"].get(s, 0.0))
        st = float(score_by["trends"].get(s, 0.0))
        sn = float(score_by["news"].get(s, 0.0))
        sp = float(pva.get(s, 0.0))
        total = (
            WEIGHT_VOL_ANOM * sp +
            WEIGHT_DII * sv +
            WEIGHT_TRENDS * st +
            WEIGHT_NEWS * sn
        ) / TOTAL_W
        items.append({
            "symbol": s,
            "score_0_1": round(max(0.0, min(total, 1.0)), 12),
            "components": {
                "price_vol_anom": round(sp, 12),
                "dii": round(sv, 12),
                "trends": round(st, 12),
                "news": round(sn, 12),
            }
        })

    # 高い順で上位10件
    items_sorted = sorted(items, key=lambda x: x["score_0_1"], reverse=True)[:10]

    payload = {
        "date": REPORT_DATE,
        "universe_size": len(universe),
        "top10": items_sorted,
        "weights": {
            "price_vol_anom": WEIGHT_VOL_ANOM,
            "dii": WEIGHT_DII,
            "trends": WEIGHT_TRENDS,
            "news": WEIGHT_NEWS
        }
    }

    (OUT_DIR / "data" / REPORT_DATE).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data" / REPORT_DATE / "top10.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    print(f"Generated top10 for {REPORT_DATE}: {len(items_sorted)} symbols (universe={len(universe)})")

if __name__ == "__main__":
    main()
