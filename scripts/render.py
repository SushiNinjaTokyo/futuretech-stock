# scripts/render.py
from __future__ import annotations
import json, os, sys, datetime, pathlib, shutil
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = pathlib.Path(__file__).resolve().parents[1] if (pathlib.Path(__file__).name == "render.py") else pathlib.Path.cwd()
SCRIPTS = ROOT / "scripts"
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = ROOT / "site"

REPORT_DATE = os.environ.get("REPORT_DATE")  # e.g. 2025-09-06

CANON_MAP = {
    # 異なる世代のキー名を正規化
    "price_vol_anom": "volume_anomaly",
    "vol_anom": "volume_anomaly",
    "vol_anomaly": "volume_anomaly",
    "news_coverage": "news",
    "news_score": "news",          # 一部の生成で score 名を渡しているケースに合わせる
    "dii": "insider_momo",
    "insider": "insider_momo",
    "insider_momentum": "insider_momo",
    "trends": "trends_breakout",
    "trends_peak": "trends_breakout",
    "trends_breakout": "trends_breakout",
    "volume_anomaly": "volume_anomaly",
    "news": "news",
    "insider_momo": "insider_momo",
}

def now_utc_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

def read_json(p: pathlib.Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_text(p: pathlib.Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def copy_asset(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def pick_date_dir() -> str:
    """
    REPORT_DATE があればそれを、なければ data 配下の最新日付ディレクトリを使う
    """
    if REPORT_DATE:
        return REPORT_DATE
    data_dir = OUT_DIR / "data"
    if not data_dir.exists():
        raise SystemExit("[ERROR] site/data が見つかりません")
    # YYYY-MM-DD の最大
    cand = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 10], reverse=True)
    if not cand:
        raise SystemExit("[ERROR] site/data/YYYY-MM-DD が見つかりません")
    return cand[0]

def canonize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        ck = CANON_MAP.get(k, k)
        out[ck] = v
    return out

def fix_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    - components / weights のキー名を正規化して整合
    - 対象コンポーネントに対する weight 合計が 0 のときは等分配でフォールバック
    - chart_url / score_pts などテンプレ側で使う値を安全化
    """
    date = item.get("date") or REPORT_DATE or ""
    sym = item.get("symbol") or item.get("ticker") or ""
    comps_raw = item.get("score_components") or {}
    weights_raw = item.get("score_weights") or {}

    comps = canonize_keys(comps_raw)
    weights = canonize_keys(weights_raw)

    # このカードに存在するキーのみに限定して重み合計をとる
    keys = [k for k, v in comps.items() if v is not None]
    wsum = sum(float(max(0.0, weights.get(k, 0.0))) for k in keys)

    if wsum <= 0 and keys:
        # 等分フォールバック（確実に 0 にならないように）
        eq = 1.0 / len(keys)
        weights = {k: (eq if k in keys else 0.0) for k in set(list(weights.keys()) + keys)}

    # グラフ URL（存在チェックまではしない。静的生成に任せる）
    chart_url = item.get("chart_url") or f"/charts/{date}/{sym}.png"

    # スコア（ポイント表示用）
    final01 = float(item.get("final_score_0_1") or 0.0)
    score_pts = int(round(final01 * 1000))

    fixed = dict(item)
    fixed.update({
        "symbol": sym,
        "score_components": comps,
        "score_weights": weights,
        "chart_url": chart_url,
        "score_pts": score_pts,
        "final_score_0_1": final01,
        # 旧世代キーがあっても JS から参照できるよう保険で併記
        "news_score": item.get("news_score", comps.get("news")),
        "trends_breakout": item.get("trends_breakout", comps.get("trends_breakout")),
        "vol_anomaly_score": item.get("vol_anomaly_score", comps.get("volume_anomaly")),
        "insider_momo": item.get("insider_momo", comps.get("insider_momo")),
    })
    return fixed

def main() -> None:
    date = pick_date_dir()
    print(f"{now_utc_iso()} [INFO] Render target date: {date}")

    # 入力 JSON（fetch_daily が生成）
    top10_path = OUT_DIR / "data" / date / "top10.json"
    if not top10_path.exists():
        # 互換: 旧パス
        alt = OUT_DIR / "data" / "top10.json"
        top10_path = alt if alt.exists() else top10_path
    if not top10_path.exists():
        raise SystemExit(f"[ERROR] top10.json が見つかりません: {top10_path}")

    top10_raw = read_json(top10_path)
    if isinstance(top10_raw, dict) and "items" in top10_raw:
        items = top10_raw["items"]
    else:
        items = top10_raw

    fixed_items = [fix_item(x) for x in items]

    # Jinja2
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=False,
        trim_blocks=True, lstrip_blocks=True,
    )
    tpl = env.get_template("daily.html.j2")

    html = tpl.render(
        date=date,
        top10=fixed_items,
        generated_at=now_utc_iso(),
    )

    # 出力
    out_html = OUT_DIR / "daily" / f"{date}.html"
    write_text(out_html, html)
    print(f"{now_utc_iso()} [INFO] Rendered daily HTML: {out_html} (template=templates/daily.html.j2)")

    # CSS を公開ディレクトリへコピー
    css_src = TEMPLATES_DIR / "daily.css"
    if not css_src.exists():
        # 互換: scripts/ と同階層に置いた場合
        alt = ROOT / "daily.css"
        css_src = alt if alt.exists() else css_src
    if css_src.exists():
        css_dst = OUT_DIR / "assets" / "daily.css"
        copy_asset(css_src, css_dst)
        print(f"{now_utc_iso()} [INFO] [ASSET] copied CSS -> {css_dst}")
    else:
        print(f"{now_utc_iso()} [WARN] CSS source not found: {css_src}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{now_utc_iso()} [ERROR] FATAL in render: {e}")
        raise
