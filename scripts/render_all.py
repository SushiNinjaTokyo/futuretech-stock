from __future__ import annotations

"""Render static Neon Tokyo pages from already-generated AI Arena JSON.

This aggregator is intentionally render-only. It must not fetch prices, rebuild
signals, run backtests, or mutate DuckDB. Data-generation workflows should run
before this script.

Daily / Weekly renderers are deliberately excluded. Those pages are being
retired and must not be reintroduced as dependencies of the AI Arena front door.
AI_LAB / discussion data is not deleted or pruned here; this script simply does
not require Daily or Weekly outputs to render the public AI Arena surface.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Order matters:
# - render_index.py reads AI Arena hero/summary JSON and prices-jp/latest.json.
# - AI Arena pages read JSON exported by scripts/lib/arena_exporter_jp.py.
# - render_static_pages.py renders non-market static pages such as disclaimer and privacy.
RENDER_SCRIPTS = [
    "render_index.py",
    "render_ai_arena_summary_jp.py",
    "render_ai_arena_ranking_jp.py",
    "render_ai_arena_positions_jp.py",
    "render_ai_arena_jp.py",
    "render_ai_arena_war_room_jp.py",
    "render_ai_arena_signals_jp.py",
    "render_ai_agent_profiles_jp.py",
    "render_static_pages.py",
]


def main() -> int:
    missing_required: list[str] = []

    for script in RENDER_SCRIPTS:
        path = SCRIPTS / script
        if not path.exists():
            missing_required.append(script)
            continue

        print(f"==> {script}")
        subprocess.run([sys.executable, str(path)], check=True)

    if missing_required:
        for script in missing_required:
            print(f"missing required renderer: {script}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
