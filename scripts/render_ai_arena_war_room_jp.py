from __future__ import annotations

"""Render the standalone AI Arena War Room page.

The page is intentionally independent from the legacy Arena Log inside
`/japan/ai-arena/`.  It reads the new `war-room/latest.json` payload generated
by `scripts/build_ai_arena_war_room_jp.py`, renders a static HTML shell, and
copies page-specific CSS/JS assets.
"""

import os
from pathlib import Path
from typing import Any

from render_common import OUT_DIR, copy_asset, env, read_json, write_text

WAR_ROOM_JSON = Path(os.getenv("AI_ARENA_WAR_ROOM_JSON", str(OUT_DIR / "data/japan/ai-arena/war-room/latest.json")))


def _fallback_payload() -> dict[str, Any]:
    return {
        "schema_version": "ai_arena_war_room_empty",
        "generated_at": "—",
        "page": {
            "title": "AI Arena War Room",
            "subtitle": "Seven trading agents debate Japan equities using live simulation evidence.",
        },
        "daily_brief": {
            "headline": "War Room data has not been generated yet.",
            "summary": "Run scripts/build_ai_arena_war_room_jp.py after AI Arena JSON export.",
            "bullets": [],
        },
        "agents": [],
        "ranking": [],
        "open_positions": [],
        "threads": [],
        "feed": [],
        "pulse": [],
        "metrics": {},
        "ai": {"enabled": False, "status": "missing_payload"},
        "disclaimer": "AI Arena is a quantitative simulation and discussion interface. Informational only. Not investment advice.",
    }


def main() -> None:
    payload = read_json(WAR_ROOM_JSON, _fallback_payload()) or _fallback_payload()
    html = env().get_template("ai_arena_war_room_jp.html.j2").render(war_room=payload)
    write_text(OUT_DIR / "japan/ai-arena/log/index.html", html)
    copy_asset("ai_arena_war_room_jp.css")
    copy_asset("ai_arena_war_room_jp.js")


if __name__ == "__main__":
    main()
