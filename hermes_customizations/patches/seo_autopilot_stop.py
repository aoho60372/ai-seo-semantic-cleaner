"""Bounded stop guard for the local model-driven SEO workflow.

This is deliberately opt-in: it activates only in E:\\AI\\seo when that
workspace's AGENTS.md carries the explicit marker.  It never runs commands or
touches data; it only gives Hermes another internal turn when the model tries
to stop while `seo_workflow next` has recorded an unfinished state.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.runtime_cwd import resolve_agent_cwd


MARKER = "HERMES_SEO_AUTOPILOT"
MAX_NUDGES = 120


def build_seo_autopilot_nudge(*, attempts: int = 0) -> str | None:
    """Return an internal continuation only for a recorded unfinished SEO job."""
    if attempts >= MAX_NUDGES:
        return None
    try:
        root = resolve_agent_cwd().resolve()
        marker_file = root / "AGENTS.md"
        if root != Path(r"E:\AI\seo") or MARKER not in marker_file.read_text(encoding="utf-8"):
            return None
        states = sorted((root / "jobs").glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not states:
            return None
        state = json.loads(states[0].read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("autopilot_last_status") != "continue":
            return None
        if state.get("stage") in {"quality_review", "knowledge_saved", "finalized", "learned"}:
            return None
        job = states[0].parent.name
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return (
        "[SEO autopilot: the workflow is still unfinished. Do not send a progress message "
        "or wait for the user. Immediately run `next --job \"%s\" --quiet`, follow its "
        "returned command, and continue the loop. Stop only if `next` returns stop or blocked.]"
    ) % job
