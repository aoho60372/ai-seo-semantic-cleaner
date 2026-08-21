"""Bounded stop guard for the local model-driven SEO workflow.

This is deliberately opt-in: it activates only in an open project whose
AGENTS.md carries the explicit marker and which contains the SEO workflow
entry points. It never runs commands or touches data; it only gives Hermes
another internal turn when the model tries to stop while `seo_workflow next`
has recorded an unfinished state.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.runtime_cwd import resolve_agent_cwd


MARKER = "HERMES_SEO_AUTOPILOT"
# 50-row batches need fewer than ten nudges even for a dirty core.  A lower
# bound limits cloud-token damage if a model ignores a terminal workflow state.
MAX_NUDGES = 30


def build_seo_autopilot_nudge(*, attempts: int = 0) -> str | None:
    """Return an internal continuation only for a recorded unfinished SEO job."""
    if attempts >= MAX_NUDGES:
        return None
    try:
        root = resolve_agent_cwd().resolve()
        marker_file = root / "AGENTS.md"
        if (
            not marker_file.is_file()
            or MARKER not in marker_file.read_text(encoding="utf-8")
            or not (root / "seo_workflow.py").is_file()
            or not (root / "seo.ps1").is_file()
        ):
            return None
        states = sorted((root / "jobs").glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not states:
            return None
        state = json.loads(states[0].read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("autopilot_last_status") != "continue":
            return None
        if state.get("run_error") or state.get("blocked_reason"):
            return None
        if state.get("stage") in {"quality_review", "knowledge_saved", "finalized", "learned"}:
            return None
        job = states[0].parent.name
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return (
        "[SEO autopilot: the workflow is still unfinished. Do not send a progress message "
        "or wait for the user. Immediately run `\"./.venv/Scripts/python.exe\" "
        "\"./seo_workflow.py\" next --job \"%s\" --quiet`, follow its "
        "returned command, and continue the loop. Stop only if `next` returns stop or blocked.]"
    ) % job
