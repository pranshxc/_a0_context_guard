"""
monologue_end -> _10_session_notes

Runs at the END of every agent turn (before _80_hard_compact_guard).

Purpose: Maintain a persistent session_notes.json file in the work
directory that survives compaction. This file stores:
  - artifact_registry : file paths produced during the session
  - tool_notes        : version quirks, syntax gotchas per tool
  - failed_approaches : dead paths that must not be retried
  - thought_hash      : hash of last assistant thought for deduplication

The agent can read this file at the start of a new phase to instantly
recover lost context without re-discovering it through failed tool calls.

This extension WRITES to the notes file by scanning recent history for
patterns (file saves, tool installs, errors) and extracting them.
It does NOT add anything to the LLM context directly.

Self-contained, zero cross-file imports.
"""
from __future__ import annotations
import os
import re
import json
import hashlib
from helpers.extension import Extension
from agent import LoopData

NOTES_DIR  = os.path.join("work", "session_notes")
ENABLED    = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")


def _notes_path(session_id: str) -> str:
    os.makedirs(NOTES_DIR, exist_ok=True)
    return os.path.join(NOTES_DIR, f"{session_id}.json")


def _load_notes(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "artifact_registry": {},
        "tool_notes": {},
        "failed_approaches": [],
        "last_thought_hash": "",
    }


def _save_notes(path: str, notes: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _session_id(agent) -> str:
    try:
        sid = (
            getattr(agent, "chat_id", None)
            or getattr(agent.context, "id", None)
        )
        if sid:
            return str(sid)[:32]
    except Exception:
        pass
    return "default"


def _extract_history_text(agent) -> str:
    """Get recent history as flat text for pattern scanning."""
    try:
        from helpers.history import output_text
        return output_text(agent.history.output())[-8000:]  # last 8k chars only
    except Exception:
        pass
    try:
        msgs = agent.history.output()
        return " ".join(str(m) for m in msgs[-10:])  # last 10 messages
    except Exception:
        return ""


def _scan_artifacts(text: str, existing: dict) -> dict:
    """Extract file paths that look like agent-produced artifacts."""
    updated = dict(existing)
    # Match paths like /home/user/..., /tmp/..., recon/..., work/...
    patterns = re.findall(
        r"(?:saved?|written?|output|created?|file)\s*(?:to|at|:)?\s*"
        r"([/~][\w./\-]+|recon/[\w./\-]+|work/[\w./\-]+)",
        text, re.IGNORECASE
    )
    for p in patterns:
        p = p.strip().rstrip(",.")
        if len(p) > 5 and "." in p.split("/")[-1]:
            if p not in updated:
                updated[p] = "auto-detected"
    return updated


def _scan_tool_notes(text: str, existing: dict) -> dict:
    """Detect tool version/syntax notes from history."""
    updated = dict(existing)
    # gowitness v3 syntax change
    if "gowitness" in text and "scan file" in text:
        updated["gowitness"] = "v3 installed (apt). Syntax: gowitness scan file -f <urls.txt>"
    if "gowitness" in text and "command not found" in text.lower():
        if "gowitness" not in updated:
            updated["gowitness"] = "WARNING: gowitness not found on PATH"
    # playwright
    if "playwright" in text and "available" in text:
        updated["playwright"] = "playwright available (chromium backend)"
    return updated


def _scan_failed_approaches(text: str, existing: list) -> list:
    """Detect failed approaches that should not be retried."""
    updated = list(existing)
    candidates = [
        (r"go install.*gowitness.*failed", "go install github.com/sensepost/gowitness@latest (Go path not configured)"),
        (r"command not found.*gowitness", "gowitness via go install (use apt instead)"),
        (r"No such file.*shodan", "shodan CLI not installed (use shodan Python lib)"),
    ]
    for pattern, note in candidates:
        if re.search(pattern, text, re.IGNORECASE) and note not in updated:
            updated.append(note)
    return updated


def _thought_hash(agent) -> str:
    """Hash the last assistant thought block for deduplication tracking."""
    try:
        msgs = agent.history.output()
        for m in reversed(msgs):
            content = str(m)
            if "thoughts" in content.lower() or "headline" in content.lower():
                return hashlib.md5(content[:500].encode()).hexdigest()
    except Exception:
        pass
    return ""


class SessionNotes(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            agent      = self.agent
            session_id = _session_id(agent)
            path       = _notes_path(session_id)
            notes      = _load_notes(path)
            text       = _extract_history_text(agent)

            notes["artifact_registry"]  = _scan_artifacts(text, notes["artifact_registry"])
            notes["tool_notes"]         = _scan_tool_notes(text, notes["tool_notes"])
            notes["failed_approaches"]  = _scan_failed_approaches(text, notes["failed_approaches"])
            notes["last_thought_hash"]  = _thought_hash(agent)

            _save_notes(path, notes)
        except Exception:
            pass  # never crash the agent
