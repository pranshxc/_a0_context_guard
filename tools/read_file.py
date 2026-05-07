"""
read_file.py — Fix 2C

Companion tool for the output offloader (_20_output_offloader.py).

Allows the agent to retrieve offloaded large outputs on demand, with
surgical access modes so it never has to dump the whole file into context.

Usage:
  { "action": "read",   "path": "/tmp/a0_outputs/.../tool_001.txt" }
  { "action": "head",   "path": "...", "lines": 50 }
  { "action": "tail",   "path": "...", "lines": 50 }
  { "action": "grep",   "path": "...", "pattern": "error", "lines": 30 }
  { "action": "list",   "path": "/tmp/a0_outputs/<session_id>/" }
  { "action": "stats",  "path": "..." }

All returned content is capped at CTX_GUARD_READ_FILE_MAX_CHARS (default 6000)
to prevent accidentally flooding context when reading large files.

Self-contained, zero cross-plugin imports.
"""
from __future__ import annotations
import os
import re
from helpers.tool import Tool, Response

MAX_CHARS = int(os.environ.get("CTX_GUARD_READ_FILE_MAX_CHARS", "6000"))
ALLOWED_BASE_DIRS = [
    "/tmp/a0_outputs",
    os.path.join("work",  "todo"),
    os.path.join("work",  "session_notes"),
    "recon",
    "/tmp",
]


class ReadFile(Tool):

    async def execute(self, **kwargs) -> Response:
        action  = (self.args.get("action") or "read").strip().lower()
        path    = (self.args.get("path")   or "").strip()

        if not path:
            return Response(message="Provide a 'path'.", break_loop=False)

        # Security: only allow reads within known safe directories
        abs_path = os.path.realpath(path)
        if not _is_allowed(abs_path):
            return Response(
                message=(
                    f"Path '{path}' is outside allowed directories. "
                    f"Allowed: {ALLOWED_BASE_DIRS}"
                ),
                break_loop=False,
            )

        # ── list directory ──────────────────────────────────────────────────────
        if action == "list" or os.path.isdir(abs_path):
            if not os.path.isdir(abs_path):
                return Response(message=f"'{path}' is not a directory.", break_loop=False)
            try:
                entries = sorted(os.listdir(abs_path))
                lines = [f"{i+1:>3}. {e}" for i, e in enumerate(entries)]
                return Response(
                    message=f"Directory: {path} ({len(entries)} files)\n" + "\n".join(lines),
                    break_loop=False,
                )
            except OSError as e:
                return Response(message=f"Cannot list '{path}': {e}", break_loop=False)

        # ── file must exist from here ────────────────────────────────────────────────
        if not os.path.isfile(abs_path):
            return Response(message=f"File not found: '{path}'", break_loop=False)

        try:
            file_size = os.path.getsize(abs_path)
        except OSError:
            file_size = 0

        # ── stats (no content) ───────────────────────────────────────────────────
        if action == "stats":
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                return Response(
                    message=(
                        f"File: {path}\n"
                        f"Size: {file_size:,} bytes | {len(lines):,} lines"
                    ),
                    break_loop=False,
                )
            except OSError as e:
                return Response(message=f"Cannot stat '{path}': {e}", break_loop=False)

        # ── read file content ────────────────────────────────────────────────────
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except OSError as e:
            return Response(message=f"Cannot read '{path}': {e}", break_loop=False)

        total_lines = len(all_lines)
        n = max(1, int(self.args.get("lines", 50)))

        if action == "head":
            selected = all_lines[:n]
            label    = f"head:{n}"
        elif action == "tail":
            selected = all_lines[-n:]
            label    = f"tail:{n}"
        elif action == "grep":
            pattern = (self.args.get("pattern") or "").strip()
            if not pattern:
                return Response(message="Provide a 'pattern' for grep.", break_loop=False)
            try:
                regex    = re.compile(pattern, re.IGNORECASE)
                matched  = [l for l in all_lines if regex.search(l)]
            except re.error as e:
                return Response(message=f"Invalid regex pattern: {e}", break_loop=False)
            selected = matched[:n]
            label    = f"grep:'{pattern}' — {len(matched)} matches, showing {len(selected)}"
        else:  # action == "read" or anything else — read all, capped
            selected = all_lines
            label    = "read:all"

        # Cap output to MAX_CHARS
        content = "".join(selected)
        truncated = False
        if len(content) > MAX_CHARS:
            content   = content[:MAX_CHARS]
            truncated = True

        header = (
            f"File: {path} | {total_lines:,} lines total | {label}\n"
            f"{'[TRUNCATED at ' + str(MAX_CHARS) + ' chars — use head/tail/grep for targeted access]' if truncated else ''}\n"
        )
        return Response(message=header + content, break_loop=False)


def _is_allowed(abs_path: str) -> bool:
    """Check path is inside one of the allowed base directories."""
    for base in ALLOWED_BASE_DIRS:
        abs_base = os.path.realpath(base) if not base.startswith("/") else base
        if abs_path.startswith(abs_base):
            return True
    return False
