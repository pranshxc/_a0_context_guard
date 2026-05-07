"""
hist_add_tool_result -> _20_output_offloader

Runs AFTER _10_result_size_cap (which caps & truncates large outputs).

When a tool result is large (> OFFLOAD_THRESHOLD chars), instead of
simply discarding the tail with a blunt truncation message, this extension:
  1. Writes the FULL original output to a file in the session output directory
  2. Replaces the context entry with a rich, compact reference block
     (~150 tokens) that tells the agent exactly what's in the file
     and how to retrieve specific parts via read_file.

This keeps context lean while making zero information permanently lost.

Config env vars:
  CTX_GUARD_OFFLOAD_THRESHOLD   default 3000   chars before offload triggers
  CTX_GUARD_OFFLOAD_DIR         default /tmp/a0_outputs
  CTX_GUARD_ENABLED             default true

The offload only fires when the result has already been marked as
truncated by _10_result_size_cap (contains CONTEXT-GUARD marker) OR
when raw result length exceeds OFFLOAD_THRESHOLD.

Self-contained, zero cross-file imports.
"""
from __future__ import annotations
import os
import json
import re
import time
from typing import Any
from helpers.extension import Extension

OFFLOAD_THRESHOLD = int(os.environ.get("CTX_GUARD_OFFLOAD_THRESHOLD", "3000"))
OFFLOAD_BASE_DIR  = os.environ.get("CTX_GUARD_OFFLOAD_DIR", "/tmp/a0_outputs")
ENABLED           = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")

# Counters per session (in-process, resets on restart - that's fine)
_call_counters: dict[str, int] = {}


class OutputOffloader(Extension):

    def execute(self, data: dict[str, Any] | None = None, **kwargs):
        if not ENABLED or not self.agent or not data or not isinstance(data, dict):
            return
        try:
            result = data.get("tool_result")
            if result is None:
                return

            result_str = (
                result if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False)
            )

            # Only offload if result is large enough to matter
            if len(result_str) <= OFFLOAD_THRESHOLD:
                return

            # Detect the original full size (before _10 truncation)
            # _10 adds a CONTEXT-GUARD marker — if present, we reconstruct
            # the truncated text; if not, we use the raw result.
            full_text = result_str  # may already be truncated by _10

            # Extract tool name for meaningful filename
            tool_name = _extract_tool_name(data) or "tool"

            # Session-scoped output directory
            session_id = _session_id(self.agent)
            out_dir = os.path.join(OFFLOAD_BASE_DIR, session_id)
            os.makedirs(out_dir, exist_ok=True)

            # Monotonic counter per session for unique filenames
            _call_counters[session_id] = _call_counters.get(session_id, 0) + 1
            count = _call_counters[session_id]
            filename = f"{tool_name}_{count:03d}.txt"
            filepath = os.path.join(out_dir, filename)

            # Write full output to file
            with open(filepath, "w", encoding="utf-8", errors="replace") as f:
                f.write(full_text)

            # Build compact reference block for context
            lines = full_text.splitlines()
            total_lines = len(lines)
            total_chars = len(full_text)
            first_5 = "\n    ".join(lines[:5]) if lines else "(empty)"
            last_5  = "\n    ".join(lines[-5:]) if len(lines) > 5 else ""

            # Detect errors/warnings in output
            error_count   = sum(1 for l in lines if re.search(r"\b(error|Error|ERROR|exception|Exception)\b", l))
            warning_count = sum(1 for l in lines if re.search(r"\b(warn|Warn|WARNING|warning)\b", l))

            # Detect exit code if present
            exit_match = re.search(r"exit(?:ed)?\s*(?:code)?\s*[:\s]?\s*(\d+)", full_text, re.IGNORECASE)
            exit_info  = f"exit={exit_match.group(1)}" if exit_match else "exit=unknown"

            # Typed summary for common tools
            typed_summary = _typed_summary(tool_name, lines, total_lines)

            reference_block = (
                f"[OUTPUT OFFLOADED — full output saved to file]\n"
                f"File   : {filepath}\n"
                f"Size   : {total_chars:,} chars | {total_lines:,} lines | {exit_info}\n"
                f"Errors : {error_count} errors, {warning_count} warnings detected\n"
            )
            if typed_summary:
                reference_block += f"Summary: {typed_summary}\n"
            reference_block += (
                f"First 5 lines:\n"
                f"    {first_5}\n"
            )
            if last_5:
                reference_block += (
                    f"Last 5 lines:\n"
                    f"    {last_5}\n"
                )
            reference_block += (
                f"\u2192 To read: use read_file tool with path above.\n"
                f"  Supports: lines=\"all\", lines=\"head:50\", lines=\"tail:50\", lines=\"grep:<pattern>\""
            )

            data["tool_result"] = reference_block

        except Exception:
            pass  # never crash the agent


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


def _extract_tool_name(data: dict) -> str:
    """Best-effort tool name extraction from history data dict."""
    try:
        tool = data.get("tool_name") or data.get("tool") or ""
        if tool:
            return re.sub(r"[^a-z0-9_]", "_", str(tool).lower())[:24]
    except Exception:
        pass
    return "tool"


def _typed_summary(tool_name: str, lines: list[str], total: int) -> str:
    """Generate a typed one-liner summary for common tool outputs."""
    text = "\n".join(lines)
    name = tool_name.lower()

    # apt-get / package install
    if any(k in name for k in ("apt", "install", "dpkg")):
        installed = re.findall(r"newly installed[^\d]*(\d+)", text)
        n = installed[0] if installed else "?"
        return f"APT install: {n} new packages installed."

    # httpx / httprobe / scan output
    if any(k in name for k in ("httpx", "httprobe", "scan", "probe")):
        http_lines = [l for l in lines if re.search(r"https?://", l)]
        return f"HTTP scan: {len(http_lines)} live URLs found in output."

    # nmap
    if "nmap" in name:
        open_ports = [l for l in lines if "open" in l]
        return f"Nmap: {len(open_ports)} open port lines detected."

    # gowitness / aquatone / screenshots
    if any(k in name for k in ("gowitness", "aquatone", "screenshot")):
        shots = re.findall(r"(\d+)\s*screenshot", text, re.IGNORECASE)
        n = shots[0] if shots else "?"
        return f"Screenshots: {n} captured."

    # python / code execution
    if any(k in name for k in ("code", "python", "exec", "terminal", "bash")):
        err = sum(1 for l in lines if re.search(r"\b(Error|Traceback|error)\b", l))
        return f"Code exec: {total} lines output, {err} error lines."

    return ""  # no typed summary for unknown tools
