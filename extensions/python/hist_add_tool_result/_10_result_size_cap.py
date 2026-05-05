"""
hist_add_tool_result  →  _10_result_size_cap

Fires BEFORE a tool result is written into agent history.
Truncates oversized results to MAX_RESULT_CHARS while preserving the most
valuable part (the beginning) and appending a transparent audit marker.

Design goals
────────────
• Never silently drop data — always append a [TRUNCATED] marker with byte count.
• Preserve structured openings: if the result looks like JSON or XML, keep the
  first structural lines intact so the agent knows the schema even after trim.
• Zero latency — pure string operation, no LLM calls.
• Fault-tolerant — any exception is swallowed; the original result passes through.
"""
from __future__ import annotations
import json
from typing import Any
from helpers.extension import Extension

# lazy import so the plugin is importable before agent0 is fully booted
def _cfg():
    try:
        from helpers.config import get  # local to plugin dir
        return get
    except ImportError:
        import os
        return lambda k, d=None: os.environ.get(f"CTX_GUARD_{k.upper()}", d)


class ResultSizeCap(Extension):
    """
    Caps every incoming tool result at `max_result_chars` characters.
    """

    def execute(self, data: dict[str, Any] | None = None, **kwargs):  # type: ignore[override]
        if not self.agent:
            return
        if not data or not isinstance(data, dict):
            return

        try:
            cfg = _cfg()
            if not cfg("enabled", True):
                return

            max_chars: int = int(cfg("max_result_chars", 4000))
            max_chars = max(500, max_chars)  # floor safety

            result = data.get("tool_result")
            if result is None:
                return

            result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

            if len(result_str) <= max_chars:
                return  # within budget — do nothing

            omitted = len(result_str) - max_chars
            truncated = result_str[:max_chars]

            # Attempt to close any open JSON bracket so the agent doesn't see
            # malformed JSON that could confuse parsing.
            truncated = _try_close_json(truncated)

            marker = (
                f"\n\n[CONTEXT-GUARD: {omitted:,} characters truncated to save context window.\n"
                f" Full result saved to disk if file-save extension is active.\n"
                f" The leading {max_chars:,} characters above contain the most relevant content.]"
            )
            data["tool_result"] = truncated + marker

        except Exception:
            # Never crash the agent — silently pass through original data
            pass


def _try_close_json(text: str) -> str:
    """Best-effort: if text looks like JSON, try to close open brackets."""
    stripped = text.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return text  # not JSON — return as-is
    open_curly = stripped.count("{") - stripped.count("}")
    open_square = stripped.count("[") - stripped.count("]")
    if open_curly <= 0 and open_square <= 0:
        return text  # already balanced
    # Close in reverse order of expected nesting
    closer = "" 
    if open_square > 0:
        closer += "]" * min(open_square, 5)
    if open_curly > 0:
        closer += "}" * min(open_curly, 5)
    return text + "\n" + closer
