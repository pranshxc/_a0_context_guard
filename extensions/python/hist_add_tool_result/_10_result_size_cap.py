"""
hist_add_tool_result  ->  _10_result_size_cap

Fires BEFORE a tool result is written into agent history.
Truncates oversized results to max_result_chars while preserving the most
valuable leading content and appending a transparent audit marker so the
agent knows exactly what happened.

Design goals
------------
- Zero latency: pure string operation, no LLM call.
- Never silently drop data: always append [CONTEXT-GUARD] marker with byte count.
- Best-effort JSON bracket closing so the agent sees valid structure.
- Full fault-tolerance: any exception is swallowed; original result passes through.

Import pattern: extension files inside a plugin MUST import plugin-local
helpers using the full dotted path  plugins.<plugin_name>.helpers.*
because Python resolves bare `from helpers.x` against A0's own helpers/ pkg.
"""
from __future__ import annotations
import json
from typing import Any
from helpers.extension import Extension


def _get_cfg():
    """Lazy-load plugin config; return a no-op getter on failure."""
    try:
        from plugins._a0_context_guard.helpers.config import get
        return get
    except Exception:
        import os
        return lambda k, d=None: os.environ.get(f"CTX_GUARD_{k.upper()}", d)


class ResultSizeCap(Extension):
    """Caps every incoming tool result at `max_result_chars` characters."""

    def execute(self, data: dict[str, Any] | None = None, **kwargs):  # type: ignore[override]
        if not self.agent:
            return
        if not data or not isinstance(data, dict):
            return

        try:
            cfg = _get_cfg()
            if not cfg("enabled", True):
                return

            max_chars: int = max(500, int(cfg("max_result_chars", 4000)))

            result = data.get("tool_result")
            if result is None:
                return

            result_str = (
                result if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False)
            )

            if len(result_str) <= max_chars:
                return  # within budget - nothing to do

            omitted = len(result_str) - max_chars
            truncated = _try_close_json(result_str[:max_chars])

            marker = (
                f"\n\n[CONTEXT-GUARD: {omitted:,} characters truncated to save "
                f"context window. The {max_chars:,} leading characters above contain "
                f"the most relevant content. Full result saved to disk if the "
                f"file-save extension is active.]"
            )
            data["tool_result"] = truncated + marker

        except Exception:
            pass  # never crash the agent


def _try_close_json(text: str) -> str:
    """Best-effort: close open JSON brackets so output stays parseable."""
    stripped = text.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return text
    open_curly  = stripped.count("{") - stripped.count("}")
    open_square = stripped.count("[") - stripped.count("]")
    if open_curly <= 0 and open_square <= 0:
        return text
    closer = ""
    if open_square > 0:
        closer += "]" * min(open_square, 5)
    if open_curly > 0:
        closer += "}" * min(open_curly, 5)
    return text + "\n" + closer
