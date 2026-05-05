"""
hist_add_tool_result -> _10_result_size_cap

Fires BEFORE a tool result is written into agent history.
Self-contained: all logic inlined, no cross-file imports.
"""
from __future__ import annotations
import os
import json
from typing import Any
from helpers.extension import Extension

# ── inline config ─────────────────────────────────────────────────────────────
def _cfg(key: str, default):
    val = os.environ.get(f"CTX_GUARD_{key.upper()}")
    if val is None:
        return default
    try:
        if isinstance(default, bool):  return val.lower() in ("1", "true", "yes")
        if isinstance(default, int):   return int(val)
        if isinstance(default, float): return float(val)
    except (ValueError, TypeError):
        pass
    return val

# ── extension ─────────────────────────────────────────────────────────────────
class ResultSizeCap(Extension):

    def execute(self, data: dict[str, Any] | None = None, **kwargs):
        if not self.agent or not data or not isinstance(data, dict):
            return
        try:
            if not _cfg("enabled", True):
                return

            max_chars = max(500, int(_cfg("max_result_chars", 4000)))
            result = data.get("tool_result")
            if result is None:
                return

            result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

            if len(result_str) <= max_chars:
                return  # within budget

            omitted   = len(result_str) - max_chars
            truncated = _close_json(result_str[:max_chars])
            marker    = (
                f"\n\n[CONTEXT-GUARD: {omitted:,} characters truncated. "
                f"Leading {max_chars:,} characters contain the most relevant content. "
                f"Full result saved to disk if file-save extension is active.]"
            )
            data["tool_result"] = truncated + marker

        except Exception:
            pass  # never crash the agent


def _close_json(text: str) -> str:
    """Best-effort: close open JSON brackets so the truncated output stays parseable."""
    s = text.strip()
    if not s or s[0] not in ("{", "["):
        return text
    open_c = s.count("{") - s.count("}")
    open_s = s.count("[") - s.count("]")
    if open_c <= 0 and open_s <= 0:
        return text
    return text + "\n" + ("]" * min(open_s, 5)) + ("}" * min(open_c, 5))
