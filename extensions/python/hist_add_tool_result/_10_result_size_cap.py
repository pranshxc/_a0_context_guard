"""
hist_add_tool_result -> _10_result_size_cap

Caps every tool result at 4000 chars BEFORE it enters history.
Self-contained, zero cross-file imports.
"""
from __future__ import annotations
import os
import json
from typing import Any
from helpers.extension import Extension

MAX_CHARS = int(os.environ.get("CTX_GUARD_MAX_RESULT_CHARS", "4000"))
ENABLED   = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")


class ResultSizeCap(Extension):

    def execute(self, data: dict[str, Any] | None = None, **kwargs):
        if not ENABLED or not self.agent or not data or not isinstance(data, dict):
            return
        try:
            max_chars = max(500, MAX_CHARS)
            result = data.get("tool_result")
            if result is None:
                return

            result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

            if len(result_str) <= max_chars:
                return

            omitted   = len(result_str) - max_chars
            truncated = _close_json(result_str[:max_chars])
            data["tool_result"] = (
                truncated
                + f"\n\n[CONTEXT-GUARD: {omitted:,} chars truncated — "
                + f"showing first {max_chars:,} chars which contain the most relevant content.]"
            )
        except Exception:
            pass


def _close_json(text: str) -> str:
    s = text.strip()
    if not s or s[0] not in ("{", "["):
        return text
    oc = s.count("{") - s.count("}")
    os_ = s.count("[") - s.count("]")
    if oc <= 0 and os_ <= 0:
        return text
    return text + "\n" + ("]" * min(os_, 5)) + ("}" * min(oc, 5))
