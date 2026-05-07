"""
monologue_end -> _80_hard_compact_guard

Stats-only. Fires after agent fully responds.
NO compaction of any kind. Zero compress() or run_compaction() calls.

NOTE: Uses output_text+approximate_tokens for actual token count display.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

ENABLED = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")


def _count_tokens(agent) -> int:
    try:
        from helpers import tokens
        from helpers.history import output_text
        text = output_text(agent.history.output())
        if text:
            return tokens.approximate_tokens(text)
    except Exception:
        pass
    try:
        return max(1, len(" ".join(str(m) for m in agent.history.output())) // 4)
    except Exception:
        return 0


class HardCompactGuard(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            current = _count_tokens(self.agent)
            self.agent.context.log.log(
                type="hint",
                heading="📊 Context Guard — Turn Stats",
                content=f"Turn complete. History: {current:,} tokens.",
            )
        except Exception:
            pass
