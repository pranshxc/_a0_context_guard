"""
monologue_end -> _80_hard_compact_guard

Stats-only. Fires after agent fully responds.
NO compaction of any kind — never calls run_compaction(), never calls compress().
All compression happens mid-turn in _20 and _30.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

ENABLED = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")


def _history_tokens(agent) -> int:
    try:
        t = agent.history.get_tokens()
        if t and t > 0: return t
    except Exception: pass
    try:
        from helpers import tokens as tok_helper
        from helpers.history import output_text
        hist_text = output_text(agent.history.output())
        if hist_text: return tok_helper.approximate_tokens(hist_text)
    except Exception: pass
    try:
        return sum(len(str(m)) for m in agent.history.output()) // 4
    except Exception: return 0


class HardCompactGuard(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            current = _history_tokens(self.agent)
            self.agent.context.log.log(
                type="hint",
                heading="📊 Context Guard — Turn Stats",
                content=f"Turn complete. History: {current:,} tokens.",
            )
        except Exception:
            pass
