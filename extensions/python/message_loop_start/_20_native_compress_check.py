"""
message_loop_start -> _20_native_compress_check

Lightweight pre-flight check at the START of each loop iteration.
Now acts as an EARLY WARNING guard only — the main compression logic
has moved to message_loop_prompts_after/_30_token_budget_enforcer.py
which fires after history_output is assembled and has access to exact
prompt token counts.

This file now just:
 - Checks if history object tokens are critically high (> HARD_LIMIT)
 - If so, pre-emptively compresses BEFORE prompt assembly even starts
 - This prevents wasted prompt-assembly work on bloated history

Self-contained, zero cross-file imports.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

HARD_LIMIT = int(os.environ.get("CTX_GUARD_HARD_LIMIT", "120000"))  # emergency brake
ENABLED    = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0","false","no")


def _history_tokens(agent) -> int:
    try:
        t = agent.history.get_tokens()
        if t and t > 0: return t
    except Exception: pass
    try:
        from helpers import tokens
        from helpers.history import output_text
        text = output_text(agent.history.output())
        if text: return tokens.approximate_tokens(text)
    except Exception: pass
    try:
        return sum(len(str(m)) for m in agent.history.output()) // 4
    except Exception: return 0


class NativeCompressCheck(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            current = _history_tokens(self.agent)
            if current <= HARD_LIMIT:
                return  # fine — token_budget_enforcer handles normal compression

            # Emergency: history is critically large, compress NOW before prompt build
            context = self.agent.context
            context.log.log(
                type="hint",
                heading="🚨 Context Guard — Emergency Compress",
                content=(
                    f"History at {current:,} tokens hits hard limit {HARD_LIMIT:,}. "
                    "Emergency compression before prompt assembly…"
                ),
            )
            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(context, use_chat_model=False)
            except ImportError:
                await self.agent.history.compress()
            except Exception:
                await self.agent.history.compress()
        except Exception:
            pass
