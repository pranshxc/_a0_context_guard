"""
message_loop_start  ->  _20_native_compress_check

Fires at the START of every message-loop iteration, before any LLM call.
Checks whether history tokens have exceeded the configured native-compress
ratio of the model's ctx_history budget and, if so, drives A0's built-in
History.compress() pipeline.

A0's compress() is:
  - 100% local for shallow passes (just restructures Topic/Bulk records)
  - Calls utility model only when Topics need summarising (same cost A0 pays anyway)
  - Iterative: runs until under budget or no more compression is possible
  - Lossless for the current topic (recent messages always preserved verbatim)

Design goals
------------
- Zero overhead on normal loops (fast-path exits in <1ms when under budget).
- No agent interference: agent never knows this happened.
- Idempotent: safe to run every single loop iteration.
- Full fault-tolerance: exceptions logged as hints, never crash the loop.

CRITICAL: import plugin-local helpers via
    from plugins._a0_context_guard.helpers.<module> import ...
not `from helpers.<module>` which resolves to A0's own helpers/ package.
"""
from __future__ import annotations
from helpers.extension import Extension
from agent import LoopData


def _get_cfg():
    try:
        from plugins._a0_context_guard.helpers.config import get
        return get
    except Exception:
        import os
        return lambda k, d=None: os.environ.get(f"CTX_GUARD_{k.upper()}", d)


class NativeCompressCheck(Extension):
    """
    Triggers A0's native history compression when history exceeds
    `native_compress_ratio` x ctx_history budget tokens.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):  # type: ignore[override]
        if not self.agent:
            return

        try:
            cfg = _get_cfg()
            if not cfg("enabled", True):
                return

            agent = self.agent

            # --- fast path: token count check ----------------------------------
            from plugins._a0_context_guard.helpers.token_utils import (
                history_token_count,
                ctx_budget_tokens,
            )
            current_tokens = history_token_count(agent)
            budget         = ctx_budget_tokens(agent)
            ratio          = float(cfg("native_compress_ratio", 0.85))
            trigger_at     = int(budget * ratio)

            if current_tokens <= trigger_at:
                return  # well within budget - skip entirely

            # --- trigger A0 native compression ---------------------------------
            # history.compress() is async; safe to await inside an extension.
            compressed = await agent.history.compress()

            if compressed:
                after = history_token_count(agent)
                saved = current_tokens - after
                agent.context.log.log(
                    type="hint",
                    heading="\U0001f5dc Context Guard \u2014 Native Compress",
                    content=(
                        f"History compressed: {current_tokens:,} \u2192 {after:,} tokens "
                        f"(saved {saved:,}). Recent context fully preserved."
                    ),
                )

        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="\u26a0 Context Guard \u2014 Compress Error",
                    content=f"Native compression skipped: {exc}",
                )
            except Exception:
                pass
