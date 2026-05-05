"""
message_loop_start  →  _20_native_compress_check

Fires at the START of every message-loop iteration, before the LLM call is
made.  Checks whether the history token count has exceeded the configured
native-compress ratio and, if so, drives A0's built-in History.compress()
pipeline.

A0's compress() is:
  • 100% local — no extra LLM calls on shallow compression
  • lossless for recent context (only old Topics/Bulks are compressed)
  • iterative — it will compress as many passes as needed to get under budget
  • already used internally by A0; we just trigger it slightly earlier

Design goals
────────────
• No agent interference — the agent never knows this happened.
• No latency on normal loops (fast path exits immediately when under budget).
• Idempotent — safe to run every single loop iteration.
• Full fault-tolerance — any exception is caught and logged as a hint only.
"""
from __future__ import annotations
import asyncio
from helpers.extension import Extension
from agent import LoopData


def _cfg():
    try:
        from helpers.config import get
        return get
    except ImportError:
        import os
        return lambda k, d=None: os.environ.get(f"CTX_GUARD_{k.upper()}", d)


class NativeCompressCheck(Extension):
    """
    Triggers A0's native history compression pipeline when history exceeds
    `native_compress_ratio` of the model's ctx_history budget.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):  # type: ignore[override]
        if not self.agent:
            return

        try:
            cfg = _cfg()
            if not cfg("enabled", True):
                return

            agent = self.agent

            # ── fast path: check token count ──────────────────────────────────
            from helpers.token_utils import history_token_count, ctx_budget_tokens  # local
            current_tokens = history_token_count(agent)
            budget = ctx_budget_tokens(agent)
            ratio = float(cfg("native_compress_ratio", 0.85))
            trigger_at = int(budget * ratio)

            if current_tokens <= trigger_at:
                return  # well within budget — skip entirely

            # ── trigger native compression ────────────────────────────────────
            # history.compress() is async and may call utility model for
            # summarisation of old topics.  We await it directly — it is safe
            # to run inside an extension.
            compressed = await agent.history.compress()

            if compressed:
                after = history_token_count(agent)
                saved = current_tokens - after
                # Surface a silent hint in the UI (not chat) so it's visible
                agent.context.log.log(
                    type="hint",
                    heading="🗜 Context Guard — Native Compress",
                    content=(
                        f"History compressed: {current_tokens:,} → {after:,} tokens "
                        f"(saved {saved:,}).  Recent context fully preserved."
                    ),
                )

        except Exception as exc:
            # Log but NEVER crash the agent loop
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Compress Error",
                    content=f"Native compression skipped due to error: {exc}",
                )
            except Exception:
                pass
