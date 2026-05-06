"""
monologue_end -> _80_hard_compact_guard

Post-monologue cleanup. Fires after the agent has fully responded and
all tool calls are complete for that user turn.

Role: persist stats and handle any edge case where token_budget_enforcer
couldn't fire (e.g. very first huge message, or compress returned False
repeatedly). Acts as the final safety net.

Self-contained, zero cross-file imports.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

POST_TURN_TARGET = int(os.environ.get("CTX_GUARD_POST_TURN_TARGET", "40000"))  # aim lower after turn ends
ENABLED          = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0","false","no")
_FLAG            = "_ctxguard_post_done"


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


class HardCompactGuard(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            context = self.agent.context
            if getattr(context, _FLAG, False):
                return
            setattr(context, _FLAG, True)

            current = _history_tokens(self.agent)
            # Log stats for every completed turn (useful for monitoring)
            last_tokens = context.data.get("_ctxguard_last_tokens", current)
            context.log.log(
                type="hint",
                heading="📊 Context Guard — Turn Stats",
                content=(
                    f"Turn complete. History: {current:,} tokens. "
                    f"Post-turn target: {POST_TURN_TARGET:,}."
                ),
            )

            if current <= POST_TURN_TARGET:
                return  # already healthy

            # Post-turn: use full LLM compaction for best quality
            # (we have time now, no active agent loop to interrupt)
            log_item = context.log.log(
                type="hint",
                heading="🗜 Context Guard — Post-Turn Compact",
                content=(
                    f"Post-turn cleanup: {current:,} tokens > target {POST_TURN_TARGET:,}. "
                    "Running full LLM summarisation now (no active work to interrupt)…"
                ),
            )
            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(context, use_chat_model=True)  # use chat model for best summary quality
                after = _history_tokens(self.agent)
                log_item.update(
                    content=(
                        f"✅ Post-turn compact: {current:,} → {after:,} tokens "
                        f"(saved {current-after:,}). "
                        f"Ready for next user message at {after:,} tokens."
                    )
                )
            except ImportError:
                # _chat_compaction not installed: use A0 native
                def _patched(): return POST_TURN_TARGET
                history = self.agent.history
                orig = history._get_ctx_size_for_history
                try:
                    history._get_ctx_size_for_history = _patched
                    await history.compress()
                finally:
                    history._get_ctx_size_for_history = orig
                after = _history_tokens(self.agent)
                log_item.update(
                    content=f"Post-turn native compress: {current:,} → {after:,} tokens."
                )
            except Exception as e:
                log_item.update(content=f"Post-turn compact failed: {e}")

        except Exception:
            pass
        finally:
            try:
                setattr(self.agent.context, _FLAG, False)
            except Exception:
                pass
