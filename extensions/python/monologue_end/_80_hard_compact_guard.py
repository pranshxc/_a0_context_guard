"""
monologue_end -> _80_hard_compact_guard

Post-monologue stats + optional cleanup. Fires after the agent has fully responded.

Post-turn compaction is DISABLED by default.
The mid-turn token budget enforcer (_30_token_budget_enforcer.py) handles
all compression while the agent is working. Running full LLM compaction after
the final response produces a "Context compacted" message that interrupts the
chat flow and confuses the user.

Enable post-turn compaction only if you explicitly want it:
  CTX_GUARD_POST_TURN_COMPACT_ENABLED=true

Config env vars:
  CTX_GUARD_POST_TURN_COMPACT_ENABLED  default false   enable/disable post-turn compact
  CTX_GUARD_POST_TURN_TARGET           default 30000   compact to this after turn ends
  CTX_GUARD_ENABLED                    default true
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

ENABLED          = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")
POST_TURN_ON     = os.environ.get("CTX_GUARD_POST_TURN_COMPACT_ENABLED", "false").lower() in ("1", "true", "yes")
POST_TURN_TARGET = int(os.environ.get("CTX_GUARD_POST_TURN_TARGET", "30000"))


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
            context = self.agent.context
            current = _history_tokens(self.agent)

            # Always log stats so the user can monitor context health
            post_status = "enabled" if POST_TURN_ON else "disabled"
            context.log.log(
                type="hint",
                heading="📊 Context Guard — Turn Stats",
                content=(
                    f"Turn complete. History: {current:,} tokens. "
                    f"Post-turn compaction: {post_status} "
                    f"(target: {POST_TURN_TARGET:,} if enabled)."
                ),
            )

            # Post-turn compaction is OFF by default — skip unless explicitly enabled
            if not POST_TURN_ON:
                return

            if current <= POST_TURN_TARGET:
                return

            log_item = context.log.log(
                type="hint",
                heading="🗃 Context Guard — Post-Turn Compact",
                content=(
                    f"History {current:,} tokens exceeds post-turn target {POST_TURN_TARGET:,}. "
                    "Running compaction after turn end…"
                ),
            )

            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(context, use_chat_model=True)
                after = _history_tokens(self.agent)
                log_item.update(
                    content=(
                        f"✅ Post-turn compact: {current:,} → {after:,} tokens "
                        f"(saved {current - after:,})."
                    )
                )
            except ImportError:
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
