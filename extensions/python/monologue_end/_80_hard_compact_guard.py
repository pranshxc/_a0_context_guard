"""
monologue_end -> _80_hard_compact_guard

Post-monologue cleanup. Fires after the agent has fully responded.
Only triggers compaction when history is genuinely over threshold (>50k tokens).
Does NOT compact on every turn. Stats log is still shown every turn.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

# Compaction only fires when history exceeds this. Default 50k.
# Override with: CTX_GUARD_COMPACT_THRESHOLD=60000
COMPACT_THRESHOLD = int(os.environ.get("CTX_GUARD_COMPACT_THRESHOLD",
                        os.environ.get("CTX_GUARD_POST_TURN_TARGET", "50000")))
ENABLED          = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0","false","no")


def _history_tokens(agent) -> int:
    try:
        t = agent.history.get_tokens()
        if t and t > 0: return t
    except Exception: pass
    try:
        from helpers import tokens as tok_helper
        hist_text = agent.history.output_text(human_label="user", ai_label="assistant")
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

            # Always log stats so the user can monitor
            context.log.log(
                type="hint",
                heading="📊 Context Guard — Turn Stats",
                content=(
                    f"Turn complete. History: {current:,} tokens. "
                    f"Compact threshold: {COMPACT_THRESHOLD:,}."
                ),
            )

            # Only compact when genuinely over threshold
            if current <= COMPACT_THRESHOLD:
                return

            log_item = context.log.log(
                type="hint",
                heading="🗜 Context Guard — Post-Turn Compact",
                content=(
                    f"History {current:,} tokens exceeds threshold {COMPACT_THRESHOLD:,}. "
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
                # Fallback to A0 native compress
                def _patched(): return COMPACT_THRESHOLD
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
