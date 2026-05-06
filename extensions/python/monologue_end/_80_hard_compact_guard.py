"""
monologue_end -> _80_hard_compact_guard

Fires after every complete monologue.
Hard ceiling: if still over threshold after loop-level compress, force compact.
Self-contained, zero cross-file imports.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

HARD_THRESHOLD = int(os.environ.get("CTX_GUARD_HARD_THRESHOLD", "100000"))
ENABLED        = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")
_FLAG          = "_ctxguard_hard_done"


def _count_tokens(agent) -> int:
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
    except Exception:
        return 0


class HardCompactGuard(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            context = self.agent.context
            if getattr(context, _FLAG, False):
                return

            current = _count_tokens(self.agent)
            if current <= HARD_THRESHOLD:
                return

            setattr(context, _FLAG, True)

            log_item = context.log.log(
                type="hint",
                heading="🗜 Context Guard — Hard Ceiling Hit",
                content=(
                    f"Tokens: {current:,} > hard limit {HARD_THRESHOLD:,}. "
                    "Forcing full compaction…"
                ),
            )

            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(context, use_chat_model=False)
                after = _count_tokens(self.agent)
                log_item.update(
                    content=f"Hard compact done: {current:,} → {after:,} tokens."
                )
            except ImportError:
                log_item.update(content="_chat_compaction not installed. Install it for full compact support.")
            except Exception as e:
                log_item.update(content=f"Hard compact failed: {e}")

        except Exception:
            pass
        finally:
            try:
                setattr(self.agent.context, _FLAG, False)
            except Exception:
                pass
