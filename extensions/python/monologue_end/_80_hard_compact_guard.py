"""
monologue_end -> _80_hard_compact_guard

Fires after a complete agent monologue. Last-resort safety ceiling.
Self-contained: all logic inlined, no cross-file imports.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

_FLAG = "_ctx_guard_hc_done"

# ── inline config ─────────────────────────────────────────────────────────────
def _cfg(key: str, default):
    val = os.environ.get(f"CTX_GUARD_{key.upper()}")
    if val is None:
        return default
    try:
        if isinstance(default, bool):  return val.lower() in ("1", "true", "yes")
        if isinstance(default, int):   return int(val)
        if isinstance(default, float): return float(val)
    except (ValueError, TypeError):
        pass
    return val

# ── inline token utils ────────────────────────────────────────────────────────
def _history_tokens(agent) -> int:
    try:
        return agent.history.get_tokens()
    except Exception:
        try:
            from helpers import tokens
            from helpers.history import output_text
            return tokens.approximate_tokens(output_text(agent.history.output()))
        except Exception:
            return 0

# ── extension ─────────────────────────────────────────────────────────────────
class HardCompactGuard(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not self.agent:
            return
        if self.agent.number != 0:   # root agent only
            return

        try:
            if not _cfg("enabled", True):
                return

            threshold = int(_cfg("hard_compact_threshold", 75000))
            if threshold <= 0:
                return

            context = self.agent.context
            if getattr(context, _FLAG, False):
                return  # already ran this monologue

            current = _history_tokens(self.agent)
            if current <= threshold:
                return

            setattr(context, _FLAG, True)

            log_item = context.log.log(
                type="hint",
                heading="\U0001f5dc Context Guard \u2014 Full Compact",
                content=(
                    f"History reached {current:,} tokens (threshold {threshold:,}). "
                    "Running full LLM summarisation\u2026"
                ),
            )

            use_utility = bool(_cfg("hard_compact_use_utility", True))

            # attempt _chat_compaction plugin first
            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(context, use_chat_model=not use_utility)
                after = _history_tokens(self.agent)
                log_item.update(
                    content=(
                        f"Full compact done: {current:,} \u2192 {after:,} tokens. "
                        "Context preserved in summary."
                    )
                )
                return
            except ImportError:
                pass
            except Exception as e:
                log_item.update(content=f"_chat_compaction failed ({e}). Trying fallback\u2026")

            # local fallback
            await _local_compact(self.agent, log_item)

        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="\u26a0 Context Guard \u2014 Hard Compact Error",
                    content=f"Hard compact skipped: {exc}",
                )
            except Exception:
                pass
        finally:
            try:
                setattr(self.agent.context, _FLAG, False)
            except Exception:
                pass


async def _local_compact(agent, log_item) -> None:
    """Fallback: summarise old Topics via utility model and fold into Bulks."""
    history    = agent.history
    old_topics = list(history.topics)
    if not old_topics:
        log_item.update(content="No old topics found for local fallback compact.")
        return

    done = 0
    for topic in old_topics:
        if not topic.summary:
            try:
                await topic.summarize()
                done += 1
            except Exception:
                pass

    if done > 0:
        from helpers.history import Bulk
        bulk = Bulk(history=history)
        bulk.records = old_topics
        history.bulks.append(bulk)
        history.topics = []
        try:
            from helpers import persist_chat
            persist_chat.save_tmp_chat(agent.context)
        except Exception:
            pass
        after = _history_tokens(agent)
        log_item.update(
            content=f"Local fallback: summarised {done} topic(s). Tokens now: {after:,}."
        )
    else:
        log_item.update(content="Local fallback: all topics already summarised.")
