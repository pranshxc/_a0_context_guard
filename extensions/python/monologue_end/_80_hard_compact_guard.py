"""
monologue_end  ->  _80_hard_compact_guard

Fires after a complete agent monologue (user turn -> all tool calls -> response).
This is the last-resort safety net.  If history is STILL above the hard
compact threshold after native compression has already run during the loop,
we trigger the full LLM-based summarisation from _chat_compaction.

When _chat_compaction is not installed a lightweight local fallback runs:
collapse old Topics into a Bulk via the utility model - much cheaper than
a full compact but still effective at reclaiming thousands of tokens.

Design goals
------------
- Trigger at most ONCE per monologue (per-context flag guard).
- Never interrupt a mid-run agent (fires after the response is finalised).
- Only runs on root agent (number == 0) to avoid sub-agent duplication.
- Graceful degradation when _chat_compaction is absent.
- Fully transparent: shows a hint message in the UI sidebar.

CRITICAL: import plugin-local helpers via
    from plugins._a0_context_guard.helpers.<module> import ...
"""
from __future__ import annotations
from helpers.extension import Extension
from agent import LoopData

_FLAG = "_ctx_guard_hard_compact_done"


def _get_cfg():
    try:
        from plugins._a0_context_guard.helpers.config import get
        return get
    except Exception:
        import os
        return lambda k, d=None: os.environ.get(f"CTX_GUARD_{k.upper()}", d)


class HardCompactGuard(Extension):
    """
    Last-resort LLM-based compaction when token count exceeds
    `hard_compact_threshold` after a full monologue completes.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):  # type: ignore[override]
        if not self.agent:
            return

        # Only root agent - sub-agents share the same context so we must not
        # double-compact from child agents.
        if self.agent.number != 0:
            return

        try:
            cfg = _get_cfg()
            if not cfg("enabled", True):
                return

            threshold = int(cfg("hard_compact_threshold", 75000))
            if threshold <= 0:
                return  # feature disabled

            context = self.agent.context

            # Per-monologue guard: fire at most once
            if getattr(context, _FLAG, False):
                return

            from plugins._a0_context_guard.helpers.token_utils import history_token_count
            current_tokens = history_token_count(self.agent)

            if current_tokens <= threshold:
                return  # still fine

            # Lock for this monologue
            setattr(context, _FLAG, True)

            log_item = context.log.log(
                type="hint",
                heading="\U0001f5dc Context Guard \u2014 Full Compact",
                content=(
                    f"History reached {current_tokens:,} tokens "
                    f"(threshold: {threshold:,}). "
                    "Running full LLM summarisation to reclaim context\u2026"
                ),
            )

            use_utility = bool(cfg("hard_compact_use_utility", True))

            # --- attempt _chat_compaction plugin -------------------------------
            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(context, use_chat_model=not use_utility)
                from plugins._a0_context_guard.helpers.token_utils import history_token_count as htc
                after = htc(self.agent)
                log_item.update(
                    content=(
                        f"Full compact complete: {current_tokens:,} \u2192 {after:,} tokens. "
                        "Conversation context preserved in summary."
                    )
                )
                return
            except ImportError:
                pass  # _chat_compaction not installed - fall through
            except Exception as compact_err:
                log_item.update(
                    content=(
                        f"_chat_compaction failed ({compact_err}). "
                        "Trying local fallback\u2026"
                    )
                )

            # --- local fallback: summarise old topics into bulks ---------------
            await _local_fallback_compact(self.agent, log_item)

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
            # Always reset the per-monologue lock
            try:
                setattr(self.agent.context, _FLAG, False)
            except Exception:
                pass


async def _local_fallback_compact(agent, log_item) -> None:
    """
    Lightweight fallback when _chat_compaction is unavailable.
    Summarises each old Topic individually via the utility model (same model
    A0 uses internally for topic summaries), then consolidates them into a
    single Bulk so future loops only pay summary-sized tokens.
    """
    history    = agent.history
    old_topics = list(history.topics)  # snapshot

    if not old_topics:
        log_item.update(content="No old topics to compact in local fallback.")
        return

    summarised = 0
    for topic in old_topics:
        if not topic.summary:
            try:
                await topic.summarize()
                summarised += 1
            except Exception:
                pass  # leave as-is if summarisation fails for this topic

    if summarised > 0:
        from helpers.history import Bulk
        bulk = Bulk(history=history)
        bulk.records = old_topics  # type: ignore[assignment]
        history.bulks.append(bulk)
        history.topics = []

        try:
            from helpers import persist_chat
            persist_chat.save_tmp_chat(agent.context)
        except Exception:
            pass

        from plugins._a0_context_guard.helpers.token_utils import history_token_count
        after = history_token_count(agent)
        log_item.update(
            content=(
                f"Local fallback complete: summarised {summarised} topic(s). "
                f"Tokens after: {after:,}."
            )
        )
    else:
        log_item.update(content="Local fallback: all topics already summarised.")
