"""
monologue_end  →  _80_hard_compact_guard

Fires after a complete agent monologue (user turn → all tool calls → response).
This is the last-resort safety net.  If, after native compression has already
run during the loop, the history is STILL above the hard compact threshold,
we trigger the full LLM-based summarisation from _chat_compaction.

When _chat_compaction is not installed a lightweight local fallback runs:
it collapses old Topic summaries into a single Bulk with a utility-model
summary, which is much cheaper than a full compact but still effective.

Design goals
────────────
• Trigger at most ONCE per monologue (guarded by a flag on the context).
• Never interrupt a mid-run agent (fires after the response is finalised).
• Graceful degradation when _chat_compaction is absent.
• Fully transparent — shows a hint message in the UI.
"""
from __future__ import annotations
from helpers.extension import Extension
from agent import LoopData

_FLAG = "_ctx_guard_hard_compact_done"


def _cfg():
    try:
        from helpers.config import get
        return get
    except ImportError:
        import os
        return lambda k, d=None: os.environ.get(f"CTX_GUARD_{k.upper()}", d)


class HardCompactGuard(Extension):
    """
    Last-resort LLM-based compaction when token count exceeds
    `hard_compact_threshold` after a monologue completes.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):  # type: ignore[override]
        if not self.agent:
            return

        # Only run for the root agent (number == 0) to avoid redundant compaction
        # on sub-agents that already share the same context.
        if self.agent.number != 0:
            return

        try:
            cfg = _cfg()
            if not cfg("enabled", True):
                return

            threshold = int(cfg("hard_compact_threshold", 75000))
            if threshold <= 0:
                return  # feature disabled

            # Guard: run at most once per monologue
            context = self.agent.context
            if getattr(context, _FLAG, False):
                return

            from helpers.token_utils import history_token_count  # local
            current_tokens = history_token_count(self.agent)

            if current_tokens <= threshold:
                return  # still fine

            # Mark so we don't re-enter during this monologue
            setattr(context, _FLAG, True)

            log_item = context.log.log(
                type="hint",
                heading="🗜 Context Guard — Full Compact",
                content=(
                    f"History reached {current_tokens:,} tokens (threshold: {threshold:,}). "
                    "Running full LLM summarisation to reclaim context…"
                ),
            )

            use_utility = bool(cfg("hard_compact_use_utility", True))

            # ── attempt _chat_compaction plugin ───────────────────────────────
            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(
                    context,
                    use_chat_model=not use_utility,
                )
                after = history_token_count(self.agent)
                log_item.update(
                    content=(
                        f"Full compact complete: {current_tokens:,} → {after:,} tokens. "
                        "Conversation context preserved in summary."
                    )
                )
                return
            except ImportError:
                pass  # _chat_compaction not installed — fall through to local fallback
            except Exception as compact_err:
                log_item.update(
                    content=f"_chat_compaction failed ({compact_err}). Trying local fallback…"
                )

            # ── local fallback: collapse old topics into summarised bulks ──────
            await _local_fallback_compact(self.agent, log_item)

        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Hard Compact Error",
                    content=f"Hard compact skipped: {exc}",
                )
            except Exception:
                pass
        finally:
            # Always reset the per-monologue flag
            try:
                setattr(self.agent.context, _FLAG, False)
            except Exception:
                pass


async def _local_fallback_compact(agent, log_item) -> None:
    """
    Lightweight fallback when _chat_compaction is unavailable.
    Summarises each old Topic individually using the utility model (same model
    A0 uses internally), then rolls them into a Bulk so future loops only pay
    for the summary tokens.
    """
    history = agent.history
    old_topics = list(history.topics)  # snapshot
    if not old_topics:
        log_item.update(content="No old topics to compact in local fallback.")
        return

    summarised = 0
    for topic in old_topics:
        if not topic.summary:  # skip already-summarised topics
            try:
                await topic.summarize()
                summarised += 1
            except Exception:
                pass  # leave as-is if summarisation fails

    # Merge summarised topics into a single Bulk
    if summarised > 0:
        from helpers.history import Bulk
        bulk = Bulk(history=history)
        bulk.records = old_topics  # type: ignore[assignment]
        history.bulks.append(bulk)
        history.topics = []

        from helpers import persist_chat
        try:
            persist_chat.save_tmp_chat(agent.context)
        except Exception:
            pass

        from helpers.token_utils import history_token_count
        after = history_token_count(agent)
        log_item.update(
            content=(
                f"Local fallback compact: summarised {summarised} topics. "
                f"Tokens after: {after:,}."
            )
        )
    else:
        log_item.update(content="Local fallback: all topics already summarised.")
