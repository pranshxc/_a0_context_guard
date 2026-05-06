"""
message_loop_start -> _20_native_compress_check

Fires at the START of every loop iteration.
Counts tokens accurately, triggers _chat_compaction directly when over threshold.
Self-contained: zero cross-file imports.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_TRIGGER_TOKENS = int(os.environ.get("CTX_GUARD_TRIGGER_TOKENS", "80000"))
DEFAULT_ENABLED        = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")

# ── token counter ─────────────────────────────────────────────────────────────
def _count_tokens(agent) -> int:
    """Try every available method to count history tokens."""
    # Method 1: A0 native
    try:
        t = agent.history.get_tokens()
        if t and t > 0:
            return t
    except Exception:
        pass
    # Method 2: approximate from text
    try:
        from helpers import tokens
        from helpers.history import output_text
        text = output_text(agent.history.output())
        if text:
            return tokens.approximate_tokens(text)
    except Exception:
        pass
    # Method 3: rough char / 4 estimate
    try:
        total = sum(len(str(m)) for m in agent.history.output())
        return total // 4
    except Exception:
        return 0

# ── extension ─────────────────────────────────────────────────────────────────
class NativeCompressCheck(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not self.agent or not DEFAULT_ENABLED:
            return
        # Only root agent
        if self.agent.number != 0:
            return

        try:
            current = _count_tokens(self.agent)
            trigger = DEFAULT_TRIGGER_TOKENS

            if current <= trigger:
                return  # ✅ under budget, nothing to do

            context = self.agent.context

            # Guard: don't compact twice in the same loop
            if getattr(context, "_ctxguard_compacting", False):
                return
            setattr(context, "_ctxguard_compacting", True)

            context.log.log(
                type="hint",
                heading="🗜 Context Guard — Compacting",
                content=(
                    f"History at {current:,} tokens (limit {trigger:,}). "
                    "Compacting now to free context window…"
                ),
            )

            # ── Strategy 1: _chat_compaction plugin ───────────────────────────
            try:
                from plugins._chat_compaction.helpers.compactor import run_compaction
                await run_compaction(context, use_chat_model=False)  # use utility model = faster
                after = _count_tokens(self.agent)
                context.log.log(
                    type="hint",
                    heading="✅ Context Guard — Done",
                    content=f"Compacted: {current:,} → {after:,} tokens saved {current - after:,}.",
                )
                return
            except ImportError:
                pass  # _chat_compaction not installed
            except Exception as e:
                context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Compaction Error",
                    content=f"_chat_compaction failed: {e}. Trying native compress…",
                )

            # ── Strategy 2: A0 native history.compress() ─────────────────────
            try:
                compressed = await self.agent.history.compress()
                after = _count_tokens(self.agent)
                context.log.log(
                    type="hint",
                    heading="✅ Context Guard — Native Compress Done",
                    content=f"Native compress: {current:,} → {after:,} tokens.",
                )
                return
            except Exception as e2:
                context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Native Compress Failed",
                    content=f"{e2}",
                )

            # ── Strategy 3: summarise old topics locally ──────────────────────
            await _topic_fallback(self.agent, current, context)

        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Error",
                    content=f"Compression error: {exc}",
                )
            except Exception:
                pass
        finally:
            try:
                setattr(self.agent.context, "_ctxguard_compacting", False)
            except Exception:
                pass


async def _topic_fallback(agent, current_tokens: int, context) -> None:
    """Last resort: summarise oldest Topics via utility model."""
    try:
        history    = agent.history
        old_topics = list(getattr(history, "topics", []))
        if not old_topics:
            context.log.log(
                type="hint",
                heading="⚠ Context Guard — No Topics",
                content="No topics found for fallback summarisation.",
            )
            return

        done = 0
        for topic in old_topics:
            if not getattr(topic, "summary", None):
                try:
                    await topic.summarize()
                    done += 1
                except Exception:
                    pass

        if done > 0:
            try:
                from helpers.history import Bulk
                bulk = Bulk(history=history)
                bulk.records = old_topics
                history.bulks.append(bulk)
                history.topics = []
                from helpers import persist_chat
                persist_chat.save_tmp_chat(context)
            except Exception:
                pass
            after = _count_tokens(agent)
            context.log.log(
                type="hint",
                heading="✅ Context Guard — Topic Fallback Done",
                content=f"Summarised {done} topic(s): {current_tokens:,} → {after:,} tokens.",
            )
        else:
            context.log.log(
                type="hint",
                heading="⚠ Context Guard — Nothing to Compact",
                content="All topics already summarised. History may need manual reset.",
            )
    except Exception as e:
        context.log.log(
            type="hint",
            heading="⚠ Context Guard — Fallback Error",
            content=f"Topic fallback failed: {e}",
        )
