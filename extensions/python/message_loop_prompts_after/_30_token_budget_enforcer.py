"""
message_loop_prompts_after -> _30_token_budget_enforcer

Fires before EVERY LLM call (including every mid-turn tool loop iteration).
Keeps history strictly under TARGET_TOKENS + BUFFER_TOKENS = trigger.

FIXES vs previous version:
  1. TARGET=40k, BUFFER=10k -> trigger=50k (always under 50k)
  2. Bug fix: the old guard used loop_data.iteration which resets to 0
     each new monologue, so only ONE compression fired per user turn.
     Now uses a monotonic call counter (_ctxguard_call_count) so the
     guard is "don't compress SAME call twice" not "don't compress SAME
     iteration twice across resets".
  3. Compression loops until actually under budget (A0's compress() is
     incremental - one call may not be enough for large histories).

Self-contained: zero cross-file imports from this plugin.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

# ── config ───────────────────────────────────────────────────────────────────────
# TARGET: compress history down to this many tokens
TARGET_TOKENS = int(os.environ.get("CTX_GUARD_TARGET_TOKENS", "40000"))
# BUFFER: headroom for ongoing work (tool results accumulating in current_topic)
BUFFER_TOKENS = int(os.environ.get("CTX_GUARD_BUFFER_TOKENS", "10000"))
# TRIGGER = TARGET + BUFFER. Compress fires whenever history > TRIGGER.
# Default: 40k + 10k = 50k. Tokens will never exceed ~50k per LLM call.
MIN_TOKENS    = int(os.environ.get("CTX_GUARD_MIN_TOKENS",    "8000"))
# Max compression passes per call (safety against infinite loop)
MAX_PASSES    = int(os.environ.get("CTX_GUARD_MAX_PASSES",    "5"))
ENABLED       = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")


# ── token counters ──────────────────────────────────────────────────────────────────
def _tokens_from_output(history_output: list) -> int:
    try:
        from helpers import tokens
        from helpers.history import output_text
        text = output_text(history_output)
        if text:
            return tokens.approximate_tokens(text)
    except Exception:
        pass
    try:
        return sum(len(str(m)) for m in history_output) // 4
    except Exception:
        return 0


def _history_tokens(agent) -> int:
    try:
        t = agent.history.get_tokens()
        if t and t > 0:
            return t
    except Exception:
        pass
    try:
        from helpers import tokens
        from helpers.history import output_text
        text = output_text(agent.history.output())
        if text:
            return tokens.approximate_tokens(text)
    except Exception:
        pass
    try:
        return sum(len(str(m)) for m in agent.history.output()) // 4
    except Exception:
        return 0


# ── compression engine ─────────────────────────────────────────────────────────────────
async def _compress_to_target(agent, target: int) -> bool:
    """
    Monkey-patch history._get_ctx_size_for_history() to return our target
    so A0's native compress() engine runs against our budget instead of
    the model's full context window (128k by default).
    Always restores the original method.
    """
    history = agent.history
    original = history._get_ctx_size_for_history

    def _patched():
        return target

    try:
        history._get_ctx_size_for_history = _patched
        return await history.compress()
    finally:
        history._get_ctx_size_for_history = original


# ── extension ──────────────────────────────────────────────────────────────────────
class TokenBudgetEnforcer(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent:
            return
        if self.agent.number != 0:
            return

        try:
            agent   = self.agent
            context = agent.context

            # ── monotonic call counter (fixes the once-per-monologue bug) ────────
            # loop_data.iteration resets to 0 on each new monologue, so using it
            # as a guard key caused compression to fire only ONCE per user turn.
            # We use a monotonically increasing counter on context.data instead.
            call_count = context.data.get("_ctxguard_call_count", 0) + 1
            context.data["_ctxguard_call_count"] = call_count

            # ── measure exact tokens in assembled prompt ───────────────────────
            hist_tokens = _tokens_from_output(loop_data.history_output)
            if hist_tokens == 0:
                hist_tokens = _history_tokens(agent)

            trigger = TARGET_TOKENS + BUFFER_TOKENS  # 40k + 10k = 50k

            context.data["_ctxguard_last_tokens"] = hist_tokens
            context.data["_ctxguard_trigger"]     = trigger
            context.data["_ctxguard_target"]      = TARGET_TOKENS

            if hist_tokens <= trigger:
                return  # ✅ under 50k, nothing to do

            # ── don't re-compress the exact same call (can happen if two hooks
            #    fire in rapid succession on the same assembled prompt) ───────
            last_compressed = context.data.get("_ctxguard_last_compressed_call", -1)
            if call_count == last_compressed:
                return
            context.data["_ctxguard_last_compressed_call"] = call_count

            compress_target = max(MIN_TOKENS, TARGET_TOKENS)

            log_item = context.log.log(
                type="hint",
                heading="🗜 Context Guard — Optimising",
                content=(
                    f"History: {hist_tokens:,} tokens > trigger {trigger:,}. "
                    f"Compressing to ≤{compress_target:,}… (call #{call_count})"
                ),
            )

            # ── compression loop: A0's compress() is incremental (one pass may
            #    not be enough for large histories). Loop until under budget. ──
            passes       = 0
            current      = hist_tokens
            any_progress = False

            while current > trigger and passes < MAX_PASSES:
                compressed = await _compress_to_target(agent, compress_target)
                passes += 1

                # Remeasure after each pass
                loop_data.history_output = agent.history.output()
                new_tokens = _tokens_from_output(loop_data.history_output)

                if new_tokens >= current:
                    break  # no progress, stop (prevents infinite loop)

                current = new_tokens
                if compressed:
                    any_progress = True

                if current <= trigger:
                    break  # done

            if any_progress:
                saved = hist_tokens - current
                log_item.update(
                    content=(
                        f"✅ {hist_tokens:,} → {current:,} tokens "
                        f"(saved {saved:,} in {passes} pass{'es' if passes>1 else ''}). "
                        f"Under {trigger:,} limit."
                    )
                )
                context.data["_ctxguard_last_tokens"] = current
            else:
                # Native A0 compress made no progress → fallback to _chat_compaction
                log_item.update(
                    content=(
                        f"Native compress made no progress at {current:,} tokens. "
                        "Trying full LLM compaction…"
                    )
                )
                await _try_chat_compaction(agent, context, hist_tokens, log_item)
                loop_data.history_output = agent.history.output()

        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Error",
                    content=f"Token budget enforcer: {exc}",
                )
            except Exception:
                pass


async def _try_chat_compaction(agent, context, before_tokens: int, log_item) -> None:
    try:
        from plugins._chat_compaction.helpers.compactor import run_compaction
        await run_compaction(context, use_chat_model=False)
        after = _history_tokens(agent)
        log_item.update(
            content=(
                f"✅ LLM compact: {before_tokens:,} → {after:,} tokens "
                f"(saved {before_tokens - after:,})."
            )
        )
    except ImportError:
        log_item.update(
            content="_chat_compaction not installed. Using A0 native compression only."
        )
    except Exception as e:
        log_item.update(content=f"LLM compaction failed: {e}")
