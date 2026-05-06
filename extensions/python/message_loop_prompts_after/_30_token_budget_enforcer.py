"""
message_loop_prompts_after -> _30_token_budget_enforcer

This is the CORRECT hook for token control in Agent Zero.

Why this hook:
 - Fires AFTER loop_data.history_output is already populated
 - We can measure the EXACT tokens about to be sent to the LLM
 - We can rewrite loop_data.history_output before the LLM sees it
 - This means: every single tool call, every agent loop iteration gets checked

Strategy:
 1. Count tokens in the assembled prompt (history_output)
 2. If over budget -> call history.compress() with a PATCHED limit so A0's
    own compression engine runs correctly against our target, not the model's
    full ctx window
 3. After compress, rebuild history_output so the LLM gets fresh trimmed history
 4. Store rolling stats so other extensions can read them

Self-contained: zero cross-file imports from this plugin.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

# ── tuneable constants (all overridable via env vars) ─────────────────────────
# Target: keep history below this many tokens before each LLM call
TARGET_TOKENS   = int(os.environ.get("CTX_GUARD_TARGET_TOKENS",   "60000"))
# Buffer: reserve this many tokens for the current agentic work (tool results,
# new responses, etc.) so we never compress so hard we lose active context
BUFFER_TOKENS   = int(os.environ.get("CTX_GUARD_BUFFER_TOKENS",   "20000"))
# Hard floor: never compress below this (protects very recent messages)
MIN_TOKENS      = int(os.environ.get("CTX_GUARD_MIN_TOKENS",      "8000"))
# Only compress if history exceeds TARGET + BUFFER (= effective trigger)
# e.g. 60k target + 20k buffer = compress when history > 80k
ENABLED         = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0","false","no")


# ── inline token counter (3 fallback methods, never returns 0) ─────────────────
def _tokens_from_output(history_output: list) -> int:
    """Count tokens from the already-assembled history_output list."""
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
    """Count tokens directly from agent.history object."""
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


# ── compression engine ─────────────────────────────────────────────────────────
async def _compress_to_target(agent, target: int) -> bool:
    """
    Trick A0's history.compress() into targeting our budget instead of the
    model's full context window.

    history.compress() calls self._get_ctx_size_for_history() internally.
    We temporarily monkey-patch that method to return our target so A0's
    compression ratios (current_topic 50%, history_topics 30%, bulks 20%)
    work correctly against our desired ceiling.
    """
    history = agent.history
    original_method = history._get_ctx_size_for_history

    def _patched_ctx_size():
        return target

    try:
        history._get_ctx_size_for_history = _patched_ctx_size
        compressed = await history.compress()
        return compressed
    finally:
        history._get_ctx_size_for_history = original_method  # always restore


# ── extension ──────────────────────────────────────────────────────────────────
class TokenBudgetEnforcer(Extension):
    """
    Fires after every prompt assembly (= before every LLM call, including
    mid-monologue tool-result loops). Keeps history under TARGET_TOKENS
    while preserving BUFFER_TOKENS headroom for ongoing agentic work.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent:
            return
        # Only root agent (sub-agents share the same history object)
        if self.agent.number != 0:
            return

        try:
            agent   = self.agent
            context = agent.context

            # ── 1. measure actual tokens in assembled history_output ────────
            hist_tokens = _tokens_from_output(loop_data.history_output)
            if hist_tokens == 0:
                hist_tokens = _history_tokens(agent)  # fallback to history object

            trigger = TARGET_TOKENS + BUFFER_TOKENS  # e.g. 80k

            # Store rolling stats on context for external visibility
            context.data["_ctxguard_last_tokens"] = hist_tokens
            context.data["_ctxguard_trigger"]     = trigger
            context.data["_ctxguard_target"]      = TARGET_TOKENS

            if hist_tokens <= trigger:
                return  # ✅ under budget, nothing to do

            # ── 2. guard: don't compress twice in one loop iteration ─────────
            iteration = getattr(loop_data, "iteration", -1)
            last_compressed_iter = context.data.get("_ctxguard_last_iter", -999)
            if iteration == last_compressed_iter:
                return

            context.data["_ctxguard_last_iter"] = iteration

            log_item = context.log.log(
                type="hint",
                heading="🗜 Context Guard — Optimising",
                content=(
                    f"History: {hist_tokens:,} tokens > trigger {trigger:,}. "
                    f"Compressing to ~{TARGET_TOKENS:,} (buffer: {BUFFER_TOKENS:,})"
                ),
            )

            # ── 3. compress via A0's native engine with patched target ────────
            compress_target = max(MIN_TOKENS, TARGET_TOKENS)
            compressed = await _compress_to_target(agent, compress_target)

            if compressed:
                # Rebuild history_output so the LLM sees the trimmed version
                loop_data.history_output = agent.history.output()
                after = _tokens_from_output(loop_data.history_output)
                saved = hist_tokens - after
                log_item.update(
                    content=(
                        f"✅ Compressed: {hist_tokens:,} → {after:,} tokens "
                        f"(saved {saved:,}). Buffer headroom: "
                        f"{compress_target + BUFFER_TOKENS - after:,} tokens."
                    )
                )
                context.data["_ctxguard_last_tokens"] = after
            else:
                # A0 native didn't compress (e.g. already minimal) → try _chat_compaction
                log_item.update(
                    content="Native compress returned nothing to do. Trying full LLM compaction…"
                )
                await _try_chat_compaction(agent, context, hist_tokens, log_item)

                # Rebuild output either way
                loop_data.history_output = agent.history.output()

        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Error",
                    content=f"Token budget enforcer error: {exc}",
                )
            except Exception:
                pass


async def _try_chat_compaction(agent, context, before_tokens: int, log_item) -> None:
    """Fallback: call _chat_compaction plugin for full LLM summarisation."""
    try:
        from plugins._chat_compaction.helpers.compactor import run_compaction
        await run_compaction(context, use_chat_model=False)
        after = _history_tokens(agent)
        log_item.update(
            content=(
                f"✅ Full LLM compact: {before_tokens:,} → {after:,} tokens "
                f"(saved {before_tokens - after:,})."
            )
        )
    except ImportError:
        log_item.update(
            content=(
                "_chat_compaction plugin not installed. "
                "Install it for full LLM summarisation support. "
                "Native A0 compression only."
            )
        )
    except Exception as e:
        log_item.update(content=f"Full compaction failed: {e}")
