"""
message_loop_prompts_after -> _30_token_budget_enforcer

Fires before EVERY LLM call (mid-turn, during active agent work).

CRITICAL RULE: NEVER call run_compaction() (full LLM "Context compacted")
from this hook. That wipes history + log mid-task, producing the
"Context compacted" message as the final response.

This hook ONLY uses A0's native history.compress() which is a lightweight
incremental summariser that does NOT wipe the log or show a final response.

Full LLM compaction (run_compaction) is only permitted in monologue_end
(_80_hard_compact_guard.py) which fires AFTER the agent has finished and
the user is already waiting for the next message.

Config env vars:
  CTX_GUARD_TARGET_TOKENS  default 60000  compress history to this target
  CTX_GUARD_BUFFER_TOKENS  default 20000  headroom -> trigger = target+buffer = 80000
  CTX_GUARD_MIN_TOKENS     default 8000   floor for compress target
  CTX_GUARD_MAX_PASSES     default 1      max native compress passes per call
  CTX_GUARD_ENABLED        default true

Effective defaults:
  Compress to : 60,000 tokens
  Trigger at  : 80,000 tokens  (60k + 20k)
  Max passes  : 1 per call (avoids over-compressing mid-turn)
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

TARGET_TOKENS = int(os.environ.get("CTX_GUARD_TARGET_TOKENS", "60000"))
BUFFER_TOKENS = int(os.environ.get("CTX_GUARD_BUFFER_TOKENS", "20000"))
MIN_TOKENS    = int(os.environ.get("CTX_GUARD_MIN_TOKENS",    "8000"))
MAX_PASSES    = int(os.environ.get("CTX_GUARD_MAX_PASSES",    "1"))
ENABLED       = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")


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


async def _compress_to_target(agent, target: int) -> bool:
    """Monkey-patch compress target and run A0 native compress."""
    history = agent.history
    original = history._get_ctx_size_for_history
    def _patched(): return target
    try:
        history._get_ctx_size_for_history = _patched
        return await history.compress()
    finally:
        history._get_ctx_size_for_history = original


class TokenBudgetEnforcer(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent:
            return
        if self.agent.number != 0:
            return

        try:
            agent   = self.agent
            context = agent.context

            call_count = context.data.get("_ctxguard_call_count", 0) + 1
            context.data["_ctxguard_call_count"] = call_count

            hist_tokens = _tokens_from_output(loop_data.history_output)
            if hist_tokens == 0:
                hist_tokens = _history_tokens(agent)

            trigger = TARGET_TOKENS + BUFFER_TOKENS  # default 80k

            context.data["_ctxguard_last_tokens"] = hist_tokens

            if hist_tokens <= trigger:
                return  # under budget, nothing to do

            # Don't re-compress the exact same assembled prompt twice
            last_compressed = context.data.get("_ctxguard_last_compressed_call", -1)
            if call_count == last_compressed:
                return
            context.data["_ctxguard_last_compressed_call"] = call_count

            compress_target = max(MIN_TOKENS, TARGET_TOKENS)

            log_item = context.log.log(
                type="hint",
                heading="🗃 Context Guard — Compressing",
                content=(
                    f"History: {hist_tokens:,} tokens > {trigger:,} trigger. "
                    f"Native compress to ≤{compress_target:,}… (call #{call_count})"
                ),
            )

            passes       = 0
            current      = hist_tokens
            any_progress = False

            while current > trigger and passes < MAX_PASSES:
                compressed = await _compress_to_target(agent, compress_target)
                passes += 1
                loop_data.history_output = agent.history.output()
                new_tokens = _tokens_from_output(loop_data.history_output)
                if new_tokens >= current:
                    break  # no progress
                current = new_tokens
                if compressed:
                    any_progress = True
                if current <= trigger:
                    break

            if any_progress:
                saved = hist_tokens - current
                log_item.update(
                    content=(
                        f"✅ {hist_tokens:,} → {current:,} tokens "
                        f"(saved {saved:,} in {passes} pass{'es' if passes > 1 else ''}). "
                        f"Under {trigger:,} limit."
                    )
                )
            else:
                log_item.update(
                    content=(
                        f"⚠️ Native compress made no progress at {current:,} tokens. "
                        "Full compaction deferred to end of turn — continuing work."
                    )
                )

        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="⚠ Context Guard — Error",
                    content=f"Token budget enforcer: {exc}",
                )
            except Exception:
                pass
