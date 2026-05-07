"""
message_loop_start -> _20_native_compress_check

Lightweight pre-flight check at the START of each loop iteration.

RULE: NEVER call run_compaction() anywhere in this plugin.
run_compaction() replaces the entire chat history with a compacted summary
and shows a 'Context compacted' message as the final response — this is
disruptive and not what we want. Native compress only.

Behaviour:
  SOFT_LIMIT (default 60k): fires early multi-pass native compress
  HARD_LIMIT (default 80k): fires aggressive multi-pass native compress
                            (same mechanism, just more passes)

The main incremental compressor is _30_token_budget_enforcer.py
(fires after prompt assembly). This file handles history that has grown
significantly between turns before prompt assembly even starts.

Config env vars:
  CTX_GUARD_HARD_LIMIT    default 80000   aggressive native-only compress
  CTX_GUARD_SOFT_LIMIT    default 60000   pre-compress trigger
  CTX_GUARD_TARGET_TOKENS default 40000   compress-down target
  CTX_GUARD_MAX_PASSES    default 5
  CTX_GUARD_ENABLED       default true
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

HARD_LIMIT    = int(os.environ.get("CTX_GUARD_HARD_LIMIT",    "80000"))
SOFT_LIMIT    = int(os.environ.get("CTX_GUARD_SOFT_LIMIT",    "60000"))
TARGET_TOKENS = int(os.environ.get("CTX_GUARD_TARGET_TOKENS", "40000"))
MAX_PASSES    = int(os.environ.get("CTX_GUARD_MAX_PASSES",    "5"))
ENABLED       = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")


def _history_tokens(agent) -> int:
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
    except Exception: return 0


async def _native_compress_to(agent, target: int, max_passes: int) -> int:
    """Multi-pass native compress to target. Returns final token count.
    NEVER calls run_compaction() — native compress only."""
    history  = agent.history
    original = history._get_ctx_size_for_history
    def _patched(): return target
    passes  = 0
    current = _history_tokens(agent)
    try:
        history._get_ctx_size_for_history = _patched
        while current > target and passes < max_passes:
            compressed = await history.compress()
            passes += 1
            new_tokens = _history_tokens(agent)
            if new_tokens >= current:
                break  # no progress, stop
            current = new_tokens
            if not compressed:
                break
    finally:
        history._get_ctx_size_for_history = original
    return current


class NativeCompressCheck(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            agent   = self.agent
            context = agent.context
            current = _history_tokens(agent)

            # ── SOFT_LIMIT: pre-compress before prompt assembly ──────────────────
            if SOFT_LIMIT < current <= HARD_LIMIT:
                log_item = context.log.log(
                    type="hint",
                    heading="🟡 Context Guard — Pre-compress",
                    content=(
                        f"History at {current:,} tokens ≥ soft limit {SOFT_LIMIT:,}. "
                        f"Native compress to ≤{TARGET_TOKENS:,} before prompt assembly…"
                    ),
                )
                after = await _native_compress_to(agent, TARGET_TOKENS, MAX_PASSES)
                log_item.update(
                    content=(
                        f"🟡 Pre-compress: {current:,} → {after:,} tokens "
                        f"(saved {current - after:,})."
                    )
                )
                return

            # ── HARD_LIMIT: aggressive native compress — NO run_compaction() ────
            if current > HARD_LIMIT:
                log_item = context.log.log(
                    type="hint",
                    heading="🚨 Context Guard — Hard Limit Compress",
                    content=(
                        f"History at {current:,} tokens hits hard limit {HARD_LIMIT:,}. "
                        f"Aggressive native compress to ≤{TARGET_TOKENS:,}…"
                    ),
                )
                # Native compress only — never run_compaction()
                after = await _native_compress_to(agent, TARGET_TOKENS, MAX_PASSES)
                log_item.update(
                    content=(
                        f"🚨 Hard compress: {current:,} → {after:,} tokens "
                        f"(saved {current - after:,})."
                    )
                )

        except Exception:
            pass
