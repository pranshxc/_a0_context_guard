"""
message_loop_start -> _20_native_compress_check

Pre-flight check at START of each loop. Fires BEFORE prompt assembly.
Native compress only — NEVER calls run_compaction().

Triggers at 50k (TARGET 40k + BUFFER 10k). Compresses back down to 40k.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

TARGET_TOKENS = int(os.environ.get("CTX_GUARD_TARGET_TOKENS", "40000"))
BUFFER_TOKENS = int(os.environ.get("CTX_GUARD_BUFFER_TOKENS", "10000"))
MAX_PASSES    = int(os.environ.get("CTX_GUARD_MAX_PASSES",    "5"))
ENABLED       = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")
TRIGGER       = TARGET_TOKENS + BUFFER_TOKENS  # 50000


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
                break
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

            if current <= TRIGGER:
                return  # under 50k, nothing to do

            log_item = context.log.log(
                type="hint",
                heading="🗃 Context Guard — Pre-compress",
                content=(
                    f"History at {current:,} tokens > {TRIGGER:,} trigger. "
                    f"Native compress to ≤{TARGET_TOKENS:,}…"
                ),
            )
            after = await _native_compress_to(agent, TARGET_TOKENS, MAX_PASSES)
            log_item.update(
                content=f"✅ Pre-compress: {current:,} → {after:,} tokens (saved {current - after:,})."
            )

        except Exception:
            pass
