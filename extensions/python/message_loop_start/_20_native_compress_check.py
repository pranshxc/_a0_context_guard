"""
message_loop_start -> _20_native_compress_check

Fires at the START of every loop iteration before prompt assembly.
Native compress only — never calls run_compaction().
Trigger: 50k tokens. Compress to: 40k tokens.

NOTE: Uses output_text+approximate_tokens for ACTUAL usage count.
      Do NOT use history.get_tokens() — it returns allocated budget, not usage.
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


def _count_tokens(agent) -> int:
    try:
        from helpers import tokens
        from helpers.history import output_text
        text = output_text(agent.history.output())
        if text:
            return tokens.approximate_tokens(text)
    except Exception:
        pass
    try:
        return max(1, len(" ".join(str(m) for m in agent.history.output())) // 4)
    except Exception:
        return 0


async def _compress_to(agent, target: int, max_passes: int) -> int:
    history  = agent.history
    original = history._get_ctx_size_for_history
    def _patched(): return target
    current  = _count_tokens(agent)
    passes   = 0
    try:
        history._get_ctx_size_for_history = _patched
        while current > target and passes < max_passes:
            ok = await history.compress()
            passes += 1
            new = _count_tokens(agent)
            if new >= current or not ok:
                break
            current = new
    finally:
        history._get_ctx_size_for_history = original
    return current


class NativeCompressCheck(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not ENABLED or not self.agent or self.agent.number != 0:
            return
        try:
            agent   = self.agent
            current = _count_tokens(agent)
            if current <= TRIGGER:
                return
            log_item = agent.context.log.log(
                type="hint",
                heading="🗃 Context Guard — Compressing",
                content=f"History {current:,} > {TRIGGER:,}. Compressing to ≤{TARGET_TOKENS:,}…",
            )
            after = await _compress_to(agent, TARGET_TOKENS, MAX_PASSES)
            log_item.update(
                content=f"✅ {current:,} → {after:,} tokens (saved {current - after:,})."
            )
        except Exception:
            pass
