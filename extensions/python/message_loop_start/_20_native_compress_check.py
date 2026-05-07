"""
message_loop_start -> _20_native_compress_check

Pre-flight check at the START of each loop iteration (before prompt assembly).

Reads ALL thresholds from helpers/config.py (-> default_config.yaml) so that
default_config.yaml is the single source of truth. No hardcoded numbers here.

RULE: NEVER call run_compaction() anywhere in this plugin.
Native history.compress() only — incremental, non-disruptive.

Behaviour (with default_config.yaml defaults):
  soft_limit  = target + buffer * 0.5  = 45,000   gentle pre-compress
  hard_limit  = target + buffer * 1.0  = 50,000   aggressive multi-pass

Config env vars (all via default_config.yaml / env overrides):
  CTX_GUARD_TARGET_TOKENS  default 40000
  CTX_GUARD_BUFFER_TOKENS  default 10000  -> trigger 50000
  CTX_GUARD_MAX_PASSES     default 5
  CTX_GUARD_ENABLED        default true
"""
from __future__ import annotations
import os
import sys
from helpers.extension import Extension
from agent import LoopData


def _cfg(key: str, default: int) -> int:
    """Read from helpers/config.py if available, else fall back to env/default."""
    try:
        plugin_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        sys.path.insert(0, os.path.abspath(plugin_dir))
        from plugins._a0_context_guard.helpers.config import get
        return int(get(key, default))
    except Exception:
        pass
    try:
        return int(os.environ.get(f"CTX_GUARD_{key.upper()}", str(default)))
    except Exception:
        return default


TARGET_TOKENS = _cfg("target_tokens", 40000)
BUFFER_TOKENS = _cfg("buffer_tokens", 10000)
MAX_PASSES    = _cfg("max_passes",    5)
ENABLED       = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")

TRIGGER   = TARGET_TOKENS + BUFFER_TOKENS          # 50000 default
SOFT_LIMIT = TARGET_TOKENS + BUFFER_TOKENS // 2    # 45000 default (gentle early warning)
HARD_LIMIT = TRIGGER                               # 50000 default (same as trigger)


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
    """Multi-pass native compress. NEVER calls run_compaction()."""
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

            # ── SOFT: gentle pre-compress at 45k ───────────────────────────
            if SOFT_LIMIT < current <= HARD_LIMIT:
                log_item = context.log.log(
                    type="hint",
                    heading="🟡 Context Guard — Pre-compress",
                    content=(
                        f"History at {current:,} tokens ≥ soft limit {SOFT_LIMIT:,}. "
                        f"Native compress to ≤{TARGET_TOKENS:,}…"
                    ),
                )
                after = await _native_compress_to(agent, TARGET_TOKENS, MAX_PASSES)
                log_item.update(
                    content=f"🟡 Pre-compress: {current:,} → {after:,} tokens (saved {current - after:,})."
                )
                return

            # ── HARD: aggressive native compress — NO run_compaction() ────────
            if current > HARD_LIMIT:
                log_item = context.log.log(
                    type="hint",
                    heading="🚨 Context Guard — Hard Limit",
                    content=(
                        f"History at {current:,} tokens > hard limit {HARD_LIMIT:,}. "
                        f"Aggressive native compress to ≤{TARGET_TOKENS:,}…"
                    ),
                )
                after = await _native_compress_to(agent, TARGET_TOKENS, MAX_PASSES)
                log_item.update(
                    content=f"🚨 Hard compress: {current:,} → {after:,} tokens (saved {current - after:,})."
                )

        except Exception:
            pass
