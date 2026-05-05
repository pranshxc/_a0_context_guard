"""
message_loop_start -> _20_native_compress_check

Fires at the START of every message-loop iteration before the LLM call.
All logic is self-contained in this single file - no imports from plugin helpers.
This is intentional: A0 extension files are loaded individually by the extension
runner and cannot reliably import from sibling plugin directories.
"""
from __future__ import annotations
import os
from helpers.extension import Extension
from agent import LoopData

# ── inline config ─────────────────────────────────────────────────────────────
def _cfg(key: str, default):
    """Read config: env var CTX_GUARD_<KEY> overrides the default."""
    val = os.environ.get(f"CTX_GUARD_{key.upper()}")
    if val is None:
        return default
    try:
        if isinstance(default, bool):  return val.lower() in ("1", "true", "yes")
        if isinstance(default, int):   return int(val)
        if isinstance(default, float): return float(val)
    except (ValueError, TypeError):
        pass
    return val

# ── inline token utils ────────────────────────────────────────────────────────
def _history_tokens(agent) -> int:
    try:
        return agent.history.get_tokens()
    except Exception:
        try:
            from helpers import tokens
            from helpers.history import output_text
            return tokens.approximate_tokens(output_text(agent.history.output()))
        except Exception:
            return 0

def _ctx_budget(agent) -> int:
    try:
        from plugins._model_config.helpers.model_config import get_chat_model_config
        cfg = get_chat_model_config(agent)
        return int(int(cfg.get("ctx_length", 128000)) * float(cfg.get("ctx_history", 0.70)))
    except Exception:
        return 89600  # 128k * 0.70

# ── extension ─────────────────────────────────────────────────────────────────
class NativeCompressCheck(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not self.agent:
            return
        try:
            if not _cfg("enabled", True):
                return

            agent = self.agent
            current = _history_tokens(agent)
            budget  = _ctx_budget(agent)
            ratio   = _cfg("native_compress_ratio", 0.85)
            trigger = int(budget * float(ratio))

            if current <= trigger:
                return  # fast path - under budget

            compressed = await agent.history.compress()

            if compressed:
                after = _history_tokens(agent)
                agent.context.log.log(
                    type="hint",
                    heading="\U0001f5dc Context Guard \u2014 Native Compress",
                    content=(
                        f"History compressed: {current:,} \u2192 {after:,} tokens "
                        f"(saved {current - after:,}). Recent context fully preserved."
                    ),
                )
        except Exception as exc:
            try:
                self.agent.context.log.log(
                    type="hint",
                    heading="\u26a0 Context Guard \u2014 Compress Skipped",
                    content=f"Native compression skipped: {exc}",
                )
            except Exception:
                pass
