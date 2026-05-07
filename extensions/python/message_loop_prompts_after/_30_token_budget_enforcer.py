"""
message_loop_prompts_after -> _30_token_budget_enforcer

Fires after prompt assembly, before each LLM call.
Reads ALL thresholds from helpers/config.py -> default_config.yaml.

RULE: NEVER call run_compaction(). Native history.compress() only.

Defaults (from default_config.yaml):
  TARGET_TOKENS = 40,000  compress to here
  BUFFER_TOKENS = 10,000  headroom
  TRIGGER       = 50,000  fires when history > 50k
  MAX_PASSES    = 5
"""
from __future__ import annotations
import os
import sys
from helpers.extension import Extension
from agent import LoopData


def _cfg(key: str, default: int) -> int:
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
MIN_TOKENS    = _cfg("min_tokens",    8000)
MAX_PASSES    = _cfg("max_passes",    5)
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

            trigger = TARGET_TOKENS + BUFFER_TOKENS  # 50k default

            context.data["_ctxguard_last_tokens"] = hist_tokens

            if hist_tokens <= trigger:
                return

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
                    break
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
                        "Continuing work — full compaction disabled."
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
