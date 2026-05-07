"""
Lightweight token utilities for _a0_context_guard.

CRITICAL: agent.history.get_tokens() returns the ALLOCATED BUDGET
(e.g. 128000 * 0.70 = 89600), NOT the actual current token usage.
Always count from output_text(history.output()) instead.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


def history_token_count(agent: "Agent") -> int:
    """Return actual token count of current history content."""
    # Method 1: count from serialised output text (most accurate)
    try:
        from helpers import tokens
        from helpers.history import output_text
        text = output_text(agent.history.output())
        if text:
            return tokens.approximate_tokens(text)
    except Exception:
        pass
    # Method 2: char-based fallback (~4 chars per token)
    try:
        text = " ".join(str(m) for m in agent.history.output())
        return max(1, len(text) // 4)
    except Exception:
        pass
    return 0


def ctx_budget_tokens(agent: "Agent") -> int:
    """Return the token budget A0 has allocated for history (ctx_length * ctx_history)."""
    try:
        from plugins._model_config.helpers.model_config import get_chat_model_config
        cfg = get_chat_model_config(agent)
        ctx_length = int(cfg.get("ctx_length", 128000))
        ctx_history = float(cfg.get("ctx_history", 0.70))
        return int(ctx_length * ctx_history)
    except Exception:
        return 89600
