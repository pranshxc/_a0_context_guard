"""
Lightweight token utilities shared across all context-guard extensions.
Uses A0's own `helpers.tokens.approximate_tokens` so estimates stay consistent
with what the rest of the framework reports.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


def history_token_count(agent: "Agent") -> int:
    """Return approximate token count of the agent's full history."""
    try:
        return agent.history.get_tokens()
    except Exception:
        # fallback: serialise and count
        try:
            from helpers import tokens
            from helpers.history import output_text
            text = output_text(agent.history.output())
            return tokens.approximate_tokens(text)
        except Exception:
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
        return 89600  # 128k * 0.70 safe fallback
