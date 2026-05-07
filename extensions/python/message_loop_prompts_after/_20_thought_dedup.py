"""
message_loop_prompts_after -> _20_thought_dedup

Fix 3B: Thought Deduplication

Fires before every LLM call (AFTER prompt is assembled, BEFORE it's sent).
Runs at priority _20, before the token budget enforcer (_30).

Problem: The agent's internal thoughts block was repeating the same
6-line Key Findings summary verbatim across 15+ consecutive turns,
burning tokens on content the LLM already knows.

Fix: Compare the last assistant message's thought content against the
previous turn's thought fingerprint (stored in context.data).
If similarity >= DEDUP_THRESHOLD (default 80%), inject a system message
that replaces the repeated block with a single delta-only line.

This does NOT modify history — it injects a corrective system prompt
message into the current prompt_msgs list only, so the next LLM call
gets a suppression hint without corrupting stored history.

Config env vars:
  CTX_GUARD_THOUGHT_DEDUP_ENABLED   default true
  CTX_GUARD_THOUGHT_DEDUP_THRESHOLD default 0.80  (0.0-1.0 similarity)
  CTX_GUARD_ENABLED                 default true

Self-contained, zero cross-file imports beyond helpers.extension.
"""
from __future__ import annotations
import os
import re
import hashlib
from helpers.extension import Extension
from agent import LoopData

ENABLED   = os.environ.get("CTX_GUARD_ENABLED", "true").lower() not in ("0", "false", "no")
DEDUP_ON  = os.environ.get("CTX_GUARD_THOUGHT_DEDUP_ENABLED", "true").lower() not in ("0", "false", "no")
THRESHOLD = float(os.environ.get("CTX_GUARD_THOUGHT_DEDUP_THRESHOLD", "0.80"))

# Context data keys
_KEY_PREV_HASH  = "_ctxguard_prev_thought_hash"
_KEY_PREV_WORDS = "_ctxguard_prev_thought_words"
_KEY_DEDUP_COUNT = "_ctxguard_dedup_count"


class ThoughtDedup(Extension):

    async def execute(
        self,
        loop_data: LoopData = LoopData(),
        prompt_msgs=None,
        **kwargs,
    ):
        if not ENABLED or not DEDUP_ON or not self.agent:
            return
        if self.agent.number != 0:
            return
        if prompt_msgs is None:
            return

        try:
            agent   = self.agent
            context = agent.context

            # Extract thought text from last assistant message in history
            thought_text = _extract_last_thought(agent)
            if not thought_text or len(thought_text) < 50:
                return  # too short to bother fingerprinting

            current_hash  = _fingerprint(thought_text)
            current_words = _word_set(thought_text)

            prev_hash  = context.data.get(_KEY_PREV_HASH, "")
            prev_words = context.data.get(_KEY_PREV_WORDS, set())

            # Update stored fingerprint regardless of whether we suppress
            context.data[_KEY_PREV_HASH]  = current_hash
            context.data[_KEY_PREV_WORDS] = current_words

            if not prev_hash:
                return  # no previous turn to compare

            # Exact hash match — 100% duplicate
            if current_hash == prev_hash:
                similarity = 1.0
            elif prev_words:
                # Jaccard similarity on word sets
                intersection = len(current_words & prev_words)
                union        = len(current_words | prev_words)
                similarity   = intersection / union if union else 0.0
            else:
                return

            if similarity < THRESHOLD:
                return  # thoughts are sufficiently different, no suppression

            # Inject suppression hint into current prompt_msgs
            dedup_count = context.data.get(_KEY_DEDUP_COUNT, 0) + 1
            context.data[_KEY_DEDUP_COUNT] = dedup_count

            delta = _extract_delta(thought_text, context.data.get("_ctxguard_prev_thought_full", ""))

            suppression_msg = (
                "[CONTEXT-GUARD: Thought Deduplication]\n"
                f"Your previous thought block is {similarity:.0%} identical to the prior turn "
                f"(dedup #{dedup_count}). DO NOT repeat the same Key Findings or status summary.\n"
            )
            if delta:
                suppression_msg += f"New delta to note: {delta}\n"
            suppression_msg += (
                "In your thoughts this turn, write ONLY what is NEW or DIFFERENT. "
                "Reference prior findings as 'findings unchanged' and proceed directly "
                "to the current action."
            )

            prompt_msgs.append({"role": "system", "content": suppression_msg})

            context.log.log(
                type="hint",
                heading="🧠 Context Guard — Thought Dedup",
                content=(
                    f"Thought similarity {similarity:.0%} ≥ {THRESHOLD:.0%} threshold. "
                    f"Suppression hint injected (#{dedup_count}). "
                    f"Saved ~{len(thought_text) // 4} tokens."
                ),
            )

        except Exception:
            pass  # never crash the agent

        # Always store full thought for next-turn delta extraction
        try:
            self.agent.context.data["_ctxguard_prev_thought_full"] = thought_text
        except Exception:
            pass


# ── helpers ──────────────────────────────────────────────────────────────

def _extract_last_thought(agent) -> str:
    """Pull thought content from the most recent assistant message."""
    try:
        msgs = agent.history.output()
        for msg in reversed(msgs):
            content = ""
            if hasattr(msg, "content"):
                content = str(msg.content)
            elif isinstance(msg, dict):
                content = str(msg.get("content", ""))
            else:
                content = str(msg)

            if not content:
                continue

            # Extract <thoughts>...</thoughts> block if present
            m = re.search(r"<thoughts>(.*?)</thoughts>", content, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

            # Fallback: look for JSON thoughts field
            m = re.search(r'"thoughts"\s*:\s*"(.*?)"(?=,|\})', content, re.DOTALL)
            if m:
                return m.group(1).strip()

            # Fallback: look for markdown ## Thoughts or # Thoughts section
            m = re.search(r"#+\s*[Tt]houghts?\s*\n(.*?)(?=\n#+|$)", content, re.DOTALL)
            if m:
                return m.group(1).strip()

    except Exception:
        pass
    return ""


def _fingerprint(text: str) -> str:
    """MD5 hash of normalised text (lowercase, whitespace collapsed)."""
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()


def _word_set(text: str) -> set:
    """Unique meaningful words for Jaccard similarity."""
    words = re.findall(r"[a-z]{4,}", text.lower())  # 4+ char words only
    return set(words)


def _extract_delta(current: str, previous: str) -> str:
    """Return sentences present in current but not in previous (new content)."""
    if not previous:
        return ""
    try:
        prev_sentences = set(re.split(r"[.!?\n]", previous.lower()))
        curr_sentences = re.split(r"[.!?\n]", current)
        new_sentences  = [
            s.strip() for s in curr_sentences
            if s.strip() and s.strip().lower() not in prev_sentences and len(s.strip()) > 20
        ]
        return " | ".join(new_sentences[:3])  # max 3 new sentences
    except Exception:
        return ""
