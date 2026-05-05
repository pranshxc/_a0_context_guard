# _a0_context_guard

Automatic context-window stabilisation plugin for [Agent Zero](https://github.com/agent0ai/agent-zero).

Keeps input token usage stable during long autonomous agent runs — without losing intelligence, interrupting the agent, or touching the chat experience.

---

## The Problem

During a long autonomous session (research, coding, multi-step tasks), input tokens grow steadily with every loop iteration:

- Large tool results (web search, file reads, code output) accumulate verbatim in history
- The `extras_persistent` injections (todo list, skills, datetime) are re-sent every iteration
- After ~20+ steps, context can balloon to 80k–90k tokens per call

## How It Works — Three Coordinated Hooks

```
User turn ──► [hist_add_tool_result] ──► Loop ──► [message_loop_start] ──► LLM call
                       │                                    │
               Cap large results                   Native compress if
               before they enter                   over ratio budget
               history (no LLM)                   (A0 built-in, no LLM
                                                    for shallow passes)
                                           ▼
                               [monologue_end] — last resort:
                               Full LLM summarisation if still
                               above hard threshold (via _chat_compaction
                               or local fallback)
```

### Hook 1 — `hist_add_tool_result/_10_result_size_cap.py`
**When:** Before any tool result enters history  
**What:** Truncates results exceeding `max_result_chars` (default: 4000).  
**Why:** A single large search result can be 5k–20k chars. Capping it is the single highest-ROI action — zero latency, no LLM call, runs every tool result.  
**Safety:** Appends a clear `[CONTEXT-GUARD: N chars truncated]` marker so the agent knows what happened.

### Hook 2 — `message_loop_start/_20_native_compress_check.py`
**When:** Start of every message loop iteration  
**What:** Checks if history tokens > `native_compress_ratio` × ctx_history budget. If so, calls `history.compress()` — A0's own built-in compression pipeline.  
**Why:** A0's native compress is free for most cases (just rearranges internal Topic/Bulk structure) and guarantees the agent never exceeds its model's configured context window.  
**Safety:** 100% local for shallow compression; utility model only called when Topics need summarising.

### Hook 3 — `monologue_end/_80_hard_compact_guard.py`
**When:** After each complete monologue (response delivered to user)  
**What:** If still above `hard_compact_threshold` (default: 75k tokens), triggers full LLM summarisation via `_chat_compaction` plugin (if installed) or a local bulk-summarise fallback.  
**Why:** Safety ceiling. Ensures the agent can never get stuck in a >80k token state across sessions.  
**Safety:** Fires at most once per monologue; only on root agent (number==0).

---

## Installation

```bash
cd ~/A0/agent-zero
git clone https://github.com/pranshxc/_a0_context_guard plugins/_a0_context_guard
docker restart agent-zero-skill
```

Works best alongside `_chat_compaction` (built into Agent Zero).

---

## Configuration

All settings in `default_config.yaml`.  Override any value with an env var:

| Setting | Default | Env Override | Description |
|---|---|---|---|
| `enabled` | `true` | `CTX_GUARD_ENABLED` | Master kill switch |
| `max_result_chars` | `4000` | `CTX_GUARD_MAX_RESULT_CHARS` | Max chars per tool result |
| `keep_recent_results` | `8` | `CTX_GUARD_KEEP_RECENT_RESULTS` | Recent results kept verbatim |
| `native_compress_ratio` | `0.85` | `CTX_GUARD_NATIVE_COMPRESS_RATIO` | Compress trigger (fraction of budget) |
| `hard_compact_threshold` | `75000` | `CTX_GUARD_HARD_COMPACT_THRESHOLD` | LLM compact threshold (tokens) |
| `hard_compact_use_utility` | `true` | `CTX_GUARD_HARD_COMPACT_USE_UTILITY` | Use cheaper utility model for compaction |

---

## Expected Token Profile

| Without plugin | With plugin |
|---|---|
| Grows ~300 tok/step, hits 88k+ | Stabilises at 40k–60k |
| Manual compact only | Automatic at every step |
| Agent can exceed model context | Hard ceiling enforced |
| All tool results stored raw | Large results capped at 4k chars |

---

## Compatibility

- Agent Zero ≥ 0.8 (tested on main branch)
- Works with any model (GLM-5, GPT-4o, Claude, Gemini)
- Compatible with `_chat_compaction`, `_a0_planner`, `a0-plugin-todo`
- No dependencies beyond what Agent Zero already ships
