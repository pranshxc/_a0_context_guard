# _a0_context_guard

Automatic, zero-config input token stabiliser for [Agent Zero](https://github.com/agent0ai/agent-zero).

## Problem

Agent Zero accumulates history across tool calls and web searches within a single user turn. By turn 3-4 in a complex session, the input token count per LLM call can reach 100k-130k, billing you for the full accumulated context on every single call.

```
Turn 1:  system + user = ~12k tokens
Turn 2:  + tool calls + web search results = ~50k tokens  
Turn 3:  + more tools = ~90k tokens
Turn 4:  + more = ~130k tokens  ← billing you 130k every call!
```

## How It Works

Three hooks run automatically at every stage of the agent loop:

```
User sends message
     │
     ▼
[message_loop_start]         Emergency brake only (> 120k tokens)
     │
     ▼
Agent does tool calls,
web searches, work...
     │
     ├── [hist_add_tool_result]  Cap every tool result at 4k chars BEFORE history
     │
     ▼
[message_loop_prompts_after]  ← MAIN ENGINE
     │   Fires before EVERY LLM call (including mid-turn tool loops)
     │   Measures exact tokens in assembled history_output
     │   If > TARGET(60k) + BUFFER(20k) = 80k:
     │     → Patches A0's compression target to 60k
     │     → Runs A0's native compression engine
     │     → Rebuilds history_output so LLM sees trimmed version
     │   Result: LLM call costs ~60-80k instead of 130k
     │
     ▼
[monologue_end]              Post-turn cleanup to 40k baseline
     │   Full LLM summarisation (best quality, no active work)
     │   Next user message starts at ~40k instead of 130k
     ▼
Ready for next user message at ~40k tokens
```

## Token Flow After Plugin

```
New chat:     ~12k  (no change)
During turn:  capped at ~80k max per LLM call  
End of turn:  compacted to ~40k baseline
Next turn:    starts at ~40k + new work = ~60-80k max
```

## Installation

```bash
cd /a0/agent-zero/plugins
git clone https://github.com/pranshxc/_a0_context_guard
docker restart agent-zero
```

## Configuration

All settings via environment variables (no restart needed for some):

| Env Var | Default | Description |
|---|---|---|
| `CTX_GUARD_TARGET_TOKENS` | `60000` | Compress history down to this |
| `CTX_GUARD_BUFFER_TOKENS` | `20000` | Headroom for active work |
| `CTX_GUARD_POST_TURN_TARGET` | `40000` | Baseline for next user message |
| `CTX_GUARD_MAX_RESULT_CHARS` | `4000` | Cap per tool result |
| `CTX_GUARD_HARD_LIMIT` | `120000` | Emergency brake threshold |
| `CTX_GUARD_ENABLED` | `true` | Disable entirely if needed |

## Requirements

- Agent Zero (any recent version)
- `_chat_compaction` plugin (recommended, for LLM summarisation)
- Without `_chat_compaction`: uses A0's native structural compression only
