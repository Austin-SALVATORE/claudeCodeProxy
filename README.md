# claudeCodeProxy

Middleware to run [Claude Code](https://claude.com/claude-code) against an
internal OpenAI-compatible LLM endpoint by translating between the Anthropic
Messages API and OpenAI Chat Completions.

## Status

Pre-design. Only the capability probe exists so far — the proxy itself is not
built yet, and its architecture depends on what the probe reports.

## Probe

`probe.py` tests an OpenAI-compatible endpoint for the capabilities Claude Code
actually depends on, so the middleware is designed against measured behaviour
rather than assumptions.

```bash
export PROBE_BASE_URL="https://your-gateway/v1"   # include /v1 if your gateway uses it
export PROBE_API_KEY="..."                        # omit if unauthenticated
export PROBE_MODEL="..."                          # optional; defaults to first from /models

uv run probe.py
```

`uv` resolves `httpx` from the inline script metadata into a throwaway
environment, so nothing is installed system-wide. The script talks only to
`$PROBE_BASE_URL`, and credentials are read from the environment — never
written to disk.

### What it checks

| Check | Why it matters |
| --- | --- |
| `GET /models` | Real model IDs, needed for `sonnet`/`haiku`/`opus` → upstream mapping |
| `chat/completions` non-stream | Reachability, auth, and which `usage` keys are returned |
| `chat/completions` stream | Claude Code always streams; verifies SSE framing and `[DONE]` |
| Tool calling, non-stream | Make-or-break: confirms `arguments` is a JSON *string* |
| Tool calling, streamed | Needs `index` on tool-call deltas to reassemble fragments |
| Parallel tool calls | Claude Code fires several tools per turn |
| Tool-result round trip | Feeds `role: "tool"` back — commonly mishandled upstream |
| ~25k-token prompt | Claude Code prompts are large; finds where the endpoint gives up |

### Reading the results

- **Tool-calling checks fail** — the model has no native tool calling, so the
  middleware needs a prompted tool-calling layer that parses calls out of plain
  text. Substantially larger build.
- **Parallel tool calls fail** — the proxy must serialize Claude Code's parallel
  calls into sequential upstream turns.
- **Large-prompt check fails** — we need context-window mapping and a lower
  advertised `max_tokens`.
