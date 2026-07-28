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

`uv` resolves dependencies from the inline script metadata into a throwaway
environment, so nothing is installed system-wide. The script talks only to
`$PROBE_BASE_URL`, and credentials are read from the environment — never
written to disk.

### TLS behind a corporate proxy

Corporate networks terminate TLS with a private root CA. Python's HTTP clients
default to the `certifi` bundle, which has no knowledge of that CA, so
verification fails even though macOS trusts the certificate everywhere else.

The probe therefore defaults to the **system trust store** via `truststore`
(macOS Keychain, Windows CertStore, Linux `ca-certificates`). If IT installed
the root CA on your machine, it works with no configuration.

| Variable | Effect |
| --- | --- |
| *(default)* | System trust store — Keychain on macOS |
| `PROBE_CA_BUNDLE=/path/corp-ca.pem` | Explicit PEM bundle; overrides the system store |
| `PROBE_TLS=certifi` | Ignore the system store, use the `certifi` bundle |
| `PROBE_TLS=insecure` | Disable verification — **diagnosis only, never a fix** |

`HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY` are honoured automatically.

If verification still fails, identify the intercepting CA:

```bash
openssl s_client -showcerts -connect your-gateway:443 </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer -subject
```

and export the system roots to a bundle:

```bash
security find-certificate -a -p \
  /Library/Keychains/System.keychain \
  /System/Library/Keychains/SystemRootCertificates.keychain \
  > corp-ca.pem

PROBE_CA_BUNDLE=$PWD/corp-ca.pem uv run probe.py
```

If the issuer is one your machine has never been given, request the root CA PEM
from IT. `PROBE_TLS=insecure` confirms TLS is the cause; it is not a remedy, and
the proxy itself must never ship with verification disabled.

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
