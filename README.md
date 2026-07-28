# claudeCodeProxy

Middleware to run [Claude Code](https://claude.com/claude-code) against an
internal OpenAI-compatible LLM endpoint by translating between the Anthropic
Messages API and OpenAI Chat Completions.

## Status

Working, not yet run against the real gateway. 79 tests, 88% coverage, all
against a mocked upstream — the live end-to-end run is the next step.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env        # then edit it
uv run ccproxy
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY=unused     # required by the client, ignored by the proxy
claude
```

```powershell
# PowerShell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"
$env:ANTHROPIC_API_KEY  = "unused"
claude
```

## Model mapping

Which upstream model backs each Claude Code tier is entirely configuration —
no model name is hardcoded in the request path.

| Env var | Default | When Claude Code uses it |
| --- | --- | --- |
| `CCPROXY_MODEL_OPUS` | `gemma-4-31b-ITG` | Only when Opus is explicitly selected |
| `CCPROXY_MODEL_SONNET` | `qwen-latest` | **The main agent loop — most turns** |
| `CCPROXY_MODEL_HAIKU` | `qwen-3-coder-next` | Background: titles, summarisation |
| `CCPROXY_DEFAULT_TIER` | `sonnet` | Fallback when a request matches no tier |
| `CCPROXY_MODEL_MAP` | *(unset)* | JSON of exact-id overrides, applied first |

Tiers are matched by **substring**, so dated ids keep working:
`claude-sonnet-4-5-20250929` → sonnet → `qwen-latest`. Anthropic ships new
dated ids regularly, and exact-match config would silently break on each one.

For a one-off experiment without editing `.env`:

```bash
CCPROXY_MODEL_SONNET=gpt-oss-120b uv run ccproxy
```

`/health` reports the live mapping, and every request logs
`requested -> upstream (tier)`.

> **On `qwen-latest`:** it is a floating alias. Pointing the workhorse tier at
> one means the gateway can repoint it during maintenance and Claude Code's
> behaviour changes with no signal. Pin the concrete version once you know what
> it resolves to.

## Architecture

| Module | Responsibility |
| --- | --- |
| `config.py` | Env-driven settings, tier resolution, validation |
| `translate/request.py` | Anthropic → OpenAI. Tool blocks, images, `cache_control` stripping |
| `translate/response.py` | OpenAI → Anthropic, including the SSE state machine |
| `upstream.py` | httpx client, TLS trust, proxy-credential redaction |
| `errors.py` | Anthropic-shaped error envelopes |
| `app.py` | Routing, streaming, `/health` |

Stateless: every request is translated independently, no session storage.

### Known limitations

- **No prompt caching.** The gateway reports no cache token accounting, so
  `cache_control` markers are stripped and every turn re-sends full context at
  full cost. This is the main cost difference versus Anthropic-hosted Claude.
- **`count_tokens` is approximate** (~4 chars/token). The gateway exposes no
  tokeniser, so the endpoint answers rather than failing the request.
- **Context ceiling unverified.** The probe confirmed 22k tokens; Claude Code
  routinely exceeds that. The real limit per model is still unmeasured.

## Development

```bash
uv run pytest              # 79 tests, coverage report
uv run ruff check src tests
uv run ruff format src tests
```

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

If verification still fails, export the system roots to a bundle and point the
probe at it. On failure the script prints the commands for the platform it is
running on.

**Windows (PowerShell):**

```powershell
Get-ChildItem Cert:\LocalMachine\Root | ForEach-Object {
  "-----BEGIN CERTIFICATE-----"
  [Convert]::ToBase64String($_.RawData, 'InsertLineBreaks')
  "-----END CERTIFICATE-----"
} | Out-File -Encoding ascii corp-ca.pem

$env:PROBE_CA_BUNDLE = "$PWD\corp-ca.pem"; uv run .\probe.py
```

**macOS:**

```bash
openssl s_client -showcerts -connect your-gateway:443 </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer -subject

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
