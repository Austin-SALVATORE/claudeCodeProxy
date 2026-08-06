"""Context-window reconciliation.

Claude Code and an OpenAI-compatible gateway disagree about what a context
window is. Claude Code assumes Anthropic's shape -- a large *input* window with
an output budget on top -- so it will ask for 64000 output tokens alongside a
198k prompt. Most OpenAI-compatible servers count input + output against one
total, and reject the request:

    This model's maximum context length is 262144 tokens. However, you
    requested 64000 output tokens and your prompt contains at least 198145
    input tokens, for a total of at least 262145 tokens.

So max_tokens is clamped to whatever is actually left in the window. Two passes:

  1. before sending, using a character-based estimate plus a safety margin
  2. if the gateway still rejects it, using the exact input count from the
     error message, which is authoritative

The second pass exists because the estimate is approximate and a single
underestimate would otherwise fail a turn the user has already waited on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Rough bytes-per-token for JSON-encoded chat payloads. Deliberately low
# (pessimistic) so the estimate over-counts rather than under-counts: an
# over-estimate merely shortens the reply, an under-estimate fails the turn.
_CHARS_PER_TOKEN = 3.2


@dataclass(frozen=True)
class ContextOverflow:
    """Facts recovered from a gateway's context-length rejection."""

    input_tokens: int | None = None
    context_window: int | None = None


_INPUT_PATTERNS = (
    re.compile(r"parameter=input_tokens,\s*value=(\d+)"),
    re.compile(r"prompt contains at least (\d+) input tokens"),
    re.compile(r"(\d+) tokens? in the messages"),
)

_WINDOW_PATTERNS = (
    re.compile(r"maximum context length is (\d+)"),
    re.compile(r"context length of (\d+)"),
    re.compile(r"max(?:imum)?[ _]context[ _](?:length|window)\D{0,20}(\d+)"),
)

_OVERFLOW_MARKERS = (
    "maximum context length",
    "context_length_exceeded",
    "context length of",
    "reduce the length of the input prompt",
    "too many tokens",
)


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> int | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def parse_context_error(body: str) -> ContextOverflow | None:
    """Recognise a context-length rejection and pull the numbers out of it.

    Returns None for any other kind of 400 -- a malformed tool schema must not
    be mistaken for an oversized prompt and silently retried.
    """
    lowered = body.lower()
    if not any(marker in lowered for marker in _OVERFLOW_MARKERS):
        return None
    return ContextOverflow(
        input_tokens=_first_match(_INPUT_PATTERNS, body),
        context_window=_first_match(_WINDOW_PATTERNS, body),
    )


def estimate_input_tokens(payload: dict[str, Any]) -> int:
    """Approximate the prompt size of an OpenAI request.

    The gateway exposes no tokeniser, so this counts serialised characters.
    Tools are included: Claude Code ships ~15 tool schemas, which is a
    substantial and easily forgotten part of every prompt.
    """
    text = json.dumps(payload.get("messages", []), ensure_ascii=False)
    if payload.get("tools"):
        text += json.dumps(payload["tools"], ensure_ascii=False)
    return int(len(text) / _CHARS_PER_TOKEN)


class PromptTooLong(ValueError):
    """The prompt alone leaves no usable room for a reply."""

    def __init__(self, input_tokens: int, context_window: int) -> None:
        self.input_tokens = input_tokens
        self.context_window = context_window
        # Phrased to match what Anthropic's API returns, because Claude Code
        # keys its automatic compaction off this wording.
        super().__init__(f"prompt is too long: {input_tokens} tokens > {context_window} maximum")


def clamp_max_tokens(
    payload: dict[str, Any],
    *,
    context_window: int,
    margin: int,
    floor: int,
    ceiling: int | None = None,
    input_tokens: int | None = None,
) -> dict[str, Any]:
    """Return a new payload whose max_tokens fits the remaining window.

    Never mutates the payload it is given.
    """
    used = input_tokens if input_tokens is not None else estimate_input_tokens(payload)
    available = context_window - used - margin
    if available < floor:
        raise PromptTooLong(used, context_window)

    requested = payload.get("max_tokens") or available
    allowed = min(requested, available)
    if ceiling:
        allowed = min(allowed, ceiling)

    if allowed == payload.get("max_tokens"):
        return payload
    return {**payload, "max_tokens": max(allowed, floor)}
