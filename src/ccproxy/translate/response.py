"""OpenAI Chat Completions response -> Anthropic Messages response.

Two paths:

  build_anthropic_response  -- whole non-streaming body
  StreamTranslator          -- incremental SSE

The streaming path is the delicate one. Claude Code's client expects a strict
event order and exactly one open content block at a time:

    message_start
      content_block_start(0) .. content_block_delta* .. content_block_stop(0)
      content_block_start(1) .. content_block_delta* .. content_block_stop(1)
    message_delta
    message_stop

A malformed sequence does not raise on the client -- it hangs. So the
translator closes any open block before opening the next, and always emits
message_delta and message_stop, including on upstream error.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

Event = tuple[str, dict[str, Any]]

_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(finish_reason: str | None) -> str:
    return _STOP_REASONS.get(finish_reason or "", "end_turn")


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def map_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. Never let a bad one crash a turn."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_anthropic_response(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a complete non-streaming Chat Completions body."""
    choices = payload.get("choices") or [{}]
    message = choices[0].get("message") or {}

    content: list[dict[str, Any]] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "input": _parse_arguments(fn.get("arguments")),
            }
        )

    return {
        "id": payload.get("id") or new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": map_stop_reason(choices[0].get("finish_reason")),
        "stop_sequence": None,
        "usage": map_usage(payload.get("usage")),
    }


class StreamTranslator:
    """Converts a stream of OpenAI chunks into Anthropic SSE events.

    Stateful by necessity: OpenAI emits tool-call fragments identified by an
    `index`, which must be reassembled into contiguous Anthropic content
    blocks. State is confined to one request.
    """

    def __init__(self, model: str, message_id: str | None = None) -> None:
        self.model = model
        self.message_id = message_id or new_message_id()
        self._started = False
        self._finished = False
        self._next_index = 0
        self._open_index: int | None = None
        self._text_index: int | None = None
        self._tool_indices: dict[int, int] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] = {}

    # ------------------------------------------------------------- internals

    def _start(self) -> Iterator[Event]:
        if self._started:
            return
        self._started = True
        yield (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    def _close_open_block(self) -> Iterator[Event]:
        if self._open_index is None:
            return
        yield ("content_block_stop", {"type": "content_block_stop", "index": self._open_index})
        self._open_index = None

    def _open_text_block(self) -> Iterator[Event]:
        if self._text_index is not None and self._open_index == self._text_index:
            return
        yield from self._close_open_block()
        self._text_index = self._next_index
        self._next_index += 1
        self._open_index = self._text_index
        yield (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self._text_index,
                "content_block": {"type": "text", "text": ""},
            },
        )

    def _open_tool_block(self, tool_index: int, call: dict[str, Any]) -> Iterator[Event]:
        yield from self._close_open_block()
        index = self._next_index
        self._next_index += 1
        self._tool_indices[tool_index] = index
        self._open_index = index
        fn = call.get("function") or {}
        yield (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": fn.get("name", ""),
                    "input": {},
                },
            },
        )

    # ---------------------------------------------------------------- public

    def feed(self, chunk: dict[str, Any]) -> Iterator[Event]:
        """Translate one OpenAI chunk into zero or more Anthropic events."""
        yield from self._start()

        if chunk.get("usage"):
            self._usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta") or {}

        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]

        text = delta.get("content")
        if text:
            yield from self._open_text_block()
            yield (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._text_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        for call in delta.get("tool_calls") or []:
            tool_index = call.get("index", 0)
            if tool_index not in self._tool_indices:
                yield from self._open_tool_block(tool_index, call)
            block_index = self._tool_indices[tool_index]

            # A fragment may reopen a block that a later tool call closed.
            if self._open_index != block_index:
                yield from self._close_open_block()
                self._open_index = block_index

            fragment = (call.get("function") or {}).get("arguments")
            if fragment:
                yield (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": fragment},
                    },
                )

    def finish(self) -> Iterator[Event]:
        """Close the message. Safe to call after an upstream error."""
        if self._finished:
            return
        self._finished = True
        yield from self._start()
        yield from self._close_open_block()

        # Tool calls that never produced a finish_reason still end the turn as
        # tool_use -- otherwise Claude Code treats the turn as final text.
        reason = self._finish_reason
        if reason is None and self._tool_indices:
            reason = "tool_calls"

        yield (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": map_stop_reason(reason), "stop_sequence": None},
                "usage": {"output_tokens": map_usage(self._usage)["output_tokens"]},
            },
        )
        yield ("message_stop", {"type": "message_stop"})


def format_sse(event: Event) -> str:
    """Render an (name, data) pair as an SSE frame."""
    name, data = event
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
