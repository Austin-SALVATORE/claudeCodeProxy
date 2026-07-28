"""Anthropic Messages request -> OpenAI Chat Completions request.

Pure functions over plain dicts. Nothing here mutates its input: every helper
builds and returns new structures, so a request can be translated twice and
logged safely.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

# Claude Code marks cache breakpoints with cache_control. The gateway reports no
# cache token accounting, so these are stripped rather than forwarded -- an
# unknown key can trip strict OpenAI-compatible servers.
_STRIPPED_BLOCK_KEYS = frozenset({"cache_control"})

_STOP_REASON_UNSUPPORTED = object()


class TranslationError(ValueError):
    """Raised when a request cannot be represented in the OpenAI schema."""


def _clean(block: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if k not in _STRIPPED_BLOCK_KEYS}


def _text_of(content: Any) -> str:
    """Flatten Anthropic content into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def translate_system(system: Any) -> str:
    """Anthropic `system` (string or list of text blocks) -> a single string."""
    return _text_of(system)


def _image_block(block: dict[str, Any]) -> dict[str, Any]:
    """Anthropic image block -> OpenAI image_url part."""
    source = block.get("source") or {}
    stype = source.get("type")
    if stype == "base64":
        media = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if stype == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    raise TranslationError(f"unsupported image source type: {stype!r}")


def _user_parts(blocks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build OpenAI content parts for the non-tool_result portion of a turn."""
    parts: list[dict[str, Any]] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            parts.append(_image_block(block))
    return parts


def _tool_result_message(block: dict[str, Any]) -> dict[str, Any]:
    """Anthropic tool_result -> OpenAI role:"tool" message.

    Anthropic nests tool results inside a user turn; OpenAI requires one
    standalone message per result, keyed by tool_call_id.
    """
    content = block.get("content")
    text = _text_of(content) if not isinstance(content, str) else content
    if block.get("is_error"):
        text = f"Error: {text}" if text else "Error"
    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": text or "(no output)",
    }


def _assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Anthropic assistant turn -> OpenAI assistant message with tool_calls."""
    text = _text_of(blocks)
    tool_calls = [
        {
            "id": block.get("id", ""),
            "type": "function",
            "function": {
                "name": block.get("name", ""),
                # OpenAI requires arguments as a JSON *string*, not an object.
                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
            },
        }
        for block in blocks
        if block.get("type") == "tool_use"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic messages[] -> OpenAI messages[].

    A single user turn carrying N tool_results expands into N tool messages,
    optionally followed by a user message holding any remaining text or images.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        blocks = [_clean(b) for b in (content or []) if isinstance(b, dict)]

        if role == "assistant":
            out.append(_assistant_message(blocks))
            continue

        results = [b for b in blocks if b.get("type") == "tool_result"]
        out.extend(_tool_result_message(b) for b in results)

        parts = _user_parts(b for b in blocks if b.get("type") != "tool_result")
        if parts:
            # Collapse a lone text part to a plain string: some gateways reject
            # the parts array for text-only turns.
            if len(parts) == 1 and parts[0]["type"] == "text":
                out.append({"role": "user", "content": parts[0]["text"]})
            else:
                out.append({"role": "user", "content": parts})
        elif not results:
            out.append({"role": role or "user", "content": ""})
    return out


def translate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Anthropic tools[] -> OpenAI tools[] function definitions."""
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
        if tool.get("name")
    ]


def translate_tool_choice(choice: Any) -> Any:
    """Anthropic tool_choice -> OpenAI tool_choice."""
    if not choice:
        return None
    ctype = choice.get("type") if isinstance(choice, dict) else None
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    if ctype == "none":
        return "none"
    return None


def build_openai_request(
    body: dict[str, Any],
    upstream_model: str,
    *,
    stream: bool | None = None,
) -> dict[str, Any]:
    """Translate a whole Anthropic Messages request."""
    if not isinstance(body.get("messages"), list):
        raise TranslationError("`messages` must be a list")

    messages: list[dict[str, Any]] = []
    system = translate_system(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(translate_messages(body["messages"]))

    streaming = body.get("stream", False) if stream is None else stream
    request: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(streaming),
    }

    if body.get("max_tokens") is not None:
        request["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            request[key] = body[key]
    if body.get("stop_sequences"):
        request["stop"] = body["stop_sequences"]

    tools = translate_tools(body.get("tools"))
    if tools:
        request["tools"] = tools
        choice = translate_tool_choice(body.get("tool_choice"))
        if choice is not None:
            request["tool_choice"] = choice

    if streaming:
        # Without this the stream carries no usage, and message_delta would
        # have to report output_tokens: 0.
        request["stream_options"] = {"include_usage": True}

    return request
