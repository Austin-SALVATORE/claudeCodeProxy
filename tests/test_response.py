"""OpenAI -> Anthropic response translation, including the SSE state machine."""

from __future__ import annotations

import json

import pytest

from ccproxy.translate.response import (
    StreamTranslator,
    build_anthropic_response,
    format_sse,
    map_stop_reason,
    map_usage,
)


def drain(translator: StreamTranslator, chunks: list[dict]) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for chunk in chunks:
        events.extend(translator.feed(chunk))
    events.extend(translator.finish())
    return events


def text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def tool_chunk(index: int, *, id=None, name=None, args=None, finish=None) -> dict:
    call: dict = {"index": index}
    if id:
        call["id"] = id
    fn = {}
    if name:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        call["function"] = fn
    return {"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": finish}]}


class TestStopReason:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),
            (None, "end_turn"),
            ("weird", "end_turn"),
        ],
    )
    def test_mapping(self, given, expected):
        assert map_stop_reason(given) == expected


class TestUsage:
    def test_renames_token_fields(self):
        assert map_usage({"prompt_tokens": 10, "completion_tokens": 3}) == {
            "input_tokens": 10,
            "output_tokens": 3,
        }

    def test_absent_usage_is_zero(self):
        assert map_usage(None) == {"input_tokens": 0, "output_tokens": 0}


class TestNonStreaming:
    def test_text_response(self):
        out = build_anthropic_response(
            {
                "id": "cmpl-1",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            "claude-sonnet-4-5",
        )
        assert out["content"] == [{"type": "text", "text": "hello"}]
        assert out["stop_reason"] == "end_turn"
        assert out["usage"] == {"input_tokens": 5, "output_tokens": 2}
        assert out["model"] == "claude-sonnet-4-5"

    def test_tool_call_becomes_tool_use_block(self):
        out = build_anthropic_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "Read", "arguments": '{"p":"a.py"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            "m",
        )
        assert out["content"] == [
            {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"p": "a.py"}}
        ]
        assert out["stop_reason"] == "tool_use"

    def test_malformed_arguments_degrade_to_empty(self):
        out = build_anthropic_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c", "function": {"name": "X", "arguments": "{bad"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            "m",
        )
        assert out["content"][0]["input"] == {}


class TestStreaming:
    def test_text_stream_event_order(self):
        events = drain(StreamTranslator("m"), [text_chunk("he"), text_chunk("llo")])
        assert [name for name, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert (
            "".join(d["delta"]["text"] for n, d in events if n == "content_block_delta") == "hello"
        )

    def test_tool_call_fragments_reassemble(self):
        events = drain(
            StreamTranslator("m"),
            [
                tool_chunk(0, id="call_1", name="Read", args='{"p"'),
                tool_chunk(0, args=':"a.py"}'),
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ],
        )
        starts = [d for n, d in events if n == "content_block_start"]
        assert starts[0]["content_block"] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "Read",
            "input": {},
        }
        partial = "".join(
            d["delta"]["partial_json"]
            for n, d in events
            if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(partial) == {"p": "a.py"}

    def test_text_block_is_closed_before_tool_block_opens(self):
        events = drain(
            StreamTranslator("m"),
            [text_chunk("thinking"), tool_chunk(0, id="c1", name="Read", args="{}")],
        )
        names = [n for n, _ in events]
        first_stop = names.index("content_block_stop")
        second_start = names.index("content_block_start", names.index("content_block_start") + 1)
        assert first_stop < second_start, "text block must close before the tool block opens"

    def test_parallel_tool_calls_get_distinct_indices(self):
        events = drain(
            StreamTranslator("m"),
            [
                tool_chunk(0, id="c1", name="Read", args="{}"),
                tool_chunk(1, id="c2", name="Grep", args="{}"),
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ],
        )
        starts = [d for n, d in events if n == "content_block_start"]
        assert [s["index"] for s in starts] == [0, 1]
        assert [s["content_block"]["name"] for s in starts] == ["Read", "Grep"]

    def test_every_opened_block_is_closed(self):
        events = drain(
            StreamTranslator("m"),
            [
                text_chunk("a"),
                tool_chunk(0, id="c1", name="Read", args="{}"),
                tool_chunk(1, id="c2", name="Grep", args="{}"),
            ],
        )
        opened = [d["index"] for n, d in events if n == "content_block_start"]
        closed = [d["index"] for n, d in events if n == "content_block_stop"]
        assert sorted(opened) == sorted(closed)

    def test_usage_reaches_message_delta(self):
        events = drain(
            StreamTranslator("m"),
            [
                text_chunk("hi"),
                {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
            ],
        )
        delta = next(d for n, d in events if n == "message_delta")
        assert delta["usage"]["output_tokens"] == 4

    def test_tool_calls_without_finish_reason_still_stop_as_tool_use(self):
        events = drain(StreamTranslator("m"), [tool_chunk(0, id="c1", name="Read", args="{}")])
        delta = next(d for n, d in events if n == "message_delta")
        assert delta["delta"]["stop_reason"] == "tool_use"

    def test_finish_is_idempotent(self):
        t = StreamTranslator("m")
        list(t.feed(text_chunk("x")))
        assert len(list(t.finish())) > 0
        assert list(t.finish()) == []

    def test_finish_alone_emits_a_valid_empty_message(self):
        """An upstream that dies before any chunk must not hang the client."""
        names = [n for n, _ in StreamTranslator("m").finish()]
        assert names == ["message_start", "message_delta", "message_stop"]


class TestSSEFormat:
    def test_frame_shape(self):
        frame = format_sse(("message_stop", {"type": "message_stop"}))
        assert frame == 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    def test_non_ascii_is_not_escaped(self):
        frame = format_sse(("x", {"t": "café"}))
        assert "café" in frame
