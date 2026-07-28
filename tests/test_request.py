"""Anthropic -> OpenAI request translation."""

from __future__ import annotations

import json

import pytest

from ccproxy.translate.request import (
    TranslationError,
    build_openai_request,
    translate_messages,
    translate_system,
    translate_tool_choice,
    translate_tools,
)


class TestSystem:
    def test_plain_string(self):
        assert translate_system("be terse") == "be terse"

    def test_block_list_is_joined(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert translate_system(blocks) == "ab"

    def test_absent(self):
        assert translate_system(None) == ""


class TestMessages:
    def test_string_content_passes_through(self):
        assert translate_messages([{"role": "user", "content": "hi"}]) == [
            {"role": "user", "content": "hi"}
        ]

    def test_cache_control_is_stripped(self):
        out = translate_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ]
        )
        assert out == [{"role": "user", "content": "hi"}]

    def test_assistant_tool_use_becomes_tool_calls(self):
        out = translate_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Read",
                            "input": {"p": "a.py"},
                        },
                    ],
                }
            ]
        )
        assert out[0]["content"] == "checking"
        call = out[0]["tool_calls"][0]
        assert call["id"] == "toolu_1"
        assert call["function"]["name"] == "Read"
        # arguments must be a JSON string, not an object
        assert json.loads(call["function"]["arguments"]) == {"p": "a.py"}

    def test_tool_results_become_separate_tool_messages(self):
        out = translate_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "one"},
                        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "two"},
                    ],
                }
            ]
        )
        assert [m["role"] for m in out] == ["tool", "tool"]
        assert out[0]["tool_call_id"] == "toolu_1"
        assert out[1]["content"] == "two"

    def test_tool_result_plus_text_emits_tool_then_user(self):
        out = translate_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "done"},
                        {"type": "text", "text": "now what?"},
                    ],
                }
            ]
        )
        assert [m["role"] for m in out] == ["tool", "user"]
        assert out[1]["content"] == "now what?"

    def test_error_tool_result_is_marked(self):
        out = translate_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ]
        )
        assert out[0]["content"] == "Error: boom"

    def test_empty_tool_result_still_has_content(self):
        out = translate_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": ""}],
                }
            ]
        )
        assert out[0]["content"] == "(no output)"

    def test_base64_image_becomes_data_uri(self):
        out = translate_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
                        },
                    ],
                }
            ]
        )
        parts = out[0]["content"]
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,QUJD"

    def test_unsupported_image_source_raises(self):
        with pytest.raises(TranslationError, match="image source"):
            translate_messages(
                [{"role": "user", "content": [{"type": "image", "source": {"type": "ftp"}}]}]
            )

    def test_input_is_not_mutated(self):
        original = [
            {"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"a": 1}}]}
        ]
        snapshot = json.dumps(original)
        translate_messages(original)
        assert json.dumps(original) == snapshot


class TestTools:
    def test_input_schema_becomes_parameters(self):
        schema = {"type": "object", "properties": {"p": {"type": "string"}}}
        out = translate_tools([{"name": "Read", "description": "d", "input_schema": schema}])
        assert out[0] == {
            "type": "function",
            "function": {"name": "Read", "description": "d", "parameters": schema},
        }

    def test_missing_schema_gets_empty_object(self):
        out = translate_tools([{"name": "X"}])
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_nameless_tools_dropped(self):
        assert translate_tools([{"description": "no name"}]) == []

    @pytest.mark.parametrize(
        "given,expected",
        [
            ({"type": "auto"}, "auto"),
            ({"type": "any"}, "required"),
            ({"type": "none"}, "none"),
            ({"type": "tool", "name": "Read"}, {"type": "function", "function": {"name": "Read"}}),
            (None, None),
        ],
    )
    def test_tool_choice(self, given, expected):
        assert translate_tool_choice(given) == expected


class TestWholeRequest:
    def test_system_is_prepended(self):
        req = build_openai_request(
            {
                "model": "claude-sonnet-4-5",
                "system": "S",
                "messages": [{"role": "user", "content": "hi"}],
            },
            "qwen-latest",
        )
        assert req["messages"][0] == {"role": "system", "content": "S"}
        assert req["model"] == "qwen-latest"

    def test_streaming_requests_usage(self):
        req = build_openai_request(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True}, "m"
        )
        assert req["stream_options"] == {"include_usage": True}

    def test_non_streaming_omits_stream_options(self):
        req = build_openai_request({"messages": [{"role": "user", "content": "hi"}]}, "m")
        assert "stream_options" not in req

    def test_sampling_params_forwarded(self):
        req = build_openai_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "temperature": 0.2,
                "top_p": 0.9,
                "stop_sequences": ["END"],
            },
            "m",
        )
        assert req["max_tokens"] == 100
        assert req["temperature"] == 0.2
        assert req["stop"] == ["END"]

    def test_messages_must_be_a_list(self):
        with pytest.raises(TranslationError, match="messages"):
            build_openai_request({"messages": "nope"}, "m")
