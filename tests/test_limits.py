"""Context-window reconciliation."""

from __future__ import annotations

import pytest

from ccproxy.limits import (
    PromptTooLong,
    clamp_max_tokens,
    estimate_input_tokens,
    parse_context_error,
)

# The rejection observed from the real gateway.
REAL_ERROR = (
    '{"error":{"message":"Error code: 400 - {\'error\': {\'message\': \\"This model\'s '
    "maximum context length is 262144 tokens. However, you requested 64000 output tokens "
    "and your prompt contains at least 198145 input tokens, for a total of at least 262145 "
    "tokens. Please reduce the length of the input prompt or the number of requested output "
    "tokens. (parameter=input_tokens, value=198145)\\\", 'type': 'BadRequestError', 'param': "
    '\'input_tokens\', \'code\': 400}}","type":"api_error","code":400}'
)


class TestParseContextError:
    def test_extracts_both_numbers_from_the_real_error(self):
        overflow = parse_context_error(REAL_ERROR)
        assert overflow is not None
        assert overflow.input_tokens == 198145
        assert overflow.context_window == 262144

    def test_unrelated_400_is_not_treated_as_overflow(self):
        assert parse_context_error('{"error":{"message":"tool schema invalid"}}') is None

    def test_alternate_wording_still_recognised(self):
        body = "context_length_exceeded: prompt contains at least 5000 input tokens"
        overflow = parse_context_error(body)
        assert overflow is not None
        assert overflow.input_tokens == 5000

    def test_marker_without_numbers_still_flags_overflow(self):
        overflow = parse_context_error("maximum context length reached")
        assert overflow is not None
        assert overflow.input_tokens is None


class TestEstimate:
    def test_scales_with_content(self):
        small = {"messages": [{"role": "user", "content": "hi"}]}
        large = {"messages": [{"role": "user", "content": "x" * 10000}]}
        assert estimate_input_tokens(large) > estimate_input_tokens(small) * 100

    def test_tool_schemas_are_counted(self):
        base = {"messages": [{"role": "user", "content": "hi"}]}
        with_tools = {**base, "tools": [{"function": {"name": "x" * 500}}]}
        assert estimate_input_tokens(with_tools) > estimate_input_tokens(base)


class TestClamp:
    def test_the_real_failure_is_prevented(self):
        """198145 input + 64000 requested must not exceed a 262144 window."""
        payload = {"max_tokens": 64000, "messages": []}
        out = clamp_max_tokens(
            payload, context_window=262144, margin=3072, floor=512, input_tokens=198145
        )
        assert out["max_tokens"] == 262144 - 198145 - 3072
        assert 198145 + out["max_tokens"] < 262144

    def test_small_prompts_keep_the_requested_budget(self):
        payload = {"max_tokens": 8192, "messages": []}
        out = clamp_max_tokens(
            payload, context_window=262144, margin=3072, floor=512, input_tokens=1000
        )
        assert out["max_tokens"] == 8192

    def test_ceiling_is_applied(self):
        payload = {"max_tokens": 64000, "messages": []}
        out = clamp_max_tokens(
            payload,
            context_window=262144,
            margin=3072,
            floor=512,
            ceiling=4096,
            input_tokens=1000,
        )
        assert out["max_tokens"] == 4096

    def test_hopeless_prompt_raises_with_anthropic_wording(self):
        with pytest.raises(PromptTooLong, match="prompt is too long"):
            clamp_max_tokens(
                {"max_tokens": 4096, "messages": []},
                context_window=262144,
                margin=3072,
                floor=512,
                input_tokens=261000,
            )

    def test_payload_is_not_mutated(self):
        payload = {"max_tokens": 64000, "messages": []}
        clamp_max_tokens(
            payload, context_window=262144, margin=3072, floor=512, input_tokens=198145
        )
        assert payload["max_tokens"] == 64000

    def test_unchanged_payload_is_returned_as_is(self):
        payload = {"max_tokens": 100, "messages": []}
        assert (
            clamp_max_tokens(
                payload, context_window=262144, margin=3072, floor=512, input_tokens=10
            )
            is payload
        )

    def test_missing_max_tokens_gets_the_available_budget(self):
        out = clamp_max_tokens(
            {"messages": []}, context_window=10000, margin=100, floor=512, input_tokens=1000
        )
        assert out["max_tokens"] == 8900
