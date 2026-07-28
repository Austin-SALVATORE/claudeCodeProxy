"""End-to-end tests through the FastAPI app with a mocked upstream."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from ccproxy.app import create_app
from ccproxy.config import DEFAULT_TIER_MODELS, Settings
from ccproxy.upstream import redact_proxy

UPSTREAM = "https://gw.example/v1"
COMPLETIONS = f"{UPSTREAM}/chat/completions"


@pytest.fixture
def settings() -> Settings:
    return Settings(upstream_url=UPSTREAM, tier_models=dict(DEFAULT_TIER_MODELS))


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def sse(*chunks: dict) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


class TestHealth:
    def test_reports_configured_models(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["models"]["sonnet"] == "qwen-latest"


class TestNonStreaming:
    @respx.mock
    def test_translates_a_turn_and_maps_the_model(self, client):
        route = respx.post(COMPLETIONS).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                },
            )
        )
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        # the tier mapping was applied on the way out
        assert json.loads(route.calls[0].request.content)["model"] == "qwen-latest"
        body = r.json()
        assert body["content"] == [{"type": "text", "text": "hi"}]
        # the response echoes the model the client asked for, not the upstream one
        assert body["model"] == "claude-sonnet-4-5-20250929"

    @respx.mock
    def test_haiku_tier_routes_to_its_own_model(self, client):
        route = respx.post(COMPLETIONS).mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "t"}, "finish_reason": "stop"}]}
            )
        )
        client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert json.loads(route.calls[0].request.content)["model"] == "qwen-3-coder-next"

    def test_missing_model_is_rejected(self, client):
        r = client.post("/v1/messages", json={"messages": []})
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "invalid_request_error"

    @respx.mock
    def test_upstream_error_becomes_anthropic_error(self, client):
        respx.post(COMPLETIONS).mock(return_value=httpx.Response(429, text="slow down"))
        r = client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 429
        assert r.json()["error"]["type"] == "rate_limit_error"

    @respx.mock
    def test_unreachable_upstream_is_502(self, client):
        respx.post(COMPLETIONS).mock(side_effect=httpx.ConnectError("no route"))
        r = client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 502


class TestStreaming:
    @respx.mock
    def test_emits_a_well_formed_anthropic_stream(self, client):
        respx.post(COMPLETIONS).mock(
            return_value=httpx.Response(
                200,
                content=sse(
                    {"choices": [{"delta": {"content": "he"}}]},
                    {"choices": [{"delta": {"content": "llo"}}]},
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    },
                ),
                headers={"Content-Type": "text/event-stream"},
            )
        )
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        events = [ln[7:] for ln in r.text.splitlines() if ln.startswith("event: ")]
        assert events[0] == "message_start"
        assert events[-1] == "message_stop"
        assert "content_block_delta" in events

    @respx.mock
    def test_tool_call_stream_survives_fragmentation(self, client):
        respx.post(COMPLETIONS).mock(
            return_value=httpx.Response(
                200,
                content=sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "c1",
                                            "function": {"name": "Read", "arguments": '{"p"'},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": ':"a.py"}'}}
                                    ]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                ),
                headers={"Content-Type": "text/event-stream"},
            )
        )
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "stream": True,
                "messages": [{"role": "user", "content": "read a.py"}],
            },
        )
        frames = [json.loads(ln[6:]) for ln in r.text.splitlines() if ln.startswith("data: ")]
        start = next(f for f in frames if f.get("type") == "content_block_start")
        assert start["content_block"]["name"] == "Read"
        partial = "".join(
            f["delta"]["partial_json"]
            for f in frames
            if f.get("type") == "content_block_delta" and f["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(partial) == {"p": "a.py"}
        final = next(f for f in frames if f.get("type") == "message_delta")
        assert final["delta"]["stop_reason"] == "tool_use"

    @respx.mock
    def test_mid_stream_failure_still_closes_the_message(self, client):
        """A hung stream is the worst failure mode -- it must always terminate."""
        respx.post(COMPLETIONS).mock(side_effect=httpx.ReadTimeout("gone"))
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        events = [ln[7:] for ln in r.text.splitlines() if ln.startswith("event: ")]
        assert "error" in events
        assert events[-1] == "message_stop"

    @respx.mock
    def test_upstream_non_200_on_stream_is_reported(self, client):
        respx.post(COMPLETIONS).mock(return_value=httpx.Response(503, text="unavailable"))
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert "event: error" in r.text
        assert r.text.rstrip().endswith('{"type": "message_stop"}')


class TestCountTokens:
    def test_returns_an_approximation(self, client):
        r = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "x" * 400}],
            },
        )
        assert r.status_code == 200
        assert r.json()["input_tokens"] > 50


class TestRedaction:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("http://u:p@proxy:8080", "http://u:***@proxy:8080"),
            ("http://proxy:8080", "http://proxy:8080"),
            ("", "none"),
        ],
    )
    def test_proxy_password_never_logged(self, given, expected):
        assert redact_proxy(given) == expected
