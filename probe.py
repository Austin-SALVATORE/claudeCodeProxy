# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Probe an OpenAI-compatible endpoint for the capabilities Claude Code needs.

Usage:
    export PROBE_BASE_URL="https://your-gateway/v1"
    export PROBE_API_KEY="sk-..."
    export PROBE_MODEL="your-model-name"   # optional; auto-picked from /models
    uv run probe.py

Prints a capability report. Sends no data anywhere except your endpoint.
"""

import json
import os
import sys

import httpx

BASE_URL = os.environ.get("PROBE_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("PROBE_API_KEY", "")
MODEL = os.environ.get("PROBE_MODEL", "")

if not BASE_URL:
    sys.exit("set PROBE_BASE_URL (e.g. https://gateway.internal/v1)")

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

TIMEOUT = httpx.Timeout(60.0, connect=15.0)
results: dict[str, object] = {}


def report(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" :: {detail}" if detail else ""), flush=True)
    results[name] = {"ok": ok, "detail": detail}


def probe_models(client: httpx.Client) -> list[str]:
    try:
        r = client.get(f"{BASE_URL}/models", headers=HEADERS)
    except Exception as exc:
        report("GET /models", False, f"{type(exc).__name__}: {exc}")
        return []
    if r.status_code != 200:
        report("GET /models", False, f"HTTP {r.status_code}: {r.text[:200]}")
        return []
    try:
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception as exc:
        report("GET /models", False, f"unparseable body: {exc}")
        return []
    report("GET /models", True, f"{len(ids)} models: {', '.join(ids[:12])}")
    return ids


def probe_basic(client: httpx.Client, model: str) -> None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16,
    }
    try:
        r = client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=body)
    except Exception as exc:
        report("chat.completions (non-stream)", False, f"{type(exc).__name__}: {exc}")
        return
    if r.status_code != 200:
        report("chat.completions (non-stream)", False, f"HTTP {r.status_code}: {r.text[:300]}")
        return
    data = r.json()
    text = data["choices"][0]["message"].get("content")
    usage = data.get("usage", {})
    report("chat.completions (non-stream)", True, f"content={text!r} usage={usage}")
    results["_usage_keys"] = sorted(usage.keys())


def probe_stream(client: httpx.Client, model: str) -> None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count: one two three"}],
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunks, saw_usage, saw_done = 0, False, False
    try:
        with client.stream("POST", f"{BASE_URL}/chat/completions", headers=HEADERS, json=body) as r:
            if r.status_code != 200:
                report("chat.completions (stream)", False, f"HTTP {r.status_code}")
                return
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    break
                chunks += 1
                try:
                    if json.loads(payload).get("usage"):
                        saw_usage = True
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        report("chat.completions (stream)", False, f"{type(exc).__name__}: {exc}")
        return
    report(
        "chat.completions (stream)",
        chunks > 0,
        f"{chunks} chunks, [DONE]={saw_done}, stream_options.include_usage={saw_usage}",
    )
    results["_stream_usage"] = saw_usage


TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def probe_tools(client: httpx.Client, model: str) -> None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        "tools": [TOOL],
        "tool_choice": "auto",
        "max_tokens": 128,
    }
    try:
        r = client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=body)
    except Exception as exc:
        report("tool calling (non-stream)", False, f"{type(exc).__name__}: {exc}")
        return
    if r.status_code != 200:
        report("tool calling (non-stream)", False, f"HTTP {r.status_code}: {r.text[:300]}")
        return
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        report("tool calling (non-stream)", False, f"no tool_calls; content={str(msg.get('content'))[:200]!r}")
        return
    fn = calls[0]["function"]
    ok_args = isinstance(fn.get("arguments"), str)
    report(
        "tool calling (non-stream)",
        True,
        f"name={fn.get('name')} arguments_is_json_string={ok_args} raw={str(fn.get('arguments'))[:120]}",
    )
    results["_tool_id_present"] = bool(calls[0].get("id"))


def probe_tools_stream(client: httpx.Client, model: str) -> None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}],
        "tools": [TOOL],
        "tool_choice": "auto",
        "max_tokens": 128,
        "stream": True,
    }
    frags, saw_index, saw_id = 0, False, False
    try:
        with client.stream("POST", f"{BASE_URL}/chat/completions", headers=HEADERS, json=body) as r:
            if r.status_code != 200:
                report("tool calling (stream)", False, f"HTTP {r.status_code}")
                return
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                for tc in delta.get("tool_calls") or []:
                    frags += 1
                    saw_index = saw_index or ("index" in tc)
                    saw_id = saw_id or bool(tc.get("id"))
    except Exception as exc:
        report("tool calling (stream)", False, f"{type(exc).__name__}: {exc}")
        return
    report(
        "tool calling (stream)",
        frags > 0,
        f"{frags} tool_call deltas, has index={saw_index}, has id={saw_id}",
    )


def probe_multi_tool(client: httpx.Client, model: str) -> None:
    """Claude Code routinely emits several tool calls in one turn."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Get the weather for Paris AND Tokyo. Call the tool twice."}],
        "tools": [TOOL],
        "tool_choice": "auto",
        "max_tokens": 256,
    }
    try:
        r = client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=body)
        calls = r.json()["choices"][0]["message"].get("tool_calls") or []
    except Exception as exc:
        report("parallel tool calls", False, f"{type(exc).__name__}: {exc}")
        return
    report("parallel tool calls", len(calls) >= 2, f"{len(calls)} calls in one turn")


def probe_system_and_roundtrip(client: httpx.Client, model: str) -> None:
    """Feed a tool result back — the half that most cheap proxies get wrong."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "18C, light rain"},
        ],
        "tools": [TOOL],
        "max_tokens": 64,
    }
    try:
        r = client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=body)
    except Exception as exc:
        report("tool-result round trip", False, f"{type(exc).__name__}: {exc}")
        return
    if r.status_code != 200:
        report("tool-result round trip", False, f"HTTP {r.status_code}: {r.text[:300]}")
        return
    content = r.json()["choices"][0]["message"].get("content")
    report("tool-result round trip", bool(content), f"content={str(content)[:160]!r}")


def probe_context_window(client: httpx.Client, model: str) -> None:
    """Claude Code sends huge prompts. Find out where this endpoint gives up."""
    filler = "The quick brown fox jumps over the lazy dog. " * 2200  # ~25k tokens
    body = {
        "model": model,
        "messages": [{"role": "user", "content": filler + "\n\nReply with exactly: OK"}],
        "max_tokens": 16,
    }
    try:
        r = client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=body)
    except Exception as exc:
        report("~25k-token prompt", False, f"{type(exc).__name__}: {exc}")
        return
    if r.status_code != 200:
        report("~25k-token prompt", False, f"HTTP {r.status_code}: {r.text[:300]}")
        return
    report("~25k-token prompt", True, f"accepted; usage={r.json().get('usage', {})}")


def main() -> None:
    print(f"Probing {BASE_URL}\n" + "=" * 60)
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        ids = probe_models(client)
        model = MODEL or (ids[0] if ids else "")
        if not model:
            sys.exit("\nNo model available. Set PROBE_MODEL explicitly.")
        print(f"Using model: {model}\n" + "-" * 60)
        probe_basic(client, model)
        probe_stream(client, model)
        probe_tools(client, model)
        probe_tools_stream(client, model)
        probe_multi_tool(client, model)
        probe_system_and_roundtrip(client, model)
        probe_context_window(client, model)
    print("=" * 60)
    print("SUMMARY:", json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
