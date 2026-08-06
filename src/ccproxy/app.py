"""FastAPI application exposing an Anthropic Messages API over an OpenAI gateway."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings, from_env
from .errors import error_body, upstream_error
from .limits import PromptTooLong, clamp_max_tokens, parse_context_error
from .translate.request import TranslationError, build_openai_request
from .translate.response import StreamTranslator, build_anthropic_response, format_sse
from .upstream import build_client, redact_proxy

log = logging.getLogger("ccproxy")


def fit_to_window(
    payload: dict[str, Any],
    settings: Settings,
    upstream_model: str,
    *,
    input_tokens: int | None = None,
) -> dict[str, Any]:
    """Shrink max_tokens so input + output fits the gateway's single window."""
    return clamp_max_tokens(
        payload,
        context_window=settings.window_for(upstream_model),
        margin=settings.token_margin,
        floor=settings.min_output_tokens,
        ceiling=settings.max_output_tokens or None,
        input_tokens=input_tokens,
    )


def retry_payload(
    payload: dict[str, Any], settings: Settings, upstream_model: str, body: str
) -> dict[str, Any] | None:
    """Rebuild a payload from a gateway context-length rejection.

    The error carries the gateway's own input-token count, which beats any
    estimate we can make. Returns None if this was not a context error, or if
    the numbers leave no room for a reply.
    """
    overflow = parse_context_error(body)
    if overflow is None:
        return None
    window = overflow.context_window or settings.window_for(upstream_model)
    try:
        adjusted = clamp_max_tokens(
            payload,
            context_window=window,
            margin=settings.token_margin,
            floor=settings.min_output_tokens,
            ceiling=settings.max_output_tokens or None,
            input_tokens=overflow.input_tokens,
        )
    except PromptTooLong:
        return None
    if adjusted.get("max_tokens") == payload.get("max_tokens"):
        return None  # nothing changed; retrying would just fail again
    log.warning(
        "context overflow: retrying with max_tokens=%s (input=%s, window=%s)",
        adjusted.get("max_tokens"),
        overflow.input_tokens,
        window,
    )
    return adjusted


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.client = build_client(resolved)
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
        log.info("upstream    : %s", resolved.upstream_url)
        log.info("tls         : %s", resolved.tls)
        log.info("proxy       : %s", redact_proxy(proxy))
        for tier in ("opus", "sonnet", "haiku"):
            log.info("%-12s: %s", tier, resolved.tier_models[tier])
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="ccproxy", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "upstream": resolved.upstream_url,
            "models": dict(resolved.tier_models),
        }

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> JSONResponse:
        """Claude Code calls this before large requests.

        The gateway exposes no tokeniser, so this returns a deliberate
        approximation (~4 chars per token) rather than failing the request.
        """
        body = await request.json()
        chars = len(json.dumps(body.get("messages", []), ensure_ascii=False))
        chars += len(json.dumps(body.get("system", ""), ensure_ascii=False))
        return JSONResponse({"input_tokens": max(1, chars // 4)})

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        settings: Settings = request.app.state.settings
        client: httpx.AsyncClient = request.app.state.client

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            return JSONResponse(status_code=400, content=error_body(400, f"invalid JSON: {exc}"))

        requested = body.get("model", "")
        if not isinstance(requested, str) or not requested:
            return JSONResponse(status_code=400, content=error_body(400, "`model` is required"))

        upstream_model = settings.resolve_model(requested)
        log.info("%s -> %s (%s)", requested, upstream_model, settings.tier_of(requested))

        try:
            payload = build_openai_request(body, upstream_model)
            payload = fit_to_window(payload, settings, upstream_model)
        except TranslationError as exc:
            return JSONResponse(status_code=400, content=error_body(400, str(exc)))
        except PromptTooLong as exc:
            # Claude Code compacts the conversation when it sees this wording.
            log.warning("%s", exc)
            return JSONResponse(status_code=400, content=error_body(400, str(exc)))

        if payload["stream"]:
            return StreamingResponse(
                _stream(client, payload, requested, settings, upstream_model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            response = await client.post("/chat/completions", json=payload)
            if response.status_code == 400:
                retried = retry_payload(payload, settings, upstream_model, response.text)
                if retried is not None:
                    response = await client.post("/chat/completions", json=retried)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - mapped to an Anthropic error
            status, content = upstream_error(exc)
            return JSONResponse(status_code=status, content=content)

        return JSONResponse(build_anthropic_response(response.json(), requested))

    return app


async def _stream(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    model: str,
    settings: Settings,
    upstream_model: str,
    retries_left: int = 1,
) -> AsyncIterator[str]:
    """Proxy the upstream SSE stream, translated to Anthropic events.

    Once the response has started, an upstream failure cannot change the status
    code -- so errors are delivered as an Anthropic `error` event followed by a
    well-formed close, which Claude Code surfaces instead of hanging.
    """
    translator = StreamTranslator(model)
    try:
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                raw = await response.aread()
                detail = raw.decode("utf-8", "replace")
                retried = (
                    retry_payload(payload, settings, upstream_model, detail)
                    if response.status_code == 400 and retries_left > 0
                    else None
                )
                if retried is not None:
                    async for frame in _stream(
                        client, retried, model, settings, upstream_model, retries_left - 1
                    ):
                        yield frame
                    return
                yield format_sse(
                    ("error", error_body(response.status_code, f"upstream: {detail[:500]}"))
                )
                for event in translator.finish():
                    yield format_sse(event)
                return

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    log.warning("skipping unparseable chunk: %s", data[:200])
                    continue
                for event in translator.feed(chunk):
                    yield format_sse(event)

    except Exception as exc:  # noqa: BLE001 - must still close the stream
        status, content = upstream_error(exc)
        log.error("stream failed: %s", exc)
        yield format_sse(("error", content))

    for event in translator.finish():
        yield format_sse(event)
