"""Anthropic-shaped error responses.

Claude Code parses the Anthropic error envelope. Returning an OpenAI error, or
a bare FastAPI {"detail": ...}, surfaces as an unhelpful client-side crash.
"""

from __future__ import annotations

from typing import Any

import httpx

# HTTP status -> Anthropic error type
_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def error_body(status: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": _TYPES.get(status, "api_error"), "message": message},
    }


def upstream_error(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Map an upstream failure onto a status and Anthropic error body."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        detail = exc.response.text[:500] or exc.response.reason_phrase
        return status, error_body(status, f"upstream returned {status}: {detail}")
    if isinstance(exc, httpx.ConnectError):
        cause = exc.__cause__
        if isinstance(cause, ssl_error_types()):
            return 502, error_body(
                502,
                "TLS verification failed against the upstream gateway. If a corporate CA "
                "intercepts this connection, install its root certificate or set "
                "CCPROXY_CA_BUNDLE. Details: " + str(exc),
            )
        return 502, error_body(502, f"cannot reach upstream gateway: {exc}")
    if isinstance(exc, httpx.TimeoutException):
        return 504, error_body(504, f"upstream timed out: {exc}")
    return 500, error_body(500, f"{type(exc).__name__}: {exc}")


def ssl_error_types() -> tuple[type[BaseException], ...]:
    import ssl

    return (ssl.SSLError, ssl.SSLCertVerificationError)
