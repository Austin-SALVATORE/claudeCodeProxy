"""HTTP client for the upstream gateway, including corporate TLS handling."""

from __future__ import annotations

import ssl

import httpx

from .config import Settings


def build_verify(settings: Settings) -> ssl.SSLContext | bool:
    """Resolve TLS trust.

    Corporate networks terminate TLS with a private root CA installed in the OS
    trust store. certifi -- httpx's default -- knows nothing about it, so the
    system store is preferred.
    """
    if settings.tls == "insecure":
        return False
    if settings.ca_bundle:
        return ssl.create_default_context(cafile=settings.ca_bundle)
    if settings.tls == "certifi":
        return True
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return True


def build_client(settings: Settings) -> httpx.AsyncClient:
    headers = {"Content-Type": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    return httpx.AsyncClient(
        base_url=settings.upstream_url,
        headers=headers,
        verify=build_verify(settings),
        timeout=httpx.Timeout(settings.timeout, connect=30.0),
        follow_redirects=True,
    )


def redact_proxy(url: str) -> str:
    """Strip the password from a proxy URL before logging it.

    Corporate proxies are routinely configured as http://user:pass@host:port,
    and logging that verbatim leaks a domain password into scrollback and logs.
    """
    if not url or "@" not in url:
        return url or "none"
    scheme, _, rest = url.partition("://")
    userinfo, _, hostpart = rest.rpartition("@")
    user = userinfo.partition(":")[0]
    return f"{scheme}://{user}:***@{hostpart}" if user else f"{scheme}://***@{hostpart}"
