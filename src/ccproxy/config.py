"""Environment-driven configuration.

Every value is resolved once at startup into a frozen Settings object. Nothing
in the request path reads os.environ, so behaviour cannot drift between
requests and tests can build Settings directly.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

TIERS = ("opus", "sonnet", "haiku")

DEFAULT_TIER_MODELS: Mapping[str, str] = {
    "opus": "gemma-4-31b-ITG",
    "sonnet": "qwen-latest",
    "haiku": "qwen-3-coder-next",
}


class ConfigError(ValueError):
    """Raised for invalid configuration. Always fatal at startup."""


@dataclass(frozen=True)
class Settings:
    upstream_url: str
    upstream_api_key: str = ""
    tier_models: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_TIER_MODELS))
    model_map: Mapping[str, str] = field(default_factory=dict)
    default_tier: str = "sonnet"
    tls: str = "system"
    ca_bundle: str = ""
    allow_insecure: bool = False
    host: str = "127.0.0.1"
    port: int = 8787
    timeout: float = 600.0
    log_level: str = "info"
    # Total input+output budget the gateway enforces per request.
    context_window: int = 262144
    context_windows: Mapping[str, int] = field(default_factory=dict)
    # Headroom for the estimate being wrong, plus the template the server adds.
    token_margin: int = 3072
    min_output_tokens: int = 512
    max_output_tokens: int = 0  # 0 = no additional ceiling

    def __post_init__(self) -> None:
        if not self.upstream_url:
            raise ConfigError("CCPROXY_UPSTREAM_URL is required")
        if self.default_tier not in TIERS:
            raise ConfigError(
                f"CCPROXY_DEFAULT_TIER must be one of {TIERS}, got {self.default_tier!r}"
            )
        missing = [t for t in TIERS if not self.tier_models.get(t)]
        if missing:
            raise ConfigError(f"no upstream model configured for tier(s): {', '.join(missing)}")
        if self.tls not in ("system", "certifi", "insecure"):
            raise ConfigError(f"CCPROXY_TLS must be system|certifi|insecure, got {self.tls!r}")
        if self.tls == "insecure" and not self.allow_insecure:
            raise ConfigError(
                "CCPROXY_TLS=insecure disables certificate verification, which would expose "
                "your gateway credentials and source code to anyone on the network path. "
                "Set CCPROXY_ALLOW_INSECURE=1 to override, and only for diagnosis."
            )
        if self.ca_bundle and not Path(self.ca_bundle).is_file():
            raise ConfigError(f"CCPROXY_CA_BUNDLE does not exist: {self.ca_bundle}")
        if self.context_window <= 0:
            raise ConfigError("CCPROXY_CONTEXT_WINDOW must be positive")
        if self.min_output_tokens <= 0:
            raise ConfigError("CCPROXY_MIN_OUTPUT_TOKENS must be positive")
        if self.context_window <= self.token_margin + self.min_output_tokens:
            raise ConfigError(
                "CCPROXY_CONTEXT_WINDOW leaves no room once the margin and minimum "
                "output reservation are subtracted"
            )

    def window_for(self, upstream_model: str) -> int:
        """Context window for an upstream model, falling back to the global one."""
        return self.context_windows.get(upstream_model, self.context_window)

    def resolve_model(self, requested: str) -> str:
        """Map a requested Claude model id onto an upstream model id.

        Resolution order:
          1. exact match in the explicit model_map
          2. tier inferred from the id by substring ("sonnet", "opus", "haiku")
          3. the configured default tier

        Substring matching is deliberate: Claude Code sends dated ids such as
        claude-sonnet-4-5-20250929, and those change without notice.
        """
        if requested in self.model_map:
            return self.model_map[requested]
        lowered = requested.lower()
        for tier in TIERS:
            if tier in lowered:
                return self.tier_models[tier]
        return self.tier_models[self.default_tier]

    def tier_of(self, requested: str) -> str:
        """Which tier a requested id resolves to. For logging only."""
        lowered = requested.lower()
        return next((t for t in TIERS if t in lowered), self.default_tier)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_model_map(name: str) -> Mapping[str, str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ConfigError(f"{name} must be a JSON object of string -> string")
    return dict(parsed)


def _env_window_map(name: str) -> Mapping[str, int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, int) and v > 0 for k, v in parsed.items()
    ):
        raise ConfigError(f"{name} must be a JSON object of model -> positive integer")
    return dict(parsed)


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing vars.

    Deliberately minimal rather than pulling in a dependency: real environment
    variables always win, so a shell export can override the file.
    """
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def from_env() -> Settings:
    """Build Settings from the process environment."""
    tier_models = {
        tier: os.environ.get(f"CCPROXY_MODEL_{tier.upper()}", "").strip()
        or DEFAULT_TIER_MODELS[tier]
        for tier in TIERS
    }
    return Settings(
        upstream_url=os.environ.get("CCPROXY_UPSTREAM_URL", "").strip().rstrip("/"),
        upstream_api_key=os.environ.get("CCPROXY_UPSTREAM_API_KEY", "").strip(),
        tier_models=tier_models,
        model_map=_env_model_map("CCPROXY_MODEL_MAP"),
        default_tier=os.environ.get("CCPROXY_DEFAULT_TIER", "sonnet").strip().lower(),
        tls=os.environ.get("CCPROXY_TLS", "system").strip().lower(),
        ca_bundle=os.environ.get("CCPROXY_CA_BUNDLE", "").strip(),
        allow_insecure=_env_bool("CCPROXY_ALLOW_INSECURE"),
        host=os.environ.get("CCPROXY_HOST", "127.0.0.1").strip(),
        port=_env_int("CCPROXY_PORT", 8787),
        timeout=_env_float("CCPROXY_TIMEOUT", 600.0),
        log_level=os.environ.get("CCPROXY_LOG_LEVEL", "info").strip().lower(),
        context_window=_env_int("CCPROXY_CONTEXT_WINDOW", 262144),
        context_windows=_env_window_map("CCPROXY_CONTEXT_WINDOWS"),
        token_margin=_env_int("CCPROXY_TOKEN_MARGIN", 3072),
        min_output_tokens=_env_int("CCPROXY_MIN_OUTPUT_TOKENS", 512),
        max_output_tokens=_env_int("CCPROXY_MAX_OUTPUT_TOKENS", 0),
    )
