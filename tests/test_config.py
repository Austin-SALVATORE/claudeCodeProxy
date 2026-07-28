"""Model-tier mapping and configuration validation."""

from __future__ import annotations

import pytest

from ccproxy.config import DEFAULT_TIER_MODELS, ConfigError, Settings, from_env


def make(**overrides) -> Settings:
    base = {"upstream_url": "https://gw.example/v1", "tier_models": dict(DEFAULT_TIER_MODELS)}
    return Settings(**{**base, **overrides})


class TestModelResolution:
    def test_dated_claude_ids_resolve_by_tier(self):
        s = make()
        assert s.resolve_model("claude-sonnet-4-5-20250929") == "qwen-latest"
        assert s.resolve_model("claude-opus-4-6-20260514") == "gemma-4-31b-ITG"
        assert s.resolve_model("claude-3-5-haiku-20241022") == "qwen-3-coder-next"

    def test_tier_models_are_parameterizable(self):
        s = make(
            tier_models={"opus": "llama3.3-70b", "sonnet": "gpt-oss-120b", "haiku": "llama3.1-8b"}
        )
        assert s.resolve_model("claude-opus-4-6") == "llama3.3-70b"
        assert s.resolve_model("claude-sonnet-4-5") == "gpt-oss-120b"
        assert s.resolve_model("claude-3-5-haiku") == "llama3.1-8b"

    def test_case_insensitive(self):
        assert make().resolve_model("Claude-SONNET-4-5") == "qwen-latest"

    def test_unknown_model_falls_back_to_default_tier(self):
        assert make().resolve_model("some-unknown-model") == "qwen-latest"
        assert make(default_tier="haiku").resolve_model("mystery") == "qwen-3-coder-next"

    def test_explicit_map_wins_over_tier_matching(self):
        s = make(model_map={"claude-sonnet-4-5-20250929": "mistral-medium-2508-ITG"})
        assert s.resolve_model("claude-sonnet-4-5-20250929") == "mistral-medium-2508-ITG"
        # a different sonnet id still uses the tier
        assert s.resolve_model("claude-sonnet-9-9") == "qwen-latest"

    def test_tier_of_reports_matched_tier(self):
        s = make()
        assert s.tier_of("claude-opus-4-6") == "opus"
        assert s.tier_of("nonsense") == "sonnet"


class TestValidation:
    def test_upstream_url_required(self):
        with pytest.raises(ConfigError, match="UPSTREAM_URL"):
            Settings(upstream_url="")

    def test_every_tier_needs_a_model(self):
        with pytest.raises(ConfigError, match="haiku"):
            make(tier_models={"opus": "a", "sonnet": "b", "haiku": ""})

    def test_bad_default_tier_rejected(self):
        with pytest.raises(ConfigError, match="DEFAULT_TIER"):
            make(default_tier="turbo")

    def test_bad_tls_mode_rejected(self):
        with pytest.raises(ConfigError, match="CCPROXY_TLS"):
            make(tls="whatever")

    def test_insecure_tls_requires_explicit_optin(self):
        with pytest.raises(ConfigError, match="disables certificate verification"):
            make(tls="insecure")
        assert make(tls="insecure", allow_insecure=True).tls == "insecure"

    def test_missing_ca_bundle_rejected(self):
        with pytest.raises(ConfigError, match="does not exist"):
            make(ca_bundle="/no/such/bundle.pem")


class TestFromEnv:
    def test_reads_tier_models_from_environment(self, monkeypatch):
        monkeypatch.setenv("CCPROXY_UPSTREAM_URL", "https://gw.example/v1/openai/")
        monkeypatch.setenv("CCPROXY_MODEL_OPUS", "gpt-oss-120b")
        monkeypatch.setenv("CCPROXY_MODEL_SONNET", "llama3.3-70b")
        monkeypatch.setenv("CCPROXY_MODEL_HAIKU", "llama3.1-8b")
        s = from_env()
        assert s.upstream_url == "https://gw.example/v1/openai"  # trailing slash stripped
        assert s.tier_models == {
            "opus": "gpt-oss-120b",
            "sonnet": "llama3.3-70b",
            "haiku": "llama3.1-8b",
        }

    def test_unset_tiers_use_defaults(self, monkeypatch):
        monkeypatch.setenv("CCPROXY_UPSTREAM_URL", "https://gw.example/v1")
        monkeypatch.delenv("CCPROXY_MODEL_OPUS", raising=False)
        assert from_env().tier_models["opus"] == DEFAULT_TIER_MODELS["opus"]

    def test_model_map_json_parsed(self, monkeypatch):
        monkeypatch.setenv("CCPROXY_UPSTREAM_URL", "https://gw.example/v1")
        monkeypatch.setenv("CCPROXY_MODEL_MAP", '{"a": "b"}')
        assert from_env().model_map == {"a": "b"}

    def test_malformed_model_map_rejected(self, monkeypatch):
        monkeypatch.setenv("CCPROXY_UPSTREAM_URL", "https://gw.example/v1")
        monkeypatch.setenv("CCPROXY_MODEL_MAP", "not json")
        with pytest.raises(ConfigError, match="JSON object"):
            from_env()

    def test_non_numeric_port_rejected(self, monkeypatch):
        monkeypatch.setenv("CCPROXY_UPSTREAM_URL", "https://gw.example/v1")
        monkeypatch.setenv("CCPROXY_PORT", "eight")
        with pytest.raises(ConfigError, match="integer"):
            from_env()
