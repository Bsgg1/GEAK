"""Tests for RAG postprocessor model class routing.

Verifies that RAGPostProcessor creates the correct model type based on model_config,
matching the behavior of the main agent's get_model() routing.
"""

from unittest.mock import patch

import pytest

from minisweagent.models import get_model_class
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.tools.rag_postprocessor import RAGPostProcessor, RAGPostProcessorConfig


def _make_deterministic_model(**kwargs):
    """Factory that creates a DeterministicModel, ignoring unknown kwargs."""
    outputs = kwargs.get("outputs", [make_output("test", [])])
    model_name = kwargs.get("model_name", "test-model")
    return DeterministicModel(outputs=outputs, model_name=model_name)


class TestRAGPostProcessorModelRouting:
    """Verify that postprocessor creates the correct model class based on model_config."""

    def test_model_config_routes_to_get_model(self):
        """model_config is passed through to get_model(), which selects the correct class."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = _make_deterministic_model

            config = RAGPostProcessorConfig(
                enabled=True,
                model_config={"model_class": "amd_llm", "model_name": "claude-opus-4.6"},
            )
            pp = RAGPostProcessor(config)
            model = pp.model

            mock_get_class.assert_called_once_with("claude-opus-4.6", "amd_llm")
            assert isinstance(model, DeterministicModel)

    def test_litellm_model_config_routes_correctly(self):
        """model_config with model_class=litellm routes through get_model_class('litellm')."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = _make_deterministic_model

            config = RAGPostProcessorConfig(
                enabled=True,
                model_config={"model_class": "litellm", "model_name": "gpt-4"},
            )
            pp = RAGPostProcessor(config)
            model = pp.model

            mock_get_class.assert_called_once_with("gpt-4", "litellm")
            assert isinstance(model, DeterministicModel)

    def test_none_model_config_fallback(self):
        """model_config=None still calls get_model() with defaults."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = _make_deterministic_model

            config = RAGPostProcessorConfig(enabled=True, model_config=None)
            pp = RAGPostProcessor(config)
            model = pp.model

            assert mock_get_class.called
            assert isinstance(model, DeterministicModel)

    def test_model_config_not_mutated(self):
        """The original model_config dict is not mutated by model creation."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = _make_deterministic_model

            original = {"model_class": "litellm", "model_name": "test-model"}
            original_copy = dict(original)

            pp = RAGPostProcessor(RAGPostProcessorConfig(enabled=True, model_config=original))
            _ = pp.model

            assert original == original_copy

    def test_model_config_preserves_api_key(self):
        """api_key in model_config.model_kwargs is passed through to the model factory."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            received_kwargs = {}

            def capture_factory(**kwargs):
                received_kwargs.update(kwargs)
                return _make_deterministic_model(**kwargs)

            mock_get_class.return_value = capture_factory

            config = RAGPostProcessorConfig(
                enabled=True,
                model_config={
                    "model_class": "litellm",
                    "model_name": "test",
                    "model_kwargs": {"api_key": "test-key-123"},
                },
            )
            pp = RAGPostProcessor(config)
            _ = pp.model

            assert received_kwargs.get("model_kwargs", {}).get("api_key") == "test-key-123"

    def test_disabled_postprocessor_returns_raw(self):
        """Disabled postprocessor returns raw input without creating a model."""
        config = RAGPostProcessorConfig(enabled=False)
        pp = RAGPostProcessor(config)
        raw = "some raw RAG result"
        assert pp.process(raw) == raw
        assert pp._model is None

    def test_model_created_once(self):
        """Model is lazily created once and reused."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = _make_deterministic_model

            pp = RAGPostProcessor(
                RAGPostProcessorConfig(
                    enabled=True,
                    model_config={"model_class": "amd_llm", "model_name": "test"},
                )
            )
            model1 = pp.model
            model2 = pp.model
            assert model1 is model2


class _RecordingModel:
    """Minimal model double that records the ``tools`` kwarg and returns a
    configurable response, so we can assert how the postprocessor calls it."""

    def __init__(self, content):
        self._content = content
        self.tools = [{"function": {"name": "agent_tool"}}]  # mimic a live agent model
        self.calls = []

    def query(self, messages, **kwargs):
        self.calls.append(kwargs)
        return {"content": self._content}


class TestRAGPostProcessorPassedModel:
    """Behavior when a live (tool-bearing) agent model is reused by the postprocessor."""

    def test_passed_model_takes_precedence(self):
        model = _RecordingModel("cleaned text")
        pp = RAGPostProcessor(RAGPostProcessorConfig(enabled=True), model=model)
        assert pp.model is model

    def test_process_disables_tools_on_call(self):
        # The postprocessor must pass tools=[] so the reused agent model cannot
        # answer with a tool call (which would yield empty content).
        model = _RecordingModel("cleaned text")
        pp = RAGPostProcessor(RAGPostProcessorConfig(enabled=True), model=model)
        out = pp.process("raw rag chunks", query="q")
        assert out == "cleaned text"
        assert model.calls and model.calls[0].get("tools") == []

    def test_empty_response_falls_back_to_raw(self):
        # Regression: if the model returns empty content (e.g. it emitted a tool
        # call), process() must return the raw RAG result, not "".
        raw = "raw rag chunks that must survive"
        for empty in ("", "   ", None):
            model = _RecordingModel(empty)
            pp = RAGPostProcessor(RAGPostProcessorConfig(enabled=True), model=model)
            assert pp.process(raw) == raw


class TestRAGPostProcessorConfig:
    """Verify RAGPostProcessorConfig field defaults and construction."""

    def test_defaults(self):
        cfg = RAGPostProcessorConfig()
        assert cfg.model_config is None
        assert cfg.system_prompt is None
        assert cfg.enabled is True

    def test_with_model_config(self):
        mc = {"model_class": "amd_llm", "model_name": "test"}
        cfg = RAGPostProcessorConfig(model_config=mc)
        assert cfg.model_config is mc

    def test_old_api_key_field_removed(self):
        """Verify that the old api_key/model_name/model_kwargs fields no longer exist."""
        assert "api_key" not in RAGPostProcessorConfig.__dataclass_fields__
        assert "model_name" not in RAGPostProcessorConfig.__dataclass_fields__
        assert "model_kwargs" not in RAGPostProcessorConfig.__dataclass_fields__


class TestGetModelClassRouting:
    """Verify that get_model_class correctly routes model_class strings."""

    def test_amd_llm_resolves(self):
        from minisweagent.models.amd_llm import AmdLlmModel

        assert get_model_class("any-model", "amd_llm") == AmdLlmModel

    def test_deterministic_resolves(self):
        assert get_model_class("any-model", "deterministic") == DeterministicModel

    def test_no_model_class_defaults_to_litellm(self):
        try:
            from minisweagent.models.litellm_model import LitellmModel
        except ImportError:
            pytest.skip("litellm not installed")
        assert get_model_class("any-model", "") == LitellmModel
