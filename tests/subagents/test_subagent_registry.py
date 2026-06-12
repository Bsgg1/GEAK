from __future__ import annotations

import os
from pathlib import Path

import pytest

from minisweagent import package_dir, resolve_entry_script
from minisweagent.subagents import SubAgentRegistry


EXPECTED_BUNDLED_SUBAGENTS = {
    "codebase-explore",
    "gemm-tuning",
    "general-kernel-optimization",
    "pytorch-to-flydsl",
    "reverse-knowledge",
    "speedup-verify",
}


def test_discovers_bundled_subagents() -> None:
    registry = SubAgentRegistry()

    assert EXPECTED_BUNDLED_SUBAGENTS.issubset(set(registry.list_names()))


def test_reverse_knowledge_entry_script_resolves_bundled() -> None:
    """The subprocess subagent's entry_script must resolve to a packaged file.

    Regression guard: ``scripts/`` used to live outside the importable package,
    so a non-editable ``pip install`` never shipped it and the subprocess-mode
    reverse-knowledge subagent could not start. The script now ships under
    ``src/minisweagent/`` and must resolve in-package and be executable.
    """
    registry = SubAgentRegistry()
    descriptor = registry.get("reverse-knowledge")

    assert descriptor is not None
    assert descriptor.execution_mode == "subprocess"
    assert descriptor.entry_script

    resolved = resolve_entry_script(descriptor.entry_script)
    assert resolved is not None, "entry_script did not resolve to any existing file"
    assert resolved.is_file()
    # The resolved script ships inside the importable package.
    assert resolved.is_relative_to(package_dir)
    # subagent_cli executes it directly, so it must carry the executable bit.
    assert os.access(resolved, os.X_OK)


def test_resolve_entry_script_missing_returns_none() -> None:
    assert resolve_entry_script("scripts/does-not-exist-12345.sh") is None


def test_loads_registered_system_prompt() -> None:
    registry = SubAgentRegistry()
    descriptor = registry.get("general-kernel-optimization")

    assert descriptor is not None
    prompt = registry.load_system_prompt(descriptor)
    assert prompt is not None
    assert "kernel" in prompt.lower()


def test_build_tool_schema_includes_registered_agent_names() -> None:
    registry = SubAgentRegistry()
    schema = registry.build_tool_schema()

    assert schema["name"] == "sub_agent"
    assert schema["parameters"]["required"] == ["task"]

    agent_name_schema = schema["parameters"]["properties"]["agent_name"]
    assert set(agent_name_schema["enum"]) >= EXPECTED_BUNDLED_SUBAGENTS


def test_build_taskgen_catalog_uses_descriptors() -> None:
    registry = SubAgentRegistry()
    catalog = registry.build_taskgen_catalog()

    assert "general-kernel-optimization" in catalog
    assert "gemm-tuning" in catalog
    assert "tool_profile" in catalog


def test_match_language_prefers_marker_matches(tmp_path: Path) -> None:
    subagents_dir = tmp_path / "subagents"
    agent_dir = subagents_dir / "triton-agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SUBAGENT.yaml").write_text(
        """
name: triton-agent
description: Handles Triton kernels.
agent:
  language_match:
    extensions: [".py"]
    markers: ["@triton.jit", "import triton"]
    confidence_boost: 0.25
""".strip()
        + "\n",
        encoding="utf-8",
    )

    registry = SubAgentRegistry(subagents_dir=subagents_dir)
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kernel(x, y):\n"
        "    tl.store(y, tl.load(x))\n",
        encoding="utf-8",
    )

    assert registry.match_language(str(kernel)) == "triton-agent"


def test_register_from_dict_adds_runtime_descriptor(tmp_path: Path) -> None:
    registry = SubAgentRegistry(subagents_dir=tmp_path)

    descriptor = registry.register_from_dict(
        {
            "name": "temporary-agent",
            "description": "Temporary runtime agent.",
            "execution_mode": "inprocess",
            "parameters": [
                {
                    "name": "task",
                    "type": "string",
                    "description": "Task text.",
                    "required": True,
                }
            ],
            "agent": {"tool_profile": "swe"},
        }
    )

    assert descriptor.name == "temporary-agent"
    assert registry.get("temporary-agent") is descriptor
    assert descriptor.parameters[0].required is True
    assert "temporary-agent" in registry.build_tool_schema()["parameters"]["properties"]["agent_name"]["enum"]


def test_create_subagent_can_persist_definition(tmp_path: Path) -> None:
    registry = SubAgentRegistry(subagents_dir=tmp_path)

    descriptor = registry.create_subagent(
        "persisted-agent",
        "Persisted test agent.",
        agent_config={"tool_profile": "swe"},
        persist=True,
    )

    yaml_path = tmp_path / "persisted-agent" / "SUBAGENT.yaml"
    assert yaml_path.exists()
    assert descriptor.path == tmp_path / "persisted-agent"

    rediscovered = SubAgentRegistry(subagents_dir=tmp_path)
    assert rediscovered.get("persisted-agent") is not None


def test_duplicate_runtime_registration_is_rejected(tmp_path: Path) -> None:
    registry = SubAgentRegistry(subagents_dir=tmp_path)
    definition = {"name": "dupe", "description": "Duplicate."}

    registry.register_from_dict(definition)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_from_dict(definition)

