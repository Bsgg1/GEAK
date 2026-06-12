"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mini-swe-agent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "1.14.4"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console

from minisweagent.utils.log import logger

package_dir = Path(__file__).resolve().parent


def get_repo_root() -> Path:
    """Locate the GEAK repository root.

    Checks, in order: GEAK_ROOT env var, /workspace (Docker convention),
    then __file__-relative (works for editable installs where source is
    in the repo tree).
    """
    if env_root := os.environ.get("GEAK_ROOT"):
        p = Path(env_root)
        if p.is_dir():
            return p
    workspace = Path("/workspace")
    if (workspace / "pyproject.toml").exists():
        return workspace
    return package_dir.parent.parent


def get_data_dir(name: str) -> Path:
    """Resolve a bundled data dir ('subagents' or 'skills').

    Order: GEAK_ROOT override -> in-package (wheel/editable) ->
    /workspace (Docker) -> source-tree walk-up -> in-package guess.

    The in-package location is the primary path and works for a plain
    (non-editable) ``pip install``. The remaining branches keep older setups
    (custom GEAK_ROOT tree, /workspace staging, source checkouts) working.

    GEAK_SUBAGENTS_ROOT is intentionally NOT honored here: it historically
    points at the ``subagents/preprocess`` subdir, while callers of this helper
    want the ``subagents`` parent. The preprocess resolver handles that env var
    itself (see run/preprocess_v3/registry.py).
    """
    if r := os.environ.get("GEAK_ROOT"):
        if (p := Path(r) / name).is_dir():
            return p
    p = package_dir / name
    if p.is_dir():
        return p
    if (p := Path("/workspace") / name).is_dir():
        return p
    for c in Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists() and (c / name).is_dir():
            return c / name
    return package_dir / name


def resolve_entry_script(entry_script: str) -> Path | None:
    """Resolve a subagent subprocess ``entry_script`` to an existing file.

    ``entry_script`` is declared in SUBAGENT.yaml as a path that mirrors the
    repo layout (e.g. ``scripts/run-reverse_knowledge.sh``). The scripts that
    back subprocess subagents are bundled under ``src/minisweagent/`` so a plain
    (non-editable) ``pip install`` ships them; the in-package location is tried
    first. GEAK_ROOT, /workspace, and source-checkout roots are kept as
    fallbacks so custom trees, Docker staging, and editable checkouts keep
    working.

    Returns the first existing path, or ``None`` if the script cannot be found.
    """
    rel = Path(entry_script)
    candidates: list[Path] = [package_dir / rel]
    if r := os.environ.get("GEAK_ROOT"):
        candidates.append(Path(r) / rel)
    candidates.append(Path("/workspace") / rel)
    for c in Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            candidates.append(c / rel)
            break
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or user_config_dir("mini-swe-agent"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

if not os.getenv("MSWEA_SILENT_STARTUP"):
    Console().print(
        f"[bold green]GEAK-v3 agent v0.1[/bold green] (core: mini-swe-agent {__version__})\n"
        f"Loading global config from [bold green]'{global_config_file}'[/bold green]"
    )
dotenv.load_dotenv(dotenv_path=global_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "") -> dict[str, str]: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Agent(Protocol):
    """Protocol for agents."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]
    config: Any

    def run(self, task: str, **kwargs) -> tuple[str, str]: ...


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "get_repo_root",
    "get_data_dir",
    "resolve_entry_script",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]
