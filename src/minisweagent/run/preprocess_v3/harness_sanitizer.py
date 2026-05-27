"""Sanitize hardcoded repo paths in user-provided test/eval scripts.

When users supply ``--test-command "bash /path/to/run_eval.sh"``, the
script may contain hardcoded absolute paths to the target repo (e.g.
``/sgl-workspace/aiter``).  GEAK creates worktrees for each optimization
agent, but if the script points to the original repo, the worktree's
modifications are never tested.

This module detects hardcoded repo paths in referenced script files and
invokes a subagent to rewrite them using GEAK env vars
(``GEAK_WORK_DIR``, ``GEAK_GPU_DEVICE``, etc.).

Only triggered when the user explicitly provides a test command.  Does
NOT affect auto-generated harnesses (UnitTestAgent path).
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path

from minisweagent import Model

logger = logging.getLogger(__name__)


def _extract_script_paths(test_command: str) -> list[Path]:
    """Extract file paths referenced in the test command that exist on disk."""
    paths: list[Path] = []
    try:
        tokens = shlex.split(test_command)
    except ValueError:
        tokens = test_command.split()

    for token in tokens:
        if not token.startswith("/"):
            continue
        p = Path(token)
        if p.is_file() and p.suffix in (".sh", ".py", ".bash"):
            paths.append(p)
    return paths


def _has_hardcoded_repo_paths(content: str, repo_root: str) -> bool:
    """Check if the script content contains the repo root as a literal path."""
    normalized = repo_root.rstrip("/")
    return normalized in content


def sanitize_test_harness(
    test_command: str,
    repo_root: str,
    output_dir: str,
    model: Model,
    *,
    console: object | None = None,
) -> str:
    """Sanitize hardcoded repo paths in scripts referenced by test_command.

    Scans each script file referenced in ``test_command`` for hardcoded
    ``repo_root`` paths.  If found, invokes the HarnessSanitizerAgent to
    rewrite the script using GEAK env vars.  The sanitized script is
    written to ``output_dir`` and the returned test_command is updated
    to point to it.

    Returns the (possibly updated) test_command string.
    """
    repo_root = str(Path(repo_root).resolve())
    script_paths = _extract_script_paths(test_command)

    if not script_paths:
        logger.debug("harness_sanitizer: no script files found in test_command")
        return test_command

    scripts_to_sanitize: list[Path] = []
    for sp in script_paths:
        try:
            content = sp.read_text()
        except OSError:
            continue
        if _has_hardcoded_repo_paths(content, repo_root):
            scripts_to_sanitize.append(sp)

    if not scripts_to_sanitize:
        logger.info("harness_sanitizer: no hardcoded repo paths found in referenced scripts")
        return test_command

    logger.info(
        "harness_sanitizer: found hardcoded repo paths in %d script(s): %s",
        len(scripts_to_sanitize),
        [str(p) for p in scripts_to_sanitize],
    )

    updated_command = test_command
    out_dir = Path(output_dir)

    for script_path in scripts_to_sanitize:
        sanitized_name = f"_geak_sanitized_{script_path.name}"
        output_path = out_dir / sanitized_name

        try:
            sanitized_path = _run_sanitizer_agent(
                script_path=script_path,
                repo_root=repo_root,
                output_path=output_path,
                model=model,
                log_dir=out_dir,
            )
        except Exception as exc:
            logger.warning(
                "harness_sanitizer: agent failed for %s: %s; using original",
                script_path,
                exc,
            )
            continue

        if sanitized_path and sanitized_path.is_file():
            updated_command = updated_command.replace(str(script_path), str(sanitized_path))
            logger.info("harness_sanitizer: %s → %s", script_path, sanitized_path)
        else:
            logger.info("harness_sanitizer: no changes needed for %s", script_path)

    return updated_command


def _load_sanitizer_spec():
    """Load the harness-sanitizer SubagentSpec from subagents/preprocess/."""
    from minisweagent.run.preprocess_v3.registry import SubagentRegistry

    registry = SubagentRegistry()
    registry.discover()
    return registry.get("harness-sanitizer")


def _run_sanitizer_agent(
    *,
    script_path: Path,
    repo_root: str,
    output_path: Path,
    model: Model,
    log_dir: Path | None = None,
) -> Path | None:
    """Invoke the HarnessSanitizerAgent and return the sanitized script path.

    Uses the ``harness-sanitizer`` subagent definition from
    ``subagents/preprocess/harness-sanitizer/SUBAGENT.yaml`` via
    :class:`PreprocessSubagent`.

    Returns ``output_path`` on success, ``None`` if no changes were needed.
    """
    from minisweagent.run.preprocess_v3.subagent import PreprocessSubagent

    spec = _load_sanitizer_spec()

    log_path = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "harness_sanitizer.log"

    agent = PreprocessSubagent(
        model=model,
        system_prompt=spec.system_prompt,
        tools=list(spec.tools) if spec.tools else ["bash", "str_replace_editor"],
        step_limit=spec.max_steps,
        cwd=str(script_path.parent),
        log_path=log_path,
    )

    task = (
        f"Sanitize hardcoded repo paths in a test script.\n\n"
        f"script_path: {script_path}\n"
        f"repo_root: {repo_root}\n"
        f"output_path: {output_path}\n\n"
        f"Read the script, replace all hardcoded occurrences of '{repo_root}' "
        f"with GEAK env var patterns, and write the result to '{output_path}'."
    )

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        logger.warning("HarnessSanitizerAgent did not finish successfully: %s", exit_status)

    return _parse_sanitizer_output(result, output_path)


def _parse_sanitizer_output(text: str, expected_output: Path) -> Path | None:
    """Parse the agent's output for SANITIZED: or NO_CHANGES_NEEDED."""
    if "NO_CHANGES_NEEDED" in text:
        return None

    match = re.search(r"SANITIZED:\s*(.+)\s*$", text.strip(), re.MULTILINE)
    if match:
        path = Path(match.group(1).strip())
        if path.is_file():
            return path

    if expected_output.is_file():
        return expected_output

    return None
