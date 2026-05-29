"""Sanitize hardcoded repo paths in user-provided test/eval commands.

When users supply ``--test-command "cd /path && bash run_eval.sh"``, the
referenced scripts may contain hardcoded absolute paths to the target repo
(e.g. ``/sgl-workspace/aiter``).  GEAK creates worktrees for each
optimization agent, but if the script points to the original repo, the
worktree's modifications are never tested.

This module sends the entire test command to a subagent that discovers
referenced scripts (handling ``cd`` + relative paths, sourced scripts,
etc.), detects hardcoded repo paths, and rewrites them using GEAK env vars
(``GEAK_WORK_DIR``, ``GEAK_GPU_DEVICE``, etc.).

Only triggered when the user explicitly provides a test command.  Does
NOT affect auto-generated harnesses (UnitTestAgent path).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from minisweagent import Model

logger = logging.getLogger(__name__)


def sanitize_test_harness(
    test_command: str,
    repo_root: str,
    output_dir: str,
    model: Model,
    *,
    console: object | None = None,
) -> str:
    """Sanitize hardcoded repo paths in scripts referenced by test_command.

    Sends the full ``test_command`` and ``repo_root`` to the
    harness-sanitizer subagent, which discovers referenced scripts,
    checks them for hardcoded paths, rewrites them, and returns a
    new test command pointing to the sanitized copies.

    Returns the (possibly updated) test_command string.
    """
    repo_root = str(Path(repo_root).resolve())
    out_dir = Path(output_dir)

    logger.info(
        "harness_sanitizer: sending test_command to subagent for analysis (repo_root=%s, output_dir=%s)",
        repo_root,
        output_dir,
    )

    try:
        updated_command = _run_sanitizer_agent(
            test_command=test_command,
            repo_root=repo_root,
            output_dir=out_dir,
            model=model,
        )
    except Exception as exc:
        logger.warning(
            "harness_sanitizer: agent failed: %s; using original command",
            exc,
        )
        return test_command

    if updated_command is not None:
        logger.info("harness_sanitizer: command rewritten → %s", updated_command)
        return updated_command

    logger.info("harness_sanitizer: no changes needed")
    return test_command


def _load_sanitizer_spec():
    """Load the harness-sanitizer SubagentSpec from subagents/preprocess/."""
    from minisweagent.run.preprocess_v3.registry import SubagentRegistry

    registry = SubagentRegistry()
    registry.discover()
    return registry.get("harness-sanitizer")


def _run_sanitizer_agent(
    *,
    test_command: str,
    repo_root: str,
    output_dir: Path,
    model: Model,
) -> str | None:
    """Invoke the HarnessSanitizerAgent and return the sanitized command.

    Returns the rewritten test command on success, ``None`` if no changes
    were needed.
    """
    from minisweagent.run.preprocess_v3.subagent import PreprocessSubagent

    spec = _load_sanitizer_spec()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "harness_sanitizer.log"

    agent = PreprocessSubagent(
        model=model,
        system_prompt=spec.system_prompt,
        tools=list(spec.tools) if spec.tools else ["bash", "str_replace_editor"],
        step_limit=spec.max_steps,
        cwd=str(output_dir),
        log_path=log_path,
    )

    task = (
        f"Sanitize a user-provided test command for GEAK worktree isolation.\n\n"
        f"test_command: {test_command}\n"
        f"repo_root: {repo_root}\n"
        f"output_dir: {output_dir}\n\n"
        f"You must do THREE things:\n\n"
        f"1. **Sanitize hardcoded paths**: Discover all scripts referenced by "
        f"the test_command (handle cd + relative paths, sourced scripts, etc.), "
        f"check each for hardcoded occurrences of '{repo_root}', rewrite them "
        f"with GEAK env vars, write sanitized copies to '{output_dir}', and "
        f"output the rewritten test command that uses the sanitized scripts.\n\n"
        f"2. **Detect and inject missing compile step**: Check whether the "
        f"test command and all scripts it calls include a compile/build step "
        f"that recompiles .so files from source (e.g. make, cmake, hipcc, "
        f"setup.py build_ext, pip install -e). If NO compile step exists, "
        f"investigate '{repo_root}' for build files (Makefile, setup.py, "
        f"CMakeLists.txt, pyproject.toml, etc.) and prepend the appropriate "
        f"compile command to the sanitized script so that each GEAK agent "
        f"recompiles from its own modified source tree before running tests.\n\n"
        f"3. **Standardize benchmark output markers**: Check whether the "
        f"script already prints GEAK_RESULT_METRIC/GEAK_RESULT_LATENCY_MS "
        f"markers. If not, detect how benchmark results are reported (e.g. "
        f"mean_us, p50_us, Bandwidth GB/s, TFLOPS, JSON timing fields) and "
        f"append a block that parses the output and emits:\n"
        f"  GEAK_RESULT_METRIC=<float>\n"
        f"  GEAK_RESULT_UNIT=<ms|us|ns|s|GB/s|TB/s|TFLOPS|GFLOPS|items/s>\n"
        f"  GEAK_RESULT_DIRECTION=<lower_is_better|higher_is_better>\n"
        f"Prefer median/p50 over mean. For multi-shape benchmarks, emit the "
        f"geometric mean. Time metrics → lower_is_better, throughput → "
        f"higher_is_better."
    )

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        logger.warning("HarnessSanitizerAgent did not finish successfully: %s", exit_status)

    return _parse_sanitizer_output(result)


def _parse_sanitizer_output(text: str) -> str | None:
    """Parse the agent's output for SANITIZED_COMMAND: or NO_CHANGES_NEEDED."""
    if "NO_CHANGES_NEEDED" in text:
        return None

    match = re.search(r"SANITIZED_COMMAND:\s*(.+)", text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()

    return None
