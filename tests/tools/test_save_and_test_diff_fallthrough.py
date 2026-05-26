"""Regression tests for ``SaveAndTestTool._get_patch_content`` fall-through.

The bug being pinned here:

    When a slot worktree is created via ``git worktree add`` without
    ``--recurse-submodules``, the underlying repo's submodules (e.g.
    ``3rdparty/composable_kernel``) are missing their ``.git`` files.
    ``git add -N . && git diff -- .`` then aborts with::

        fatal: not a git repository: '3rdparty/composable_kernel/.git'

    On the previous code path that error was swallowed: ``result.stdout``
    was returned verbatim (empty string), the orchestrator recorded a
    zero-byte ``patch_*.patch``, and the round was discarded as
    "no changes detected" even when the agent had really edited
    ``kernel.py`` and the test had really passed.

The fix is twofold:

  1. ``3rdparty/`` is added to the git-branch exclude list (and the bare
     ``3rdparty`` token to the ``diff -ruN`` backup branch's exclude list)
     so the common case never trips the submodule traversal.
  2. The git branch now falls through to the ``diff -ruN`` backup branch
     when ``git diff`` returns non-zero or empty stdout, instead of
     returning empty unconditionally. ``diff -ruN`` does not understand
     submodule pointers and therefore is robust to this failure mode.

These tests pin both behaviours.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

from minisweagent.tools.save_and_test import SaveAndTestContext, SaveAndTestTool


def _make_tool(tmp_path: Path, *, base_repo: Path | None) -> SaveAndTestTool:
    """Build a tool wired to ``tmp_path`` with optional ``base_repo`` fallback."""
    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command=None,
            timeout=10,
            patch_output_dir=None,
            base_repo_path=base_repo,
        )
    )
    return tool


def _git_diff_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    """Build a ``CompletedProcess`` shaped like ``git diff``'s output."""
    return subprocess.CompletedProcess(
        args=["git", "add", "-N", ".", "&&", "git", "diff"],
        returncode=returncode,
        stdout=stdout,
        stderr="" if returncode == 0 else "fatal: not a git repository: '3rdparty/composable_kernel/.git'\n",
    )


def _diff_ruN_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["diff", "-ruN"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def test_git_branch_returns_stdout_when_nonempty_and_zero_exit(tmp_path):
    """Regression guard: the happy path must keep returning git's stdout verbatim."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    git_patch = "diff --git a/kernel.py b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n"

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            return_value=_git_diff_result(git_patch, returncode=0),
        ) as run,
    ):
        out = tool._get_patch_content()

    assert out == git_patch
    # The fall-through branch must NOT have been invoked when git already
    # produced a real patch.
    assert run.call_count == 1


def test_git_branch_falls_through_on_nonzero_exit(tmp_path):
    """Submodule-missing case: git diff fails -> diff -ruN backup runs."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    (base_repo / "kernel.py").write_text("old\n")
    (tmp_path / "kernel.py").write_text("new\n")
    tool = _make_tool(tmp_path, base_repo=base_repo)

    diff_ruN_patch = (
        f"diff -ruN {base_repo}/kernel.py {tmp_path}/kernel.py\n"
        f"--- {base_repo}/kernel.py\t2026-01-01 00:00:00 +0000\n"
        f"+++ {tmp_path}/kernel.py\t2026-01-01 00:00:01 +0000\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    side_effects = [
        _git_diff_result("", returncode=128),  # git aborted on missing submodule .git
        _diff_ruN_result(diff_ruN_patch, returncode=1),  # diff -ruN exits 1 when files differ
    ]

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=side_effects,
        ) as run,
    ):
        out = tool._get_patch_content()

    assert run.call_count == 2, "fall-through must invoke diff -ruN backup branch"
    # The output should be the normalised git-style patch derived from diff -ruN.
    assert out.startswith("diff --git a/kernel.py b/kernel.py"), (
        f"expected normalised git-style header, got: {out[:120]!r}"
    )
    assert "+new" in out and "-old" in out


def test_git_branch_falls_through_on_empty_stdout(tmp_path):
    """git diff exits 0 but returns empty -> diff -ruN backup picks up the change."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    (base_repo / "kernel.py").write_text("old\n")
    (tmp_path / "kernel.py").write_text("new\n")
    tool = _make_tool(tmp_path, base_repo=base_repo)

    diff_ruN_patch = (
        f"diff -ruN {base_repo}/kernel.py {tmp_path}/kernel.py\n"
        f"--- {base_repo}/kernel.py\t2026-01-01 00:00:00 +0000\n"
        f"+++ {tmp_path}/kernel.py\t2026-01-01 00:00:01 +0000\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    side_effects = [
        _git_diff_result("", returncode=0),
        _diff_ruN_result(diff_ruN_patch, returncode=1),
    ]

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=side_effects,
        ) as run,
    ):
        out = tool._get_patch_content()

    assert run.call_count == 2
    assert "+new" in out and "-old" in out


def test_git_branch_empty_returns_empty_when_no_base_repo(tmp_path):
    """No base_repo_path -> can't fall through -> behaviour unchanged (empty string)."""
    tool = _make_tool(tmp_path, base_repo=None)

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            return_value=_git_diff_result("", returncode=0),
        ) as run,
    ):
        out = tool._get_patch_content()

    assert out == ""
    assert run.call_count == 1


def test_git_diff_excludes_3rdparty(tmp_path):
    """The git-branch exclude list must scope ``3rdparty/`` out of the diff."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    captured: dict[str, str] = {}

    def _capture(cmd, *args, **kwargs):  # noqa: ARG001 - mock signature
        captured["cmd"] = cmd
        return _git_diff_result("diff --git a/kernel.py b/kernel.py\n", returncode=0)

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=_capture,
        ),
    ):
        tool._get_patch_content()

    assert "':(exclude)3rdparty/'" in captured["cmd"], (
        f"git diff command must include ':(exclude)3rdparty/' pathspec; got: {captured['cmd']}"
    )


def test_diff_ruN_excludes_3rdparty(tmp_path):
    """The diff -ruN backup branch must pass ``--exclude=3rdparty``."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    captured_args: list = []

    def _capture(cmd, *args, **kwargs):  # noqa: ARG001 - mock signature
        captured_args.append(cmd)
        # Caller A (git branch) is shell=True, list-vs-string distinguishes them.
        if isinstance(cmd, str):
            return _git_diff_result("", returncode=0)  # force fall-through
        return _diff_ruN_result("", returncode=0)

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=_capture,
        ),
    ):
        tool._get_patch_content()

    # The second call is the diff-ruN list-form invocation.
    diff_ruN_cmd = next(c for c in captured_args if isinstance(c, list) and c[0] == "diff")
    assert "--exclude=3rdparty" in diff_ruN_cmd, f"diff -ruN must include --exclude=3rdparty; got: {diff_ruN_cmd}"
