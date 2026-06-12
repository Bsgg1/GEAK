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

The fix: the git branch falls through to a backup directory-diff path when
``git diff`` returns non-zero or empty stdout, instead of returning empty
unconditionally. The backup uses ``git diff --no-index <base> <cwd>`` (git's
own directory-tree diff). Unlike the previous plain ``diff -ruN``, it emits
``--- /dev/null`` + ``new file mode`` for files the agent created and derives
the executable bit from ``stat``, so a later ``git apply`` recreates new files
with the correct mode. The only post-processing is stripping the absolute
base/cwd prefixes from the ``a/`` ``b/`` headers.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

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


def _no_index_result(stdout: str, returncode: int = 1) -> subprocess.CompletedProcess:
    """Build a ``CompletedProcess`` shaped like ``git diff --no-index``'s output.

    ``git diff --no-index`` exits 1 when the trees differ (like plain diff),
    so the default return code here is 1.
    """
    return subprocess.CompletedProcess(
        args=["git", "diff", "--no-index"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def _no_index_modified(base_repo: Path, cwd: Path) -> str:
    """``git diff --no-index`` output for a modified ``kernel.py`` (absolute headers)."""
    base_abs = str(base_repo.resolve()).lstrip("/")
    cwd_abs = str(cwd.resolve()).lstrip("/")
    return (
        f"diff --git a/{base_abs}/kernel.py b/{cwd_abs}/kernel.py\n"
        "index 3367afd..3e75765 100644\n"
        f"--- a/{base_abs}/kernel.py\n"
        f"+++ b/{cwd_abs}/kernel.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )


def _no_index_new_file(cwd: Path, rel: str, mode: str = "100644", body: str = "import os\n") -> str:
    """``git diff --no-index`` output for a file present only in ``cwd``.

    For a created file git uses the cwd-side path on BOTH ``a/`` and ``b/``
    sides and emits ``--- /dev/null`` + ``new file mode``.
    """
    cwd_abs = str(cwd.resolve()).lstrip("/")
    lines = body.splitlines()
    hunk_body = "".join(f"+{ln}\n" for ln in lines)
    return (
        f"diff --git a/{cwd_abs}/{rel} b/{cwd_abs}/{rel}\n"
        f"new file mode {mode}\n"
        "index 0000000..21b405d\n"
        "--- /dev/null\n"
        f"+++ b/{cwd_abs}/{rel}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{hunk_body}"
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
    """Submodule-missing case: git diff fails -> no-index backup runs."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    (base_repo / "kernel.py").write_text("old\n")
    (tmp_path / "kernel.py").write_text("new\n")
    tool = _make_tool(tmp_path, base_repo=base_repo)

    side_effects = [
        _git_diff_result("", returncode=128),  # git aborted on missing submodule .git
        _no_index_result(_no_index_modified(base_repo, tmp_path)),
    ]

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=side_effects,
        ) as run,
    ):
        out = tool._get_patch_content()

    assert run.call_count == 2, "fall-through must invoke the no-index backup branch"
    # The output should be the normalised (relative) git-style patch.
    assert out.startswith("diff --git a/kernel.py b/kernel.py"), (
        f"expected normalised git-style header, got: {out[:120]!r}"
    )
    assert "+new" in out and "-old" in out
    # Absolute prefixes must be gone.
    assert str(base_repo.resolve()).lstrip("/") not in out
    assert str(tmp_path.resolve()).lstrip("/") not in out


def test_git_branch_falls_through_on_empty_stdout(tmp_path):
    """git diff exits 0 but returns empty -> no-index backup picks up the change."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    (base_repo / "kernel.py").write_text("old\n")
    (tmp_path / "kernel.py").write_text("new\n")
    tool = _make_tool(tmp_path, base_repo=base_repo)

    side_effects = [
        _git_diff_result("", returncode=0),
        _no_index_result(_no_index_modified(base_repo, tmp_path)),
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


def test_no_index_new_file_becomes_create_section(tmp_path):
    """A file present only in cwd must normalise to a proper create section.

    This pins the core bug fix: ``git diff --no-index`` emits
    ``--- /dev/null`` + ``new file mode`` for created files, and the
    normaliser only strips the absolute prefix, so the result is a valid
    ``git apply`` create. The previous ``diff -ruN`` path mislabelled new
    files as modifications of a non-existent base file and apply failed.
    """
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    side_effects = [
        _git_diff_result("", returncode=0),
        _no_index_result(_no_index_new_file(tmp_path, "newfile.py")),
    ]

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=side_effects,
        ),
    ):
        out = tool._get_patch_content()

    assert "diff --git a/newfile.py b/newfile.py" in out
    assert "new file mode 100644" in out
    assert "--- /dev/null" in out
    assert "+++ b/newfile.py" in out
    # Must NOT look like a modification of a (non-existent) base file.
    assert "--- a/newfile.py" not in out


def test_no_index_new_executable_file_preserves_mode(tmp_path):
    """The stat-derived executable bit (100755) must survive normalisation."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    # NB: use ``bench.sh`` not ``run.sh`` — the latter is a GEAK-generated harness
    # name that the exclude filter deliberately strips (see the generated-helper
    # excludes), which would defeat the executable-bit assertion below.
    side_effects = [
        _git_diff_result("", returncode=0),
        _no_index_result(_no_index_new_file(tmp_path, "bench.sh", mode="100755", body="#!/bin/sh\necho hi\n")),
    ]

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=side_effects,
        ),
    ):
        out = tool._get_patch_content()

    assert "diff --git a/bench.sh b/bench.sh" in out
    assert "new file mode 100755" in out, "executable bit must be preserved"
    assert "--- /dev/null" in out


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


def test_git_branch_success_path_still_strips_jit_cache(tmp_path):
    """Composition guard: the fall-through fix and the JIT-cache strip must coexist.

    The successful git-branch return path must still apply
    ``_strip_jit_cache_from_patch`` — otherwise we silently regress the
    "no JIT pkls in final_report.optimized_codes" behaviour.
    """
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    real_section = "diff --git a/kernel.py b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n"
    # Use a JIT-cache section the stripper is guaranteed to recognise, so this
    # test does not encode a private path convention from generated_artifacts.py.
    jit_section = (
        "diff --git a/flydsl_cache/abc.pkl b/flydsl_cache/abc.pkl\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "Binary files /dev/null and b/flydsl_cache/abc.pkl differ\n"
    )

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            return_value=_git_diff_result(real_section + jit_section, returncode=0),
        ),
    ):
        out = tool._get_patch_content()

    assert "kernel.py" in out, "real kernel.py edit must survive"
    assert "flydsl_cache" not in out, (
        f"JIT-cache section must be stripped from the git-branch successful return; got: {out!r}"
    )


def test_no_index_command_carries_no_pathspec(tmp_path):
    """The ``git diff --no-index`` command must NOT pass pathspec excludes.

    ``git diff --no-index`` rejects any pathspec on git < ~2.45 (errors with a
    usage message and returns an empty patch), which silently drops the agent's
    whole change set on e.g. git 2.34 (Ubuntu 22.04 CI). The command must be the
    bare two-path form; exclusion is handled by post-filtering the rendered patch
    (see :func:`test_no_index_strips_jit_cache_sections`).
    """
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    captured_args: list = []

    def _capture(cmd, *args, **kwargs):  # noqa: ARG001
        captured_args.append(cmd)
        if isinstance(cmd, str):
            return _git_diff_result("", returncode=0)
        return _no_index_result("", returncode=0)

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=_capture,
        ),
    ):
        tool._get_patch_content()

    no_index_cmd = next(c for c in captured_args if isinstance(c, list) and c[:3] == ["git", "diff", "--no-index"])
    assert "--" not in no_index_cmd, f"no-index command must not carry a pathspec separator; got: {no_index_cmd}"
    assert not any(str(a).startswith(":(exclude)") for a in no_index_cmd), (
        f"no-index command must not carry :(exclude) pathspecs; got: {no_index_cmd}"
    )


def test_no_index_strips_jit_cache_sections(tmp_path):
    """JIT-cache sections must be stripped from the rendered no-index patch.

    Exclusion moved from unsupported ``--no-index`` pathspecs to a post-render
    filter; this pins that a JIT-cache file present only in ``cwd`` is dropped
    while a real source edit survives.
    """
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _make_tool(tmp_path, base_repo=base_repo)

    patch = _no_index_modified(base_repo, tmp_path) + _no_index_new_file(tmp_path, "flydsl_cache/abc.pkl")
    side_effects = [
        _git_diff_result("", returncode=0),
        _no_index_result(patch),
    ]

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch(
            "minisweagent.tools.save_and_test.subprocess.run",
            side_effect=side_effects,
        ),
    ):
        out = tool._get_patch_content()

    assert "kernel.py" in out, "real kernel.py edit must survive"
    assert "flydsl_cache" not in out, f"JIT-cache section must be stripped from the no-index patch; got: {out!r}"


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def test_real_round_trip_new_file_applies_with_correct_mode(tmp_path):
    """End-to-end proof: capture via the real fall-through, then ``git apply``.

    Builds a base dir + a cwd dir on disk (cwd is NOT a git repo, forcing the
    no-index fall-through), runs the real ``_get_patch_content``, then applies
    the captured patch into a throwaway repo and asserts the new files are
    created — the executable one with its bit preserved (0o755), the regular
    one without (0o644). This is the regression the previous ``diff -ruN``
    path failed on (``newfile.py: No such file or directory``).
    """
    if not os.access("/usr/bin/git", os.X_OK) and subprocess.run(["git", "--version"], capture_output=True).returncode:
        pytest.skip("git not available")

    base_repo = tmp_path / "base"
    cwd = tmp_path / "cwd"
    base_repo.mkdir()
    cwd.mkdir()
    (base_repo / "kernel.py").write_text("old\n")
    (cwd / "kernel.py").write_text("new\n")
    (cwd / "newfile.py").write_text("import os\n")
    # An executable the agent might legitimately create. NB: do not use
    # ``run.sh`` etc. here — those are GEAK-generated harness names that
    # ``_strip_jit_cache_from_patch``'s sibling stripper deliberately removes.
    script = cwd / "bench.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)

    tool = _make_tool(cwd, base_repo=base_repo)
    # cwd is not a git repo, so the primary branch is skipped and the
    # no-index fall-through produces the patch.
    patch = tool._get_patch_content()

    assert "new file mode 100755" in patch, f"executable create section missing: {patch!r}"
    assert "new file mode 100644" in patch
    assert "--- a/newfile.py" not in patch, "new file must use /dev/null, not a modification header"

    # Apply into a fresh repo seeded with the baseline kernel.py.
    target = tmp_path / "target"
    target.mkdir()
    _git(["init", "-q"], target)
    _git(["config", "user.email", "t@t"], target)
    _git(["config", "user.name", "t"], target)
    (target / "kernel.py").write_text("old\n")
    _git(["add", "."], target)
    _git(["commit", "-qm", "base"], target)

    apply_res = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=str(target),
        input=patch,
        capture_output=True,
        text=True,
    )
    assert apply_res.returncode == 0, f"git apply failed: {apply_res.stderr}"

    assert (target / "newfile.py").read_text() == "import os\n"
    assert (target / "kernel.py").read_text() == "new\n"
    assert (target / "bench.sh").exists()
    assert os.stat(target / "bench.sh").st_mode & 0o111, "executable bit must survive apply"
    assert not (os.stat(target / "newfile.py").st_mode & 0o111), "regular file must not be executable"
