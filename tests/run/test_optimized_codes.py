"""Tests for the optimized_codes snapshot helper.

Exercises the end-to-end ``collect_optimized_codes`` flow against real
git repos created in tmp_path:

  * Plain modification              -> snapshotted, classified as ``modified``
  * New file (added)                -> snapshotted, classified as ``added``
  * Deleted file                    -> only recorded, not in directory
  * Rename                          -> new path snapshotted, recorded as renamed
  * Mixed multi-file patch          -> all snapshots correct
  * Missing / empty / bad patch     -> graceful skipped manifest
  * Path traversal in diff          -> ignored (defensive)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from minisweagent.run.postprocess.optimized_codes import (
    DEFAULT_TARGET_DIR_NAME,
    collect_optimized_codes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


def _init_git_repo(root: Path, initial_files: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "test@local"], root)
    _run(["git", "config", "user.name", "Test"], root)
    for rel, content in initial_files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-m", "initial"], root)


def _make_patch_from_changes(
    repo: Path,
    *,
    write: dict[str, str] | None = None,
    delete: list[str] | None = None,
    rename: dict[str, str] | None = None,
    patch_path: Path | None = None,
) -> Path:
    """Apply the given changes to `repo`, capture `git diff HEAD` as a
    patch file, then revert the repo so it's back at HEAD. Returns the
    written patch path.
    """
    write = write or {}
    delete = delete or []
    rename = rename or {}

    for rel, content in write.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    for rel in delete:
        (repo / rel).unlink()

    for old, new in rename.items():
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "mv", old, new], repo)

    _run(["git", "add", "-A"], repo)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--binary", "-M"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    if patch_path is None:
        patch_path = repo.parent / "candidate.patch"
    patch_path.write_text(diff.stdout)

    _run(["git", "reset", "--hard", "HEAD"], repo)
    return patch_path


# ---------------------------------------------------------------------------
# Happy-path: each kind of change
# ---------------------------------------------------------------------------


class TestCollectOptimizedCodes:
    def test_modified_file_snapshots_post_patch_content(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"aiter/ops/foo.py": "def foo(): return 1\n"})
        patch = _make_patch_from_changes(
            repo, write={"aiter/ops/foo.py": "def foo(): return 2\n"}
        )
        out = tmp_path / "out"
        out.mkdir()

        manifest = collect_optimized_codes(repo, patch, out)

        assert manifest["status"] == "complete"
        assert manifest["modified"] == ["aiter/ops/foo.py"]
        assert manifest["added"] == []
        assert manifest["deleted"] == []
        assert manifest["renamed"] == []
        assert manifest["files"] == ["aiter/ops/foo.py"]

        snapshot = out / DEFAULT_TARGET_DIR_NAME / "aiter" / "ops" / "foo.py"
        assert snapshot.is_file()
        assert snapshot.read_text() == "def foo(): return 2\n"
        assert manifest["directory"] == str((out / DEFAULT_TARGET_DIR_NAME).resolve())

    def test_added_file_appears_in_snapshot(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"keep.py": "keep\n"})
        patch = _make_patch_from_changes(repo, write={"new/added.py": "added\n"})
        out = tmp_path / "out"
        out.mkdir()

        manifest = collect_optimized_codes(repo, patch, out)

        assert manifest["status"] == "complete"
        assert manifest["added"] == ["new/added.py"]
        snapshot = out / DEFAULT_TARGET_DIR_NAME / "new" / "added.py"
        assert snapshot.is_file() and snapshot.read_text() == "added\n"

    def test_deleted_file_recorded_but_not_snapshotted(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"obsolete.py": "old\n", "keep.py": "keep\n"})
        patch = _make_patch_from_changes(repo, delete=["obsolete.py"])
        out = tmp_path / "out"
        out.mkdir()

        manifest = collect_optimized_codes(repo, patch, out)

        assert manifest["status"] == "complete"
        assert manifest["deleted"] == ["obsolete.py"]
        assert manifest["files"] == []
        assert not (out / DEFAULT_TARGET_DIR_NAME / "obsolete.py").exists()

    def test_rename_snapshots_destination(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        content = "x" * 200
        _init_git_repo(repo, {"old/path.py": content})
        patch = _make_patch_from_changes(repo, rename={"old/path.py": "new/path.py"})
        out = tmp_path / "out"
        out.mkdir()

        manifest = collect_optimized_codes(repo, patch, out)

        assert manifest["status"] == "complete"
        if manifest["renamed"]:
            assert manifest["renamed"][0]["to"] == "new/path.py"
            assert manifest["renamed"][0]["from"] == "old/path.py"
        else:
            assert "new/path.py" in manifest["added"]
            assert "old/path.py" in manifest["deleted"]
        snapshot = out / DEFAULT_TARGET_DIR_NAME / "new" / "path.py"
        assert snapshot.is_file() and snapshot.read_text() == content

    def test_multi_file_mixed_change(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(
            repo,
            {
                "aiter/ops/a.py": "a v1\n",
                "aiter/ops/b.py": "b v1\n",
                "trash.py": "delete me\n",
            },
        )
        patch = _make_patch_from_changes(
            repo,
            write={
                "aiter/ops/a.py": "a v2\n",
                "aiter/new/c.py": "c new\n",
            },
            delete=["trash.py"],
        )
        out = tmp_path / "out"
        out.mkdir()

        manifest = collect_optimized_codes(repo, patch, out)

        assert manifest["status"] == "complete"
        assert manifest["added"] == ["aiter/new/c.py"]
        assert manifest["modified"] == ["aiter/ops/a.py"]
        assert manifest["deleted"] == ["trash.py"]
        assert manifest["files"] == ["aiter/new/c.py", "aiter/ops/a.py"]

        snap_root = out / DEFAULT_TARGET_DIR_NAME
        assert (snap_root / "aiter" / "ops" / "a.py").read_text() == "a v2\n"
        assert (snap_root / "aiter" / "new" / "c.py").read_text() == "c new\n"
        assert not (snap_root / "trash.py").exists()
        assert not (snap_root / "aiter" / "ops" / "b.py").exists()
        assert manifest["total_bytes"] > 0

    def test_total_bytes_sums_added_and_modified(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"a.py": "x\n"})
        patch = _make_patch_from_changes(
            repo, write={"a.py": "longer content here\n", "b.py": "another file\n"}
        )
        out = tmp_path / "out"
        out.mkdir()

        manifest = collect_optimized_codes(repo, patch, out)

        snap_root = out / DEFAULT_TARGET_DIR_NAME
        expected = (snap_root / "a.py").stat().st_size + (snap_root / "b.py").stat().st_size
        assert manifest["total_bytes"] == expected

    def test_snapshot_directory_is_replaced_on_rerun(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"a.py": "a v1\n"})
        out = tmp_path / "out"
        out.mkdir()

        junk = out / DEFAULT_TARGET_DIR_NAME / "stale" / "junk.txt"
        junk.parent.mkdir(parents=True)
        junk.write_text("stale")

        patch = _make_patch_from_changes(repo, write={"a.py": "a v2\n"})
        manifest = collect_optimized_codes(repo, patch, out)
        assert manifest["status"] == "complete"
        assert not junk.exists()
        assert (out / DEFAULT_TARGET_DIR_NAME / "a.py").read_text() == "a v2\n"


class TestSkippedCases:
    def test_none_patch_returns_skipped(self, tmp_path: Path) -> None:
        result = collect_optimized_codes(tmp_path, None, tmp_path)
        assert result == {"status": "skipped", "reason": "no_best_patch"}

    def test_missing_patch_returns_skipped(self, tmp_path: Path) -> None:
        result = collect_optimized_codes(tmp_path, tmp_path / "nope.patch", tmp_path)
        assert result["status"] == "skipped"
        assert result["reason"] == "patch_file_missing"

    def test_empty_patch_returns_skipped(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.patch"
        empty.write_text("")
        result = collect_optimized_codes(tmp_path, empty, tmp_path)
        assert result["status"] == "skipped"
        assert result["reason"] == "empty_patch"

    def test_missing_repo_returns_skipped(self, tmp_path: Path) -> None:
        patch = tmp_path / "p.patch"
        patch.write_text("garbage\n")
        result = collect_optimized_codes(tmp_path / "no_such_repo", patch, tmp_path)
        assert result["status"] == "skipped"
        assert result["reason"] == "repo_root_missing"

    def test_unappliable_patch_returns_skipped(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"a.py": "real content\n"})

        patch = tmp_path / "bad.patch"
        patch.write_text(
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-this line does not exist in the file\n"
            "+replacement\n"
        )
        result = collect_optimized_codes(repo, patch, tmp_path / "out")
        assert result["status"] == "skipped"
        assert result["reason"] == "patch_apply_failed"

    def test_returns_skipped_on_cleanup_failure_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"a.py": "v1\n"})
        patch = _make_patch_from_changes(repo, write={"a.py": "v2\n"})
        out = tmp_path / "out"
        out.mkdir()

        from minisweagent.run.postprocess import optimized_codes as mod

        def _boom(*_a, **_kw):
            raise RuntimeError("synthetic cleanup error")

        monkeypatch.setattr(mod, "cleanup_eval_worktree", _boom)

        manifest = collect_optimized_codes(repo, patch, out)
        assert manifest["status"] == "complete"
        assert manifest["modified"] == ["a.py"]


class TestDefensive:
    def test_path_traversal_entry_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"a.py": "v1\n"})
        patch = _make_patch_from_changes(repo, write={"a.py": "v2\n"})
        out = tmp_path / "out"
        out.mkdir()

        from minisweagent.run.postprocess import optimized_codes as mod

        original = mod._git_diff_name_status

        def _poisoned(eval_dir):  # type: ignore[no-untyped-def]
            entries = original(eval_dir)
            entries.append(("M", ["../escape.txt"]))
            entries.append(("M", ["/absolute/leak.txt"]))
            return entries

        monkeypatch.setattr(mod, "_git_diff_name_status", _poisoned)

        manifest = collect_optimized_codes(repo, patch, out)
        assert manifest["status"] == "complete"
        assert (out / DEFAULT_TARGET_DIR_NAME / "a.py").is_file()
        assert not (tmp_path / "escape.txt").exists()
        assert not Path("/absolute/leak.txt").exists()
        files_outside = [f for f in manifest["files"] if f.startswith("..") or os.path.isabs(f)]
        assert files_outside == []


class TestJitCacheFilter:
    def test_jit_pkls_dropped_from_manifest_and_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"aiter/ops/foo.py": "v1\n"})
        patch = _make_patch_from_changes(repo, write={"aiter/ops/foo.py": "v2\n"})
        out = tmp_path / "out"
        out.mkdir()

        from minisweagent.run.postprocess import optimized_codes as mod

        original = mod._git_diff_name_status

        def _inject_jit_pkls(eval_dir):  # type: ignore[no-untyped-def]
            entries = original(eval_dir)
            for i in range(50):
                rel = (
                    f"aiter/jit/flydsl_cache/launch_gemm_{i:03d}/"
                    f"hash_{i:03d}.pkl"
                )
                entries.append(("A", [rel]))
            entries.append(("A", ["pkg/.triton/cache/blob"]))
            entries.append(("A", ["pkg/torch_compile_cache/0/out.py"]))
            entries.append(("M", ["aiter/jit/build/foo.o"]))
            return entries

        monkeypatch.setattr(mod, "_git_diff_name_status", _inject_jit_pkls)

        manifest = collect_optimized_codes(repo, patch, out)

        assert manifest["status"] == "complete"
        assert manifest["added"] == []
        assert manifest["modified"] == ["aiter/ops/foo.py"]
        assert manifest["files"] == ["aiter/ops/foo.py"]

        dropped = manifest.get("filtered_jit_cache", [])
        assert len(dropped) == 53

        snap_root = out / DEFAULT_TARGET_DIR_NAME
        assert (snap_root / "aiter" / "ops" / "foo.py").is_file()
        assert not (snap_root / "aiter" / "jit").exists()
        assert not (snap_root / "pkg").exists()

    def test_clean_patch_has_empty_filtered_jit_cache(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"a.py": "v1\n"})
        patch = _make_patch_from_changes(repo, write={"a.py": "v2\n"})
        out = tmp_path / "out"
        out.mkdir()

        manifest = collect_optimized_codes(repo, patch, out)

        assert manifest["status"] == "complete"
        assert manifest["filtered_jit_cache"] == []
        assert manifest["modified"] == ["a.py"]

    def test_renamed_jit_path_dropped_with_both_sides_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"a.py": "v1\n"})
        patch = _make_patch_from_changes(repo, write={"a.py": "v2\n"})
        out = tmp_path / "out"
        out.mkdir()

        from minisweagent.run.postprocess import optimized_codes as mod

        original = mod._git_diff_name_status

        def _inject_renamed_jit(eval_dir):  # type: ignore[no-untyped-def]
            entries = original(eval_dir)
            entries.append(
                (
                    "R100",
                    [
                        "aiter/jit/flydsl_cache/old/a.pkl",
                        "aiter/jit/flydsl_cache/new/a.pkl",
                    ],
                )
            )
            return entries

        monkeypatch.setattr(mod, "_git_diff_name_status", _inject_renamed_jit)

        manifest = collect_optimized_codes(repo, patch, out)
        assert manifest["status"] == "complete"
        assert manifest["renamed"] == []
        dropped = set(manifest["filtered_jit_cache"])
        assert "aiter/jit/flydsl_cache/old/a.pkl" in dropped
        assert "aiter/jit/flydsl_cache/new/a.pkl" in dropped
