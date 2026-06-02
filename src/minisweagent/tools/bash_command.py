import locale
import logging
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

# Default per-command wall-clock timeout (seconds). A find/grep over a huge
# shared mount (e.g. /wekafs, tens of TB) would otherwise block until the
# preprocess soft/hard cap fires, burning the whole preprocess budget. Override
# with GEAK_BASH_TIMEOUT_S.
_DEFAULT_BASH_TIMEOUT_S = 300.0

# Recursive-traversal commands whose explicit path operands are range-checked
# against the denylist below. Non-recursive commands (and any command we cannot
# confidently parse) are left untouched and rely on the timeout backstop.
_RECURSIVE_SCANNERS = frozenset({"find", "grep", "egrep", "fgrep", "rg", "ag", "fd", "fdfind", "ls", "du", "tree"})
_GREP_LIKE = frozenset({"grep", "egrep", "fgrep"})

# Scanners whose FIRST non-flag operand is a pattern/regex, not a path, so it
# must be skipped before range-checking the remaining (path) operands.
_PATTERN_FIRST = frozenset({"grep", "egrep", "fgrep", "rg", "ag", "fd", "fdfind"})

# Search roots that must never be scanned recursively: the multi-TB shared data
# lake and the bare filesystem root (which contains it). Override/extend with a
# ':'-separated GEAK_SEARCH_DENY_ROOTS (a bare "/" means "exactly /", not a
# prefix, so it does not block legitimate /opt/rocm or /usr/include searches).
_DEFAULT_DENY_ROOTS = ("/wekafs", "/")

# Shell tokens that begin a new simple-command within a compound line, so each
# sub-command's argv[0] can be re-evaluated against the scanner set.
_CMD_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "{", "}", "\n"})

_OUTPUT_UNREADABLE = (
    "The combined command output could not be decoded as a whole using the "
    "process locale encoding. Part of the command (e.g. one stage such as "
    '"cat" of a binary file) may have produced invalid or non-text bytes, so '
    "none of the captured stdout is shown. Run text-producing steps separately "
    "or use a tool suited for binary data."
)

logger = logging.getLogger(__name__)


def _process_stream_encoding() -> str:
    try:
        return locale.getencoding()
    except AttributeError:
        return locale.getpreferredencoding(False) or "utf-8"


def _decode_captured_output(stdout_b: bytes | None, stderr_b: bytes | None) -> str:
    """Decode subprocess bytes with the locale encoding and strict errors.

    If the chosen stream is non-empty but not valid for that encoding, return
    ``_OUTPUT_UNREADABLE`` instead of partial or replacement-character output.
    """
    enc = _process_stream_encoding()
    out = (stdout_b or b"").strip()
    err = (stderr_b or b"").strip()
    if out:
        try:
            return out.decode(enc, "strict")
        except UnicodeDecodeError:
            return _OUTPUT_UNREADABLE
    if err:
        try:
            return err.decode(enc, "strict")
        except UnicodeDecodeError:
            return _OUTPUT_UNREADABLE
    return ""


# Matches shell redirect / heredoc patterns that write to COMMANDMENT.md,
# e.g. ``cat > path/COMMANDMENT.md``, ``tee path/COMMANDMENT.md``,
# ``> path/COMMANDMENT.md << 'EOF'``.
_COMMANDMENT_WRITE_RE = re.compile(
    r"""(?:cat\s+>|>\s*|tee\s+)"""
    r"""\s*([^\s<|&]+COMMANDMENT\.md)"""
    r"""|"""
    r"""(?:>\s*|\s+)([^\s<|&]+COMMANDMENT\.md)\s*<<""",
    re.VERBOSE,
)


_BASH_TIMEOUT_S = int(os.environ.get("GEAK_BASH_TIMEOUT", "300"))


class BashCommand:
    def __init__(self):
        self._env_override: dict[str, str] = {}
        self._cwd: str | None = None
        self.timeout: int = _BASH_TIMEOUT_S
        self.blocklist: list[str] = [
            "vim",
            "vi",
            "emacs",
            "nano",
            "nohup",
            "gdb",
            "less",
            "tail -f",
            "python -m venv",
            "make",
        ]
        self.blocklist_standalone: list[str] = [
            "python",
            "python3",
            "ipython",
            "bash",
            "sh",
            "/bin/bash",
            "/bin/sh",
            "nohup",
            "vi",
            "vim",
            "emacs",
            "nano",
            "su",
            "reboot",
            "shutdown",
            "mkfs",
            "rm -rf /",
        ]

    @staticmethod
    def _sandbox_command(command: str) -> str:
        """Rewrite absolute paths in the command that target the original repo.

        Agents in worktrees must never write to the original repo
        (``GEAK_REPO_ROOT``).  Replace occurrences of the repo root with
        the agent's worktree (``GEAK_WORK_DIR``) so that ``cat >``,
        ``cp``, ``cd``, and similar commands land in the worktree.

        Safe because every legitimate repo-root reference in agent bash
        commands (``cd``, ``python -c``, ``cp``) works identically with
        the worktree path. The PYTHONPATH and COMMANDMENT ``run.sh``
        scripts read ``$GEAK_REPO_ROOT`` at shell-expansion time, not
        from the command string, so they are unaffected.
        """
        repo_root = os.environ.get("GEAK_REPO_ROOT", "")
        work_dir = os.environ.get("GEAK_WORK_DIR", "")
        if not repo_root or not work_dir or repo_root == work_dir:
            return command
        if repo_root in command:
            rewritten = command.replace(repo_root, work_dir)
            logger.debug("bash_command: rewrote repo_root paths in command")
            return rewritten
        return command

    @staticmethod
    def _denied_roots() -> list[str]:
        raw = os.environ.get("GEAK_SEARCH_DENY_ROOTS")
        if raw is None:
            return list(_DEFAULT_DENY_ROOTS)
        return [d for d in raw.split(":") if d]

    @staticmethod
    def _path_in_deny(token: str, deny: list[str]) -> str | None:
        """Resolve an absolute/`~` path token and return the matched deny root,
        or None. A deny entry of "/" matches only the exact filesystem root, so
        bounded system trees (``/opt/rocm``, ``/usr/include``) stay searchable."""
        if not token.startswith(("/", "~")):
            return None  # relative -> resolves under cwd (worktree); safe.
        rp = os.path.realpath(Path(token).expanduser())
        for d in deny:
            if d == "/":
                if rp == "/":
                    return rp
            elif rp == d.rstrip("/") or rp.startswith(d.rstrip("/") + "/"):
                return rp
        return None

    @staticmethod
    def _grep_is_recursive(flags: list[str]) -> bool:
        for f in flags:
            if f in ("--recursive", "--dereference-recursive"):
                return True
            if f.startswith("-") and not f.startswith("--") and ("r" in f[1:] or "R" in f[1:]):
                return True
        return False

    @staticmethod
    def _ls_is_recursive(flags: list[str]) -> bool:
        # ls: only -R / --recursive walks; -r is merely reverse-sort.
        for f in flags:
            if f == "--recursive":
                return True
            if f.startswith("-") and not f.startswith("--") and "R" in f[1:]:
                return True
        return False

    @classmethod
    def _blocked_scan_root(cls, command: str) -> str | None:
        """Return the offending deny-root if ``command`` recursively scans it.

        Conservative & fail-open: any command we cannot confidently tokenize
        (exotic quoting, ``$(...)``, etc.) returns None and falls back to the
        timeout backstop. Only the recursive scanners in ``_RECURSIVE_SCANNERS``
        are inspected, and grep/ls only when a recursion flag is present (a
        single-file ``grep`` or a flat ``ls`` is bounded). Everything else
        passes through untouched.
        """
        deny = cls._denied_roots()
        if not deny:
            return None
        try:
            tokens = shlex.split(command, comments=False)
        except ValueError:
            return None  # unbalanced quotes etc. -> let timeout handle it.

        i = 0
        at_cmd_start = True
        while i < len(tokens):
            tok = tokens[i]
            if tok in _CMD_SEPARATORS:
                at_cmd_start = True
                i += 1
                continue
            if not at_cmd_start:
                i += 1
                continue
            at_cmd_start = False
            cmd = Path(tok).name
            if cmd not in _RECURSIVE_SCANNERS:
                i += 1
                continue
            # Collect this simple-command's operands up to the next separator.
            j = i + 1
            operands: list[str] = []
            while j < len(tokens) and tokens[j] not in _CMD_SEPARATORS:
                operands.append(tokens[j])
                j += 1
            flags = [o for o in operands if o.startswith("-")]
            # grep/ls only walk when their recursion flag is present.
            if cmd in _GREP_LIKE and not cls._grep_is_recursive(flags):
                i = j
                continue
            if cmd == "ls" and not cls._ls_is_recursive(flags):
                i = j
                continue
            # Pattern-first commands take a regex as their first non-flag
            # operand (e.g. ``rg "/x" /path``); skip it so it is not mistaken
            # for a path. find/du/tree/ls are path-first -> check all operands.
            skipped_pattern = cmd not in _PATTERN_FIRST
            for operand in operands:
                if operand.startswith("-"):
                    continue
                if not skipped_pattern:
                    skipped_pattern = True
                    continue
                hit = cls._path_in_deny(operand, deny)
                if hit:
                    return hit
            i = j
        return None

    @staticmethod
    def _run(command: str, env, cwd, timeout_s: float):
        """Run ``command`` in its own process group and enforce ``timeout_s``.

        ``start_new_session=True`` puts the shell and all its children in a
        fresh process group, so on timeout we ``killpg`` the WHOLE group —
        otherwise a runaway ``find`` orphaned by killing only the shell would
        keep hammering the filesystem. Returns ``(stdout, stderr, rc, timed_out)``.
        """
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            out_b, err_b = proc.communicate(timeout=timeout_s)
            return out_b, err_b, proc.returncode, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                out_b, err_b = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out_b, err_b = b"", b""
            return out_b, err_b, -signal.SIGKILL, True

    def __call__(
        self,
        *,
        command: str,
        **kwargs,
    ):
        if not command:
            return {
                "output": "bash tool call need a command argument, it must not be empty.",
                "returncode": 1,
            }
        if any(f.startswith(command) for f in self.blocklist) or command in self.blocklist_standalone:
            return {
                "output": f"Blocked dangerous command: {command}",
                "returncode": 1,
            }
        denied_root = self._blocked_scan_root(command)
        if denied_root:
            return {
                "output": (
                    f"Blocked recursive scan of '{denied_root}': it is a multi-TB "
                    "shared mount outside the allowed search scope and would stall "
                    "for a very long time. Restrict the search to your worktree "
                    "($GEAK_WORK_DIR) or the source repo ($GEAK_REPO_ROOT), e.g. "
                    '`grep -r <pattern> "$GEAK_WORK_DIR"`.'
                ),
                "returncode": 1,
            }
        else:
            command = self._sandbox_command(command)
            env = os.environ | self._env_override if self._env_override else None
            cwd = self._cwd if self._cwd and Path(self._cwd).is_dir() else None
            try:
                # GEAK_BASH_TIMEOUT_S takes precedence; fall back to self.timeout
                # (which itself reads main's GEAK_BASH_TIMEOUT) so both env names work.
                timeout_s = float(os.environ.get("GEAK_BASH_TIMEOUT_S", self.timeout))
            except (TypeError, ValueError):
                timeout_s = float(self.timeout)
            stdout_b, stderr_b, returncode, timed_out = self._run(command, env, cwd, timeout_s)
            output_text = _decode_captured_output(stdout_b, stderr_b)

            if timed_out:
                notice = (
                    f"Command terminated after exceeding the {timeout_s:.0f}s timeout "
                    "(process group killed). If you were searching the filesystem, "
                    "scope it to $GEAK_WORK_DIR / $GEAK_REPO_ROOT instead of a shared "
                    "mount; otherwise split the work into smaller steps."
                )
                output_text = f"{output_text}\n\n{notice}" if output_text else notice

            if "COMMANDMENT.md" in command:
                output_text = self._maybe_validate_commandment(command, output_text)

            if returncode != 0:
                output_text = output_text or "Command failed with no output."
            return {
                "output": output_text or "Bash command executed successfully.",
                "returncode": returncode,
            }

    @staticmethod
    def _maybe_validate_commandment(command: str, output_text: str) -> str:
        """Validate COMMANDMENT.md if the bash command wrote one.

        COMMANDMENT.md is the evaluation contract between sub-agents and the
        orchestrator.  Sub-agents must not silently produce an invalid one, so
        every bash command that touches the file is validated on the spot and
        any errors are appended to the command output as immediate feedback.
        """
        path_str: str | None = None

        m = _COMMANDMENT_WRITE_RE.search(command)
        if m:
            path_str = m.group(1) or m.group(2)
        else:
            for token in command.split():
                if token.endswith("COMMANDMENT.md") and "/" in token:
                    path_str = token
                    break

        if path_str:
            p = Path(path_str)
            if p.exists():
                try:
                    from minisweagent.tools.validate_commandment import (  # pylint: disable=no-name-in-module
                        format_validation_message,
                        validate_commandment,
                    )

                    result = validate_commandment(p.read_text())
                    msg = format_validation_message(result)
                    if msg:
                        output_text += f"\n\n{msg}"
                except Exception:
                    logger.debug("COMMANDMENT validation failed", exc_info=True)

        return output_text
