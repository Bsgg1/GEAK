"""Contract validators for harness + commandment artifacts.

Enforced by `preprocess/phases/*.py` (PR-2 lands these). Today this module
provides the validator API so other code can import and call — but the checks
are permissive stubs until PR-2 tightens them against the fixture corpus.

See docs/refactor/EXECUTION_PLAN.md §4 "Contract validators" + §16.7 (fixture
corpus spec).

The UNIVERSAL harness contract (what `HarnessBuilder` produces and
`validate_harness` enforces):

  harness.py MUST expose argparse with mutually-exclusive flags:
    --correctness        run correctness check, print OK/FAIL
    --benchmark          run in-loop timing, print GEAK_RESULT_LATENCY_MS=<float>
    --full-benchmark     run verification with iteration count, also print
                         GEAK_RESULT_SPEEDUP=<float>
    --profile            run with the language's profiler attached

  AND emit STDOUT markers:
    GEAK_RESULT_LATENCY_MS=<float>      on --benchmark
    GEAK_RESULT_SPEEDUP=<float>         on --full-benchmark

The UNIVERSAL commandment contract (what `validate_commandment` enforces):

  COMMANDMENT.md MUST contain these level-2 headers in order:
    ## Setup
    ## Correctness
    ## Benchmark
    ## Full Benchmark
    ## Profile

  Each section's fenced ``` block MUST parse as shell. Each command MUST
  reference the harness.py path consistent with preprocess/artifacts/harness.py.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


class ContractViolation(RuntimeError):
    """Raised when an artifact doesn't satisfy its contract."""


# ---------------------------------------------------------------------------
# Harness contract
# ---------------------------------------------------------------------------

REQUIRED_HARNESS_FLAGS = ("--correctness", "--benchmark", "--full-benchmark", "--profile")
REQUIRED_HARNESS_MARKERS = ("GEAK_RESULT_LATENCY_MS", "GEAK_RESULT_SPEEDUP")


# ---------------------------------------------------------------------------
# Worktree-bypass detection (the "always ~1.00x speedup" root cause)
# ---------------------------------------------------------------------------
#
# GEAK evaluates a candidate by copying the repo into a per-slot worktree,
# applying the patch there, exporting ``GEAK_WORK_DIR=<worktree>`` and putting
# it first on ``PYTHONPATH`` (see ``kernel_languages/*/commandment.j2``), then
# running ``harness.py``.  The contract is therefore: **the harness must resolve
# every repository path from ``GEAK_WORK_DIR``** (include dirs, ``#include``
# roots, build/output dirs, ``sys.path`` entries, files it opens).
#
# When an LLM-generated harness instead hardcodes an absolute path that points
# at the ORIGINAL source repo (``REPO_ROOT = "/sgl-workspace/sglang"``,
# ``hipcc -I /sgl-workspace/.../include``, ``sys.path.insert(0, "/abs")``), it
# silently compiles/imports the un-patched baseline.  Correctness then always
# PASSes and every measured speedup is ~1.00x, with no error — the optimizer
# trains on a flat signal.  This detector is intentionally *mechanism-agnostic*:
# it keys on "an absolute literal that points into the known repo root and is
# NOT derived from the GEAK_WORK_DIR/GEAK_REPO_ROOT env contract", so it covers
# C/C++ ``-I``, Python ``sys.path``, ``open()``, ``os.walk()``, build caches,
# etc. with a single rule rather than one regex per kernel language.

# Tokens whose presence on a line means the absolute path is the *sanctioned
# env-derived fallback default* (e.g. ``os.environ.get("GEAK_WORK_DIR", "/repo")``)
# rather than a hardcoded bypass.  Such lines are allowed.
_ENV_DERIVE_TOKENS = (
    "GEAK_WORK_DIR",
    "GEAK_REPO_ROOT",
    "os.environ",
    "environ",
    "getenv",
)

# Backstop patterns (used when the repo root is unknown at validation time):
# absolute-path literals used in the canonical worktree-bypass forms.
_ABS_LITERAL = r"/[^'\"\n]*"
_BYPASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "sys.path pinned to an absolute literal",
        re.compile(r"""sys\.path\.(?:insert|append|extend)\s*\([^)]*?['"](?P<path>/[^'"\n]+)['"]"""),
    ),
    (
        "compiler include (-I) pinned to an absolute literal",
        re.compile(r"""['"]?-I\s*(?P<path>/[^'"\n\s]+)"""),
    ),
    (
        "absolute repo/include path assigned to a module-level constant",
        re.compile(
            r"""(?im)^\s*(?:REPO_ROOT|REPO|SRC_ROOT|SOURCE_ROOT|KERNEL_ROOT|INCLUDE_DIR|INC_DIR|"""
            r"""[A-Z_]*INCLUDE[A-Z_]*|[A-Z_]*ROOT)\s*=\s*['"](?P<path>/[^'"\n]+)['"]"""
        ),
    ),
)

# System / toolchain prefixes that are legitimately referenced by absolute
# literal (ROCm/CUDA includes, std headers, scratch, device nodes). A bare
# absolute literal under one of these is NOT a source-repo bypass.
_SYSTEM_PATH_PREFIXES = (
    "/usr/", "/opt/rocm", "/opt/conda", "/opt/venv", "/opt/cuda", "/usr/local",
    "/lib/", "/lib64/", "/bin/", "/sbin/", "/etc/", "/proc/", "/sys/", "/dev/",
    "/tmp/", "/var/", "/run/", "/root/.cache", "/home/",
)

# A bare absolute-path *string literal* that appears as a collection element or
# RHS value (e.g. ``CANDIDATES = ["/repo/python"]`` then iterated into
# ``sys.path.insert(0, _p)``). The named-constant + sys.path-literal patterns
# above miss this because the path hides in a list and the insert uses a
# variable. We flag it when it points at a recognizable code/source tree — the
# contract forbids ANY hardcoded source path; the only sanctioned literal is the
# default arg of os.environ.get(...).
_BARE_ABS_LITERAL = re.compile(r"""(?P<q>['"])(?P<path>/[A-Za-z0-9._][^'"\n]*?)(?P=q)""")

# Source/package directory markers: an absolute literal whose path ends with (or
# contains) one of these is almost certainly an import/include root, not a data
# file or scratch path. Keeps backstop (3) precise (low false-positive).
_CODE_TREE_MARKERS = (
    "/python", "/src", "/csrc", "/include", "/lib", "/cpp", "/cuda", "/hip",
    "/kernels", "/srt",
)


def _strip_comment_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(lineno, line)`` for lines that are not pure ``#`` comments.

    ``lineno`` is 1-based.  We keep inline code with trailing comments (the
    code part can still hold a bypass) but drop full-comment lines so a
    documented counter-example doesn't trip the detector.
    """
    out: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        out.append((i, line))
    return out


def _scan_context(text: str) -> tuple[set[int], dict[int, bool]]:
    """Tokenize-based context so the line scanner avoids two false positives.

    Returns ``(docstring_lines, env_lines)``:

    * ``docstring_lines`` — physical line numbers that belong to a bare string
      *expression statement* (module/function docstrings, stray string blocks).
      A path inside one is documentation, not a real import/include path, so the
      scanner skips it.
    * ``env_lines`` — maps each physical line number to whether its enclosing
      *logical* line contains an env-derive token. This collapses multi-line
      calls so the literal default of e.g. ::

          WORK_DIR = os.environ.get(
              "GEAK_WORK_DIR",
              "/repo",          # <- env-derived, NOT a hardcoded bypass
          )

      is recognised as the sanctioned env fallback even though the env token is
      on a *previous* physical line.

    On any tokenization failure returns empty structures so the caller falls
    back to the (stricter) pure per-line behaviour.
    """
    import io
    import tokenize

    doc_lines: set[int] = set()
    env_lines: dict[int, bool] = {}
    physical = text.splitlines()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except Exception:  # noqa: BLE001 — best-effort; caller falls back
        return doc_lines, env_lines

    _skip = {
        tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
        tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }
    cur: list[tokenize.TokenInfo] = []

    def _flush(group: list[tokenize.TokenInfo]) -> None:
        sig = [t for t in group if t.type not in _skip]
        if not sig:
            return
        start = min(t.start[0] for t in sig)
        end = max(t.end[0] for t in sig)
        unit_text = "\n".join(physical[start - 1 : end])
        has_env = any(tok in unit_text for tok in _ENV_DERIVE_TOKENS)
        is_bare_string = len(sig) == 1 and sig[0].type == tokenize.STRING
        for ln in range(start, end + 1):
            if is_bare_string:
                doc_lines.add(ln)
            if has_env:
                env_lines[ln] = True

    for t in toks:
        if t.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
            continue
        cur.append(t)
        if t.type == tokenize.NEWLINE:
            _flush(cur)
            cur = []
    _flush(cur)
    return doc_lines, env_lines


def _resolve_repo_roots(repo_root: str | os.PathLike[str] | None) -> list[str]:
    """Collect candidate source-repo absolute prefixes to scan for.

    Order/source: explicit ``repo_root`` arg, then ``GEAK_REPO_ROOT`` and
    ``GEAK_WORK_DIR`` from the environment.  Both the literal and the
    fully-resolved (realpath) form are included so a symlinked mount still
    matches.  Returns a de-duplicated list of absolute path strings.
    """
    candidates: list[str] = []
    raw = [repo_root, os.environ.get("GEAK_REPO_ROOT"), os.environ.get("GEAK_WORK_DIR")]
    for r in raw:
        if not r:
            continue
        s = str(r).rstrip("/")
        if not s.startswith("/") or len(s) <= 1:
            continue
        candidates.append(s)
        try:
            real = str(Path(s).resolve()).rstrip("/")
            if real and real != s and real.startswith("/") and len(real) > 1:
                candidates.append(real)
        except Exception:  # noqa: BLE001 — resolve is best-effort
            pass
    # de-dupe preserving order
    seen: set[str] = set()
    roots: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            roots.append(c)
    return roots


#: Tokens that mean a path on this line feeds an import / compile / exec search
#: context (the only way a hardcoded source path actually causes the *baseline*
#: to be evaluated). If any of these is present the line is NOT provenance.
_PATH_SINK_TOKENS: tuple[str, ...] = (
    "sys.path", "pythonpath", "import ", "__import__", "open(", "-isystem",
    "load(", "load_inline", "cpp_extension", "subprocess", "popen", "exec(",
    "include_dirs", "extra_include",
)

#: Tokens that mean the path is being emitted as text (recorded to a file or
#: logged) rather than used as a filesystem search path.
_OUTPUT_CALL_TOKENS: tuple[str, ...] = (
    ".write(", "print(", ".info(", ".debug(", ".warning(", ".error(",
    ".exception(", "logging.", "logger.", "sys.stderr", "sys.stdout",
)


def _is_provenance_only(line: str) -> bool:
    """True when a source-repo path on this line is pure provenance/log output.

    A path written to a text file (``f.write("/repo/bench.py")``) or printed for
    audit never participates in import/compile resolution, so it cannot make the
    harness evaluate the unpatched baseline. Such a line is only provenance when
    it emits the path via a write/print/log call AND carries no import/compile
    sink token. Lines like ``sys.path.insert(0, "/repo")`` or ``-I/repo`` contain
    a sink token and are therefore never treated as provenance.
    """
    low = line.lower()
    if not any(tok in low for tok in _OUTPUT_CALL_TOKENS):
        return False
    return not any(tok in low for tok in _PATH_SINK_TOKENS)


def find_source_repo_path_leaks(
    text: str,
    repo_root: str | os.PathLike[str] | None = None,
) -> list[tuple[int, str, str]]:
    """Detect harness lines that bypass ``GEAK_WORK_DIR`` via a hardcoded path.

    Returns a list of ``(lineno, snippet, reason)`` for every offending line.
    A line is an offender when it either:

      * contains an absolute literal that begins with the known source-repo
        root (``repo_root`` arg or ``GEAK_REPO_ROOT``/``GEAK_WORK_DIR`` env), or
      * matches one of the canonical worktree-bypass forms
        (``sys.path.insert(0, "/abs")``, ``-I/abs``, ``REPO_ROOT = "/abs"``),

    AND the same line does NOT derive that path from the env contract (i.e. it
    does not mention ``GEAK_WORK_DIR`` / ``os.environ`` / ``getenv``).  The
    env-derived form, e.g. ``os.environ.get("GEAK_WORK_DIR", "/repo")``, is the
    sanctioned fallback and is intentionally allowed.

    Mechanism-agnostic by design: it keys on "absolute literal into the repo
    root, not env-derived", so no per-language special-casing is needed.
    """
    roots = _resolve_repo_roots(repo_root)
    findings: list[tuple[int, str, str]] = []
    seen_lines: set[int] = set()
    doc_lines, env_lines = _scan_context(text)

    for lineno, line in _strip_comment_lines(text):
        if lineno in seen_lines:
            continue
        # Skip documentation strings (paths there don't resolve imports).
        if lineno in doc_lines:
            continue
        # Skip pure provenance/log output: a source path written to a text file
        # or printed for audit never feeds an import/compile search path, so it
        # cannot cause the unpatched baseline to be evaluated. (Lines that also
        # contain an import/compile sink token are NOT exempted.)
        if _is_provenance_only(line):
            continue
        # Env-derived if the literal sits on (or its enclosing multi-line call
        # contains) an env-derive token.
        env_derived = env_lines.get(lineno, False) or any(tok in line for tok in _ENV_DERIVE_TOKENS)

        # (1) Prefix match against the known source-repo root(s).
        if roots and not env_derived:
            for root in roots:
                # Match the root as a quoted-or-flagged absolute literal so we
                # don't trip on unrelated substrings.
                if re.search(rf"""['"=:\s]{re.escape(root)}(?:/|['"\s)]|$)""", line) or (
                    root in line and "/" in line
                ):
                    findings.append(
                        (
                            lineno,
                            line.strip()[:200],
                            f"hardcoded absolute path into the source repo ('{root}') — "
                            "derive it from os.environ['GEAK_WORK_DIR'] instead",
                        )
                    )
                    seen_lines.add(lineno)
                    break
            if lineno in seen_lines:
                continue

        # (2) Backstop: canonical bypass forms regardless of repo_root.
        if not env_derived:
            for reason, pat in _BYPASS_PATTERNS:
                m = pat.search(line)
                if m:
                    findings.append((lineno, line.strip()[:200], reason))
                    seen_lines.add(lineno)
                    break
            if lineno in seen_lines:
                continue

        # (3) Backstop: a bare absolute-path string literal pointing at a
        # code/source tree (>=2 path segments, not a system prefix), not
        # env-derived. Catches a hardcoded source path hidden in a candidate
        # list/tuple that is later fed into sys.path / -I via a variable — the
        # exact shape that bypasses (1) (repo root unknown) and (2) (literal
        # not adjacent to sys.path/-I/REPO_ROOT=). Only fires when the file
        # actually manipulates import/include search paths, to avoid flagging
        # unrelated string constants (e.g. result file paths).
        if not env_derived and ("sys.path" in text or "-I" in text or "PYTHONPATH" in text):
            for m in _BARE_ABS_LITERAL.finditer(line):
                p = m.group("path").rstrip("/")
                if p.lower().startswith(_SYSTEM_PATH_PREFIXES):
                    continue
                # require a real directory tree (>=2 segments) to skip e.g. "/"
                if p.count("/") < 2:
                    continue
                # only a recognizable code/source tree (import/include root),
                # not a data/result file path
                if not any(mk in p.lower() for mk in _CODE_TREE_MARKERS):
                    continue
                findings.append(
                    (
                        lineno,
                        line.strip()[:200],
                        f"hardcoded absolute source path literal ('{p}') used in an "
                        "import/include search context — derive it from "
                        "os.environ['GEAK_WORK_DIR'] instead of pinning a fixed path",
                    )
                )
                seen_lines.add(lineno)
                break

    return findings


def _worktree_bypass_message(path: Path, leaks: list[tuple[int, str, str]]) -> str:
    listed = "\n".join(f"  - line {ln}: {reason}\n      {snippet}" for ln, snippet, reason in leaks)
    return (
        f"harness {path} bypasses the GEAK worktree with hardcoded absolute path(s):\n"
        f"{listed}\n\n"
        "Why this is fatal: GEAK applies the candidate patch inside a per-slot "
        "worktree exported as $GEAK_WORK_DIR (and put first on PYTHONPATH). A "
        "harness that hardcodes a source-repo absolute path compiles/imports the "
        "UNPATCHED baseline, so correctness always PASSes and every speedup reads "
        "~1.00x with no error.\n"
        "Fix (applies to every kernel language): resolve EVERY repository path "
        "from the worktree, e.g.\n"
        "    WORK_DIR = os.environ.get('GEAK_WORK_DIR', os.path.dirname(os.path.abspath(__file__)))\n"
        "then build include flags as f'-I{WORK_DIR}/<subdir>', put compiled "
        "artifacts under WORK_DIR in a DETERMINISTIC fixed-name build dir (it is "
        "already per-slot-isolated because WORK_DIR differs per worktree) and do an "
        "INCREMENTAL rebuild keyed on source mtime/hash (rebuild only when the "
        "kernel source is newer than the artifact, otherwise reuse it — never an "
        "unconditional cold rebuild every run), "
        "and rely on PYTHONPATH for Python imports. Never write a literal source "
        "path; if you keep one as a fallback, it must be the default of an "
        "os.environ.get('GEAK_WORK_DIR', ...) lookup."
    )


_AITER_IMPORT_RE = re.compile(r"^\s*(?:import\s+aiter\b|from\s+aiter\b)", re.MULTILINE)


def find_aiter_routing_violation(text: str) -> str | None:
    """Detect aiter harnesses that fail to route the JIT to ``$GEAK_WORK_DIR``.

    ``aiter`` is installed *editable* via a ``sys.meta_path`` finder, so
    ``import aiter`` resolves to the ORIGINAL repo regardless of ``sys.path``
    order. Two *independent* env vars must therefore be routed to the worktree
    (see ``aiter/jit/core.py``):

    * ``AITER_META_DIR`` controls the *source* dir
      (``AITER_CSRC_DIR = $AITER_META_DIR/csrc``). If it is not overridden the
      JIT compiles the BASELINE ``csrc/*.cu`` — the classic ~1.00x
      worktree-bypass bug (sentinel-injected corruption is never seen).
    * ``AITER_JIT_DIR`` controls the *build output* dir
      (``bd_dir = get_user_jit_dir()/build`` and the final ``<module>.so``).
      ``get_user_jit_dir()`` only honours ``$AITER_JIT_DIR``; otherwise it falls
      back to the **aiter package dir itself** (writable editable install), so
      build artifacts land in ``/sgl-workspace/aiter/aiter/jit/build`` —
      polluting the source repo AND, under the parallel ``run_pool`` path,
      making concurrent slots collide on a shared ``build/`` dir, ``<module>.so``
      and ninja ``lock_<module>`` file.

    A harness that imports aiter and compiles a ``csrc/*.cu`` kernel MUST set
    *both* ``AITER_META_DIR`` and a per-worktree ``AITER_JIT_DIR`` (derived from
    ``$GEAK_WORK_DIR``) before ``import aiter``. Returns a message describing the
    first missing piece, else ``None``.
    """
    if not _AITER_IMPORT_RE.search(text):
        return None
    # Only relevant when the harness actually drives aiter's source compile.
    compiles_csrc = ("csrc" in text) or (".cu" in text) or ("compile_ops" in text)
    if not compiles_csrc:
        return None
    if "AITER_META_DIR" not in text:
        return (
            "aiter harness imports `aiter` and compiles a csrc/*.cu kernel but never "
            "sets AITER_META_DIR from $GEAK_WORK_DIR. Because aiter is an editable "
            "package (sys.meta_path finder), `import aiter` + sys.path.insert resolves "
            "to the BASELINE repo, and aiter's JIT (AITER_CSRC_DIR=$AITER_META_DIR/csrc) "
            "compiles the baseline kernel — every speedup is ~1.00x and correctness is "
            "blind to the patch. Set os.environ['AITER_META_DIR'] = WORK_DIR (and a "
            "per-slot AITER_JIT_DIR) BEFORE `import aiter`."
        )
    if "AITER_JIT_DIR" not in text:
        return (
            "aiter harness sets AITER_META_DIR (source routing) but never sets "
            "AITER_JIT_DIR (build-output routing). aiter's get_user_jit_dir() only "
            "honours $AITER_JIT_DIR; without it the JIT writes build/ and <module>.so "
            "into the editable aiter package dir (/sgl-workspace/aiter/aiter/jit), "
            "which (a) pollutes the SOURCE repo and (b) makes parallel run_pool slots "
            "collide on a shared build dir / .so / ninja lock. Set a per-worktree "
            "os.environ['AITER_JIT_DIR'] = os.path.join(WORK_DIR, ...) BEFORE `import aiter`."
        )
    return None


def validate_harness(path: Path, repo_root: str | os.PathLike[str] | None = None) -> None:
    """Verify a harness.py conforms to the universal contract.

    Raises `ContractViolation` on any missing required surface. Today's checks
    are simple substring / regex presence — PR-2 tightens them against the
    fixture corpus (`tests/fixtures/harness_corpus/`).

    ``repo_root`` (when known by the caller) is forwarded to the worktree-bypass
    detector so a hardcoded absolute path into the *source* repo is caught even
    when ``GEAK_REPO_ROOT``/``GEAK_WORK_DIR`` are not set to that repo in the
    environment (e.g. during preprocess, where ``GEAK_WORK_DIR`` points at the
    subagent worktree, not the original source tree).
    """
    if not path.exists():
        raise ContractViolation(f"harness path does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    missing_flags = [f for f in REQUIRED_HARNESS_FLAGS if f not in text]
    missing_markers = [m for m in REQUIRED_HARNESS_MARKERS if m not in text]

    if missing_flags or missing_markers:
        # For PR-1: permissive — only raise if BOTH flags and markers are missing
        # (suggests the harness is totally non-compliant). A partial match is
        # likely just a legacy harness pre-PR-2; don't break those.
        if missing_flags and missing_markers:
            raise ContractViolation(
                f"harness {path} missing required flags {missing_flags} AND required markers {missing_markers}"
            )

    # Worktree-bypass gate: reject harnesses that hardcode an absolute path
    # into the source repo instead of deriving it from $GEAK_WORK_DIR. This is
    # the root cause of "always ~1.00x speedup" runs (baseline evaluated in
    # place of the patched worktree). Opt out with GEAK_ALLOW_HARDCODED_PATHS=1.
    if not os.environ.get("GEAK_ALLOW_HARDCODED_PATHS"):
        leaks = find_source_repo_path_leaks(text, repo_root=repo_root)
        if leaks:
            raise ContractViolation(_worktree_bypass_message(path, leaks))
        aiter_msg = find_aiter_routing_violation(text)
        if aiter_msg:
            raise ContractViolation(f"harness {path} {aiter_msg}")


# ---------------------------------------------------------------------------
# Commandment contract
# ---------------------------------------------------------------------------

REQUIRED_COMMANDMENT_SECTIONS = (
    r"^##\s+Setup\b",
    r"^##\s+Correctness\b",
    r"^##\s+Benchmark\b",
    r"^##\s+Full Benchmark\b",
    r"^##\s+Profile\b",
)


def validate_commandment(path: Path) -> None:
    """Verify a COMMANDMENT.md has the 5 required level-2 sections in order.

    Permissive today (WARN-level); tightens to FAIL once PR-2's Jinja templates
    and validators land.
    """
    if not path.exists():
        raise ContractViolation(f"commandment path does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    missing: list[str] = []
    for pat in REQUIRED_COMMANDMENT_SECTIONS:
        if not re.search(pat, text, re.MULTILINE):
            missing.append(pat)
    if missing:
        # PR-1: permissive — just warn; don't block migrations of legacy commandments.
        # PR-2 tightens.
        return


__all__ = [
    "ContractViolation",
    "validate_harness",
    "validate_commandment",
    "find_source_repo_path_leaks",
    "find_aiter_routing_violation",
    "REQUIRED_HARNESS_FLAGS",
    "REQUIRED_HARNESS_MARKERS",
    "REQUIRED_COMMANDMENT_SECTIONS",
]
