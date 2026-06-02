# HIP — HarnessBuilder hints

Language-specific idioms for producing a universal-contract harness
from a user's HIP test file.

## Universal contract the harness must satisfy

Same as all languages: argparse with `--correctness`, `--benchmark`,
`--full-benchmark`, `--profile`; emit `GEAK_RESULT_LATENCY_MS` and
`GEAK_RESULT_SPEEDUP` markers on stdout.

## HIP-specific inputs

HIP kernels come in one of three common shapes:

1. **Pybind11 / torch-extension wrapper**. Python-callable after a
   `torch.utils.cpp_extension.load_inline` or `load(...)` call. The
   harness imports the compiled module and invokes the Python-level
   function.
2. **Standalone `make` + `./bench`**. The harness shells out to the
   compiled binary and parses its stdout.
3. **Raw `hipcc` + host-side launcher**. Same shape as (2) but
   compiled per-invocation.

The builder picks the shape based on evidence in the kernel file and
the user's test file:

- `torch.utils.cpp_extension` / `pybind11::module` visible anywhere in
  the kernel file -> shape 1.
- `Makefile` at `repo_root` + existing `./bench` binary -> shape 2.
- Raw `__global__ void ...` without a Python binding -> shape 3.

## Reference selection

- For shape 1 (pybind11 wrapper), the same wrapper exposes a
  reference implementation (usually a PyTorch fallback); use it.
- For shapes 2 and 3, the user test file contains either a
  CPU reference or a separate validation run; preserve that path.

## Timing loop

- Warmup 5 iterations; measure 100 and take the median (not mean).
- `hipDeviceSynchronize()` before/after each measurement.

## aiter kernels — MANDATORY worktree routing (do NOT use sys.path alone)

If the kernel lives in the **aiter** repo (`csrc/kernels/*.cu` invoked via a
Python-callable like `aiter.add_rmsnorm_quant(...)`, built by aiter's JIT
`@compile_ops` mechanism), the generic "prepend the worktree to `sys.path`"
rule is NOT enough and will silently evaluate the BASELINE kernel:

- `aiter` is installed **editable** via a `sys.meta_path` finder
  (`__editable__...amd_aiter...finder.py`). That finder is consulted BEFORE the
  `sys.path`-based finder, so `import aiter` resolves to the ORIGINAL repo
  (`/sgl-workspace/aiter`) no matter what you insert at `sys.path[0]`.
- aiter's JIT derives its source dir from `AITER_META_DIR` →
  `AITER_CSRC_DIR = f"{AITER_META_DIR}/csrc"` (see `aiter/jit/core.py`). If you
  don't override it, it points at the baseline tree, so patches to
  `$GEAK_WORK_DIR/csrc/...cu` are never compiled. Result: correctness always
  PASSes and every speedup is ~1.00x (the worktree-bypass bug).

You MUST route aiter's JIT to the worktree by setting these env vars at the very
top of the harness, BEFORE `import aiter` (this recipe is verified to make
sentinel-injected worktree corruption fail correctness, i.e. the worktree IS
evaluated):

```python
import os
WORK_DIR = os.path.abspath(os.environ.get("GEAK_WORK_DIR", "/sgl-workspace/aiter"))
# Fail loud if the kernel under test is not actually in the worktree.
_kernel_rel = "csrc/kernels/<this_kernel>.cu"
assert os.path.exists(os.path.join(WORK_DIR, _kernel_rel)), \
    f"kernel not found under WORK_DIR={WORK_DIR}"
# Route aiter's JIT source dir (AITER_CSRC_DIR = $AITER_META_DIR/csrc) to the worktree.
os.environ["AITER_META_DIR"] = WORK_DIR
# Per-slot JIT build cache so worktrees don't share artifacts.
_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
os.environ["AITER_JIT_DIR"] = os.path.join(WORK_DIR, f"_geak_jit_gpu{_gpu}")
# Force recompile ONLY when the worktree kernel source is newer than the cached
# .so (incremental); a patched candidate has a newer mtime so it still rebuilds.
_so = os.path.join(os.environ["AITER_JIT_DIR"], "<module>.so")
_src = os.path.join(WORK_DIR, _kernel_rel)
if (not os.path.exists(_so)) or os.path.getmtime(_src) > os.path.getmtime(_so):
    os.environ["AITER_REBUILD"] = "2"
else:
    os.environ.pop("AITER_REBUILD", None)
import aiter  # now resolves source/build under $GEAK_WORK_DIR
```

Do NOT rely on deleting the `.so` plus `sys.path.insert` — that does not change
where aiter's editable finder resolves `import aiter`, nor where `AITER_CSRC_DIR`
points. The `AITER_META_DIR` override is the only mechanism that works.
