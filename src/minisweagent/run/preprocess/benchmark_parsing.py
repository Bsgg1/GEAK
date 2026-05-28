"""Deterministic benchmark output parsing and patch selection.

Provides regex-based extraction of benchmark metrics from harness output
and a ``compute_best_patch()`` function that selects the best non-empty
patch by comparing benchmark numbers -- no LLM involved.

Supports two metric marker formats:

* **New (generalized)**: ``GEAK_RESULT_METRIC=<float>``,
  ``GEAK_RESULT_UNIT=<unit>``, ``GEAK_RESULT_DIRECTION=<lower_is_better|higher_is_better>``
* **Legacy**: ``GEAK_RESULT_LATENCY_MS=<float>`` (implies ms, lower_is_better)

Falls back to heuristic parsers when no explicit marker is present.

Measurement methodology:
- Uses ``benchmark_baseline.txt`` (the canonical unmodified baseline benchmark)
- Only reports speedups > 1.0 (genuine improvements over true baseline)
- Clamps LLM-inflated results to 1.0 when no real improvement exists
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_median_latency_ms(output: str) -> float | None:
    """Extract median latency (ms) from harness benchmark output."""
    m = re.search(
        r"(?:[Mm]edian\s+(?:latency|time)[\w\s]*|total\s+median\s+time)\s*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
        output,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def parse_total_kernel_time_ms(output: str) -> float | None:
    """Extract TOTAL_KERNEL_TIME_MS or BENCHMARK_LATENCY_MS from harness benchmark output."""
    m = re.search(
        r"(?:TOTAL_KERNEL_TIME_MS|BENCHMARK_LATENCY_MS):\s*([\d.]+(?:e[+-]?\d+)?)",
        output,
    )
    return float(m.group(1)) if m else None


def _parse_benchmark_metric(output: str) -> float | None:
    """Extract from BENCHMARK_METRIC:, median_latency_ms:, or Geomean (ms): lines."""
    for pat in (
        r"BENCHMARK_METRIC:\s*median_latency_ms=([\d.]+(?:e[+-]?\d+)?)",
        r"median_latency_ms:\s*([\d.]+(?:e[+-]?\d+)?)",
        r"Geomean\s*\(ms\)\s*:\s*([\d.]+(?:e[+-]?\d+)?)",
    ):
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_google_benchmark_ms(output: str) -> float | None:
    """Parse Google Benchmark format: <name> <iters> <latency> ms."""
    m = re.search(r"^\S+\s+\d+\s+([\d.]+(?:e[+-]?\d+)?)\s+ms", output, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_labeled_latencies_ms(output: str) -> dict[str, float]:
    """Extract per-entry latencies from labeled benchmark output.

    Matches lines like::

        Perf: 0.0122 ms (shape_0_forward)
        Latency: 0.0342 ms (config_name)
        Time: 1.234 ms

    Returns ``{label: latency_ms}`` where *label* comes from the
    parenthesized suffix or a generated index when absent.
    """
    results: dict[str, float] = {}
    idx = 0
    for m in re.finditer(
        r"^\s*\w[\w\s]*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms\s*(?:\(([^)]+)\))?\s*$",
        output,
        re.MULTILINE,
    ):
        val = float(m.group(1))
        label = m.group(2) or f"entry_{idx}"
        if 0.0001 < val < 1e6:
            results[label] = val
        idx += 1
    return results


def _labeled_latencies_geomean_ms(output: str) -> float | None:
    """Geometric mean of labeled latency lines, or ``None`` if none found."""
    entries = parse_labeled_latencies_ms(output)
    if not entries:
        return None
    import math

    vals = list(entries.values())
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def parse_shape_count(output: str) -> int | None:
    """Extract shape count from harness benchmark output."""
    m = re.search(r"(\d+)\s+shapes", output, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_shape_latencies_ms(output: str) -> dict[str, float]:
    """Extract per-shape latencies from harness benchmark output.

    Supports two formats:
        ``(32,4096): 0.0503 ms``  (Triton harness)
        ``Perf: 0.0122 ms (shape_0_forward)``  (labeled, e.g. HIP task_runner)
    """
    shape_latencies: dict[str, float] = {}
    for m in re.finditer(r"^\s*(\([^)]*\)):\s*([\d.]+(?:e[+-]?\d+)?)\s*ms\s*$", output, re.MULTILINE):
        shape_latencies[m.group(1)] = float(m.group(2))
    if shape_latencies:
        return shape_latencies
    return parse_labeled_latencies_ms(output)


def extract_benchmark_config_lines(output: str) -> list[str]:
    """Extract benchmark config fingerprint lines from harness output.

    Captures the config/shape identifiers from each benchmark line,
    stripping timing numbers so only the problem description remains.
    This allows comparing whether baseline and candidate ran on the
    same benchmark configurations, regardless of kernel language or
    variable naming conventions.

    Works by finding lines that contain timing data (e.g. '0.0342ms')
    and extracting the config prefix before the first timing number.

    Examples of lines matched:
        'B=1 H=32 NQ=16 N_CTX=[512] ...  2.11ms   0.10ms  21.37x *'
        '(1, 16), k=2       0.0196ms   0.0335ms     0.58x'
        'Config (B=256,H=1024)   0.072ms  ...'

    Returns a sorted list of config identifiers (empty list if none found).
    """
    configs: list[str] = []
    timing_pattern = re.compile(r"\d+\.\d+\s*(?:ms|us|µs|s|x)")
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith(("-", "=", "#", "Status", "Geometric", "GEAK_")):
            continue
        if not timing_pattern.search(line):
            continue
        if any(kw in line.lower() for kw in ("comparing", "running", "warmup", "median", "geomean", "mean")):
            continue
        config_part = re.split(r"(?<=[=:])\s*\d+\.\d+|\s+\d+\.\d+", line)[0].strip()
        config_part = re.sub(r"[\s:|]+$", "", config_part)
        config_part = re.sub(r"\s*\|\s*\w+$", "", config_part)
        config_part = re.sub(r":\s*\w+=$", "", config_part)
        if config_part:
            configs.append(config_part)
    return sorted(configs)


def _universal_latency_fallback(text: str) -> float | None:
    """Last-resort: find a number near latency-related keywords in the last
    30 lines of output. Handles formats like 'Overall Median: 0.052ms'."""
    keywords = {"median", "overall", "geomean", "latency", "total"}
    candidates: list[float] = []
    lines = text.strip().splitlines()
    for line in lines[-30:]:
        lower = line.lower()
        if not any(kw in lower for kw in keywords):
            continue
        for m in re.finditer(r"([\d.]+(?:e[+-]?\d+)?)\s*ms", line):
            val = float(m.group(1))
            if 0.0001 < val < 100000:
                candidates.append(val)
    return candidates[-1] if candidates else None


@dataclass(frozen=True)
class BenchmarkMetric:
    """Structured benchmark metric extracted from harness output.

    Attributes:
        value: Raw numeric value in the original unit.
        unit: Unit string (``"ms"``, ``"us"``, ``"GB/s"``, ``"TFLOPS"``, …).
        direction: ``"lower_is_better"`` for latency/time metrics,
            ``"higher_is_better"`` for throughput/bandwidth metrics.
    """

    value: float
    unit: str
    direction: str  # "lower_is_better" | "higher_is_better"


_HIGHER_IS_BETTER_UNITS = frozenset({
    "gb/s", "tb/s", "mb/s",
    "tflops", "gflops", "pflops",
    "items/s", "ops/s", "samples/s",
})

_TIME_UNIT_TO_MS: dict[str, float] = {
    "ms": 1.0,
    "us": 0.001,
    "µs": 0.001,
    "ns": 0.000001,
    "s": 1000.0,
}


def _metric_to_ms(metric: BenchmarkMetric) -> float:
    """Convert a :class:`BenchmarkMetric` value to milliseconds.

    For time-based units this is a direct conversion.  For throughput
    units (higher-is-better) we return a synthetic inverse
    (``1000 / value``) so that legacy callers comparing in ms-space
    still get directionally correct results.
    """
    factor = _TIME_UNIT_TO_MS.get(metric.unit.lower())
    if factor is not None:
        return metric.value * factor
    if metric.value > 0:
        return 1000.0 / metric.value
    return 0.0


def extract_benchmark_metric(text: str) -> BenchmarkMetric | None:
    """Extract a structured benchmark metric from harness output.

    Priority:
    1. New generalized markers: ``GEAK_RESULT_METRIC=``, ``GEAK_RESULT_UNIT=``,
       ``GEAK_RESULT_DIRECTION=``
    2. Legacy ``GEAK_RESULT_LATENCY_MS=`` (implies ``ms``, ``lower_is_better``)
    3. Heuristic legacy parsers (all assume ``ms``, ``lower_is_better``)
    """
    m_val = re.search(r"GEAK_RESULT_METRIC=([\d.]+(?:e[+-]?\d+)?)", text)
    if m_val:
        value = float(m_val.group(1))
        m_unit = re.search(r"GEAK_RESULT_UNIT=(\S+)", text)
        m_dir = re.search(r"GEAK_RESULT_DIRECTION=(lower_is_better|higher_is_better)", text)
        unit = m_unit.group(1) if m_unit else "ms"
        direction = m_dir.group(1) if m_dir else (
            "higher_is_better" if unit.lower() in _HIGHER_IS_BETTER_UNITS else "lower_is_better"
        )
        return BenchmarkMetric(value=value, unit=unit, direction=direction)

    m_legacy = re.search(r"GEAK_RESULT_LATENCY_MS=([\d.]+(?:e[+-]?\d+)?)", text)
    if m_legacy:
        return BenchmarkMetric(value=float(m_legacy.group(1)), unit="ms", direction="lower_is_better")

    for parser in (
        parse_total_kernel_time_ms,
        _parse_benchmark_metric,
        parse_median_latency_ms,
        parse_google_benchmark_ms,
        _labeled_latencies_geomean_ms,
        _universal_latency_fallback,
    ):
        val = parser(text)
        if val is not None:
            return BenchmarkMetric(value=val, unit="ms", direction="lower_is_better")

    return None


def extract_latency_ms(text: str) -> float | None:
    """Extract a latency value in milliseconds from benchmark output.

    Backward-compatible wrapper around :func:`extract_benchmark_metric`.
    For throughput metrics the value is converted to a synthetic
    ms-equivalent via :func:`_metric_to_ms`.
    """
    metric = extract_benchmark_metric(text)
    if metric is None:
        return None
    return _metric_to_ms(metric)


def compute_speedup(
    baseline: float,
    candidate: float,
    direction: str = "lower_is_better",
) -> float:
    """Compute speedup ratio respecting metric direction.

    Returns a value > 1.0 when the candidate is better than the baseline,
    regardless of whether the metric is lower-is-better or higher-is-better.
    """
    if direction == "higher_is_better":
        return candidate / baseline
    return baseline / candidate


def extract_reported_speedup(text: str) -> float | None:
    """Extract a reported speedup scalar from benchmark output.

    Supported markers include:
    - ``GEAK_RESULT_GEOMEAN_SPEEDUP=<number>``
    - ``GEAK_RESULT_SPEEDUP=<number>``
    - ``Geometric mean speedup: <number>x``
    - ``Speedup (geomean): <number>x``
    """

    for pat in (
        r"GEAK_RESULT_GEOMEAN_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)",
        r"GEAK_RESULT_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)",
        r"Geometric mean speedup:\s*([\d.]+(?:e[+-]?\d+)?)x",
        r"Speedup\s*\(geomean\)\s*:\s*([\d.]+(?:e[+-]?\d+)?)x",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def compute_shape_speedups(
    baseline_shapes_ms: dict[str, float],
    candidate_shapes_ms: dict[str, float],
    direction: str = "lower_is_better",
) -> dict[str, dict[str, float]]:
    """Compute per-shape speedups for the overlap between baseline and candidate."""
    results: dict[str, dict[str, float]] = {}
    for shape, baseline_ms in baseline_shapes_ms.items():
        candidate_ms = candidate_shapes_ms.get(shape)
        if candidate_ms is None or baseline_ms <= 0 or candidate_ms <= 0:
            continue
        results[shape] = {
            "baseline_ms": round(baseline_ms, 6),
            "candidate_ms": round(candidate_ms, 6),
            "speedup": round(compute_speedup(baseline_ms, candidate_ms, direction), 6),
        }
    return results


def _find_original_baseline(patch_dir: Path) -> tuple[float, str] | None:
    """Walk up from patch_dir to find benchmark_baseline.txt (the canonical baseline).

    The preprocessing phase writes benchmark_baseline.txt at the kernel
    output root (e.g. patches/exp0/rope/benchmark_baseline.txt).  Task dirs
    are nested under results/round_N/strategy_name, so we walk upward.

    Returns ``(latency_ms, direction)`` or ``None``.
    """
    d = patch_dir
    for _ in range(8):
        bl = d / "benchmark_baseline.txt"
        if bl.is_file():
            text = bl.read_text()
            metric = extract_benchmark_metric(text)
            if metric is not None and metric.value > 0:
                return _metric_to_ms(metric), metric.direction
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def compute_best_patch(patch_dir: Path) -> dict[str, Any] | None:
    """Deterministically select the best non-empty patch from a task directory.

    Uses ``benchmark_baseline.txt`` as the canonical (unmodified) baseline rather
    than ``patch_0_test.txt`` which is the agent's first attempt.  Only
    returns a result if a patch genuinely beats the true baseline (>1.0x).
    """
    original_bl = _find_original_baseline(patch_dir)
    direction = "lower_is_better"

    baseline_file = patch_dir / "patch_0_test.txt"
    baseline_text = ""
    baseline_shape_latencies: dict[str, float] = {}
    if original_bl is not None:
        baseline_ms, direction = original_bl
        baseline_source = "benchmark_baseline.txt"
        baseline_file_path = next(
            (p for p in [patch_dir, *patch_dir.parents] if (p / "benchmark_baseline.txt").is_file()), None
        )
        if baseline_file_path is not None:
            baseline_text = (baseline_file_path / "benchmark_baseline.txt").read_text()
            baseline_shape_latencies = parse_shape_latencies_ms(baseline_text)
    elif baseline_file.exists():
        baseline_text = baseline_file.read_text()
        metric = extract_benchmark_metric(baseline_text)
        if metric is not None:
            baseline_ms = _metric_to_ms(metric)
            direction = metric.direction
        else:
            baseline_ms = None
        baseline_source = "patch_0_test.txt (FALLBACK)"
        baseline_shape_latencies = parse_shape_latencies_ms(baseline_text)
    else:
        return None

    if baseline_ms is None or baseline_ms <= 0:
        return None

    best_speedup = 0.0
    best_candidate_ms: float | None = None
    best_patch_id: str | None = None
    best_patch_file: str | None = None
    best_test_file: str | None = None
    best_patch_size: int = 0
    best_shape_speedups: dict[str, dict[str, float]] = {}
    best_candidate_shape_latencies: dict[str, float] = {}

    for test_file in sorted(patch_dir.glob("patch_*_test.txt")):
        name = test_file.stem.replace("_test", "")

        patch_file = patch_dir / f"{name}.patch"
        if not patch_file.exists():
            continue
        psz = patch_file.stat().st_size
        if psz == 0:
            continue

        candidate_text = test_file.read_text()
        candidate_ms = extract_latency_ms(candidate_text)
        if candidate_ms is None or candidate_ms <= 0:
            continue
        candidate_shape_latencies = parse_shape_latencies_ms(candidate_text)

        speedup = compute_speedup(baseline_ms, candidate_ms, direction)
        if speedup > best_speedup:
            best_speedup = speedup
            best_candidate_ms = candidate_ms
            best_patch_id = name
            best_patch_file = str(patch_file)
            best_test_file = str(test_file)
            best_patch_size = psz
            best_candidate_shape_latencies = candidate_shape_latencies
            best_shape_speedups = compute_shape_speedups(
                baseline_shape_latencies, candidate_shape_latencies, direction
            )

    if best_patch_id is None or best_speedup <= 1.0:
        return None

    return {
        "best_patch_id": best_patch_id,
        "best_patch_speedup": round(best_speedup, 6),
        "best_patch_file": best_patch_file,
        "best_patch_test_output": best_test_file,
        "best_patch_size_bytes": best_patch_size,
        "baseline_latency_ms": round(baseline_ms, 6),
        "candidate_latency_ms": round(best_candidate_ms, 6),
        "baseline_source": baseline_source,
        "metric_direction": direction,
        "baseline_shape_latency_ms": baseline_shape_latencies,
        "candidate_shape_latency_ms": best_candidate_shape_latencies,
        "per_shape_speedups": best_shape_speedups,
        "llm_selection_analysis": (
            f"Deterministic: baseline={baseline_ms:.4f}ms ({baseline_source}), "
            f"candidate={best_candidate_ms:.4f}ms from {best_patch_id}. "
            f"Speedup={best_speedup:.4f}x. Patch={best_patch_size}B."
        ),
    }


def rewrite_best_results(patch_dir: Path) -> dict[str, Any] | None:
    """Overwrite ``best_results.json`` with deterministic selection if possible.

    Uses the canonical baseline from benchmark_baseline.txt.  If no patch
    genuinely improves on the true baseline, clamps any LLM-reported
    speedup to 1.0x to prevent false positives.
    """
    det = compute_best_patch(patch_dir)
    existing_path = patch_dir / "best_results.json"
    original_bl_result = _find_original_baseline(patch_dir)
    original_bl = original_bl_result[0] if original_bl_result else None

    if det is not None:
        existing_path.write_text(json.dumps(det, indent=2))
        logger.info(
            "Deterministic best_results for %s: %s (%.4fx)",
            patch_dir.name,
            det["best_patch_id"],
            det["best_patch_speedup"],
        )
        return det

    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
            pf = existing.get("best_patch_file")

            if pf and Path(pf).exists() and Path(pf).stat().st_size == 0:
                existing["best_patch_speedup"] = 1.0
                existing["llm_selection_analysis"] = (
                    existing.get("llm_selection_analysis") or ""
                ) + " [Overridden: patch is empty (0 bytes), speedup clamped to 1.0]"
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            if original_bl is not None:
                existing["best_patch_speedup"] = 1.0
                existing["baseline_latency_ms"] = original_bl
                existing["baseline_source"] = "benchmark_baseline.txt"
                existing["llm_selection_analysis"] = (
                    existing.get("llm_selection_analysis") or ""
                ) + f" [Clamped: no patch beat true baseline {original_bl:.4f}ms]"
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            return existing
        except (json.JSONDecodeError, ValueError):
            pass

    return None
