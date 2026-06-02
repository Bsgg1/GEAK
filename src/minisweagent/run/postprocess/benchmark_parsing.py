"""Backward-compat shim — canonical module lives in preprocess.

All benchmark parsing logic is maintained in
``minisweagent.run.preprocess.benchmark_parsing``.  This stub re-exports
every public name so that existing ``from minisweagent.run.postprocess.benchmark_parsing
import ...`` statements continue to work unchanged.
"""

from minisweagent.run.preprocess.benchmark_parsing import (  # noqa: F401
    BenchmarkMetric,
    _labeled_latencies_geomean_ms,
    _metric_to_ms,
    compute_best_patch,
    compute_shape_speedups,
    compute_speedup,
    extract_benchmark_config_lines,
    extract_benchmark_metric,
    extract_latency_ms,
    extract_reported_speedup,
    parse_google_benchmark_ms,
    parse_labeled_latencies_ms,
    parse_median_latency_ms,
    parse_shape_count,
    parse_shape_latencies_ms,
    parse_total_kernel_time_ms,
    rewrite_best_results,
)
