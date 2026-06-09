"""Tests for resolve_num_parallel default subagent count."""

from __future__ import annotations

import os

import pytest

from minisweagent.run.utils.parallel_helpers import resolve_num_parallel


def test_resolve_num_parallel_gpu_counts() -> None:
    assert resolve_num_parallel(0) == 4
    assert resolve_num_parallel(1) == 4
    assert resolve_num_parallel(2) == 6
    assert resolve_num_parallel(4) == 12


def test_resolve_num_parallel_explicit_knobs() -> None:
    assert resolve_num_parallel(4, min_workers=2, workers_per_gpu=2) == 8


def test_resolve_num_parallel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEAK_MIN_PARALLEL_WORKERS", "5")
    monkeypatch.setenv("GEAK_WORKERS_PER_GPU", "2")
    assert resolve_num_parallel(4) == 8


def test_resolve_num_parallel_empty_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEAK_MIN_PARALLEL_WORKERS", "")
    monkeypatch.setenv("GEAK_WORKERS_PER_GPU", "")
    assert resolve_num_parallel(4) == 12
