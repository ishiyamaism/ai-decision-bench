"""Metric calculations shared by every provider."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from ai_decision_bench.models import (
    BenchmarkMetrics,
    CalibrationBucket,
    CaseEvaluation,
    LatencyMetrics,
)

_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.5, "0.0-0.5"),
    (0.5, 0.7, "0.5-0.7"),
    (0.7, 0.9, "0.7-0.9"),
    (0.9, 1.0, "0.9-1.0"),
)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _latency_metrics(evaluations: list[CaseEvaluation]) -> LatencyMetrics:
    values = [item.result.latency_ms for item in evaluations]
    if not values:
        return LatencyMetrics(
            mean_ms=None,
            median_ms=None,
            p95_ms=None,
            min_ms=None,
            max_ms=None,
        )
    return LatencyMetrics(
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p95_ms=_p95(values),
        min_ms=min(values),
        max_ms=max(values),
    )


def _in_bucket(probability: float, lower: float, upper: float) -> bool:
    return lower <= probability <= upper if upper == 1.0 else lower <= probability < upper


def _calibration_metrics(
    evaluations: list[CaseEvaluation],
) -> tuple[list[CalibrationBucket], int, float | None]:
    calibrated = [
        item
        for item in evaluations
        if item.result.error is None and item.result.prediction_probability is not None
    ]
    buckets: list[CalibrationBucket] = []
    weighted_error = 0.0
    for lower, upper, label in _BUCKETS:
        members = [
            item
            for item in calibrated
            if _in_bucket(item.result.prediction_probability or 0.0, lower, upper)
        ]
        if not members:
            buckets.append(
                CalibrationBucket(
                    range=label,
                    count=0,
                    average_prediction_probability=None,
                    accuracy=None,
                )
            )
            continue
        average_probability = statistics.fmean(
            item.result.prediction_probability or 0.0 for item in members
        )
        accuracy = sum(item.correct for item in members) / len(members)
        weighted_error += len(members) * abs(average_probability - accuracy)
        buckets.append(
            CalibrationBucket(
                range=label,
                count=len(members),
                average_prediction_probability=average_probability,
                accuracy=accuracy,
            )
        )
    ece = weighted_error / len(calibrated) if calibrated else None
    return buckets, len(calibrated), ece


def _optional_sum(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def calculate_metrics(evaluations: list[CaseEvaluation]) -> BenchmarkMetrics:
    """Calculate metrics; provider failures remain in the accuracy denominator."""

    total = len(evaluations)
    failures = sum(item.result.error is not None for item in evaluations)
    correct = sum(item.correct for item in evaluations)
    buckets, calibration_case_count, ece = _calibration_metrics(evaluations)
    return BenchmarkMetrics(
        total=total,
        successful=total - failures,
        correct=correct,
        failures=failures,
        accuracy=correct / total if total else 0.0,
        failure_rate=failures / total if total else 0.0,
        latency=_latency_metrics(evaluations),
        calibration_case_count=calibration_case_count,
        calibration_coverage=calibration_case_count / total if total else 0.0,
        calibration_buckets=buckets,
        expected_calibration_error=ece,
        total_reported_cost_usd=_optional_sum(
            item.result.reported_cost_usd for item in evaluations
        ),
        total_estimated_cost_usd=_optional_sum(
            item.result.estimated_cost_usd for item in evaluations
        ),
    )
