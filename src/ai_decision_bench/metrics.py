"""Metric calculations shared by every provider."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from ai_decision_bench.models import (
    BenchmarkMetrics,
    CaseEvaluation,
    ConfidenceBucket,
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


def _in_bucket(confidence: float, lower: float, upper: float) -> bool:
    return lower <= confidence <= upper if upper == 1.0 else lower <= confidence < upper


def _confidence_metrics(
    evaluations: list[CaseEvaluation],
) -> tuple[list[ConfidenceBucket], float | None]:
    calibrated = [
        item
        for item in evaluations
        if item.result.error is None and item.result.confidence is not None
    ]
    buckets: list[ConfidenceBucket] = []
    weighted_error = 0.0
    for lower, upper, label in _BUCKETS:
        members = [
            item for item in calibrated if _in_bucket(item.result.confidence or 0.0, lower, upper)
        ]
        if not members:
            buckets.append(
                ConfidenceBucket(
                    range=label,
                    count=0,
                    average_confidence=None,
                    accuracy=None,
                )
            )
            continue
        average_confidence = statistics.fmean(item.result.confidence or 0.0 for item in members)
        accuracy = sum(item.correct for item in members) / len(members)
        weighted_error += len(members) * abs(average_confidence - accuracy)
        buckets.append(
            ConfidenceBucket(
                range=label,
                count=len(members),
                average_confidence=average_confidence,
                accuracy=accuracy,
            )
        )
    ece = weighted_error / len(calibrated) if calibrated else None
    return buckets, ece


def _optional_sum(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def calculate_metrics(evaluations: list[CaseEvaluation]) -> BenchmarkMetrics:
    """Calculate metrics; provider failures remain in the accuracy denominator."""

    total = len(evaluations)
    failures = sum(item.result.error is not None for item in evaluations)
    correct = sum(item.correct for item in evaluations)
    buckets, ece = _confidence_metrics(evaluations)
    return BenchmarkMetrics(
        total=total,
        successful=total - failures,
        correct=correct,
        failures=failures,
        accuracy=correct / total if total else 0.0,
        failure_rate=failures / total if total else 0.0,
        latency=_latency_metrics(evaluations),
        confidence_buckets=buckets,
        expected_calibration_error=ece,
        total_reported_cost_usd=_optional_sum(
            item.result.reported_cost_usd for item in evaluations
        ),
        total_estimated_cost_usd=_optional_sum(
            item.result.estimated_cost_usd for item in evaluations
        ),
    )
