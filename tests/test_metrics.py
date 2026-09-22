from __future__ import annotations

import pytest

from ai_decision_bench.metrics import calculate_metrics
from ai_decision_bench.models import CaseEvaluation, DecisionResult


def _evaluation(
    case_id: str,
    *,
    correct: bool,
    latency: float,
    probability: float | None = None,
    provider_confidence: float | None = None,
    error: str | None = None,
    estimated_cost: float | None = None,
) -> CaseEvaluation:
    return CaseEvaluation(
        case_id=case_id,
        expected="a",
        correct=correct,
        result=DecisionResult(
            prediction="a" if correct else "b",
            prediction_probability=probability,
            probability_source="native_probability" if probability is not None else "unavailable",
            provider_confidence=provider_confidence,
            latency_ms=latency,
            estimated_cost_usd=estimated_cost,
            cost_source="estimated" if estimated_cost is not None else "unknown",
            error=error,
        ),
    )


def test_metrics_include_failures_in_accuracy_denominator() -> None:
    metrics = calculate_metrics(
        [
            _evaluation("1", correct=True, latency=10, probability=0.95),
            _evaluation("2", correct=True, latency=20, probability=0.8),
            _evaluation("3", correct=False, latency=30, probability=0.6),
            _evaluation("4", correct=False, latency=40, error="timeout", estimated_cost=0.002),
        ]
    )

    assert metrics.total == 4
    assert metrics.successful == 3
    assert metrics.correct == 2
    assert metrics.failures == 1
    assert metrics.accuracy == 0.5
    assert metrics.failure_rate == 0.25
    assert metrics.latency.mean_ms == 25
    assert metrics.latency.median_ms == 25
    assert metrics.latency.p95_ms == 40
    assert metrics.expected_calibration_error == pytest.approx((0.05 + 0.2 + 0.6) / 3)
    assert metrics.calibration_case_count == 3
    assert metrics.calibration_coverage == 0.75
    assert metrics.total_estimated_cost_usd == 0.002


def test_metrics_return_none_for_unavailable_probability_and_empty_cost() -> None:
    metrics = calculate_metrics([_evaluation("1", correct=True, latency=1)])

    assert metrics.expected_calibration_error is None
    assert metrics.total_estimated_cost_usd is None
    assert metrics.calibration_case_count == 0
    assert metrics.calibration_coverage == 0.0
    assert all(bucket.count == 0 for bucket in metrics.calibration_buckets)


def test_ece_uses_prediction_probability_not_provider_confidence() -> None:
    metrics = calculate_metrics(
        [
            _evaluation(
                "1",
                correct=True,
                latency=1,
                probability=0.8,
                provider_confidence=0.1,
            )
        ]
    )

    assert metrics.expected_calibration_error == pytest.approx(0.2)
