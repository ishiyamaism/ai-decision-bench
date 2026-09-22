from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

from ai_decision_bench.cli import _ProgressReporter, render_summary
from ai_decision_bench.evaluator import run_benchmark
from ai_decision_bench.models import BenchmarkConfig, ClassificationTask, EvaluationCase
from ai_decision_bench.providers.mock import MockProvider


def test_mock_provider_is_deterministic() -> None:
    case = EvaluationCase(
        id="case-1",
        input={"text": "The app crashes with an error when I upload a file."},
        task=ClassificationTask(
            type="classification",
            labels=["billing", "technical_support", "other"],
        ),
        expected="technical_support",
    )
    provider = MockProvider()

    first = asyncio.run(provider.decide(case))
    second = asyncio.run(provider.decide(case))

    assert first.prediction == second.prediction == "technical_support"
    assert first.confidence == second.confidence
    assert first.error is None


def test_benchmark_reports_incremental_progress() -> None:
    updates: list[tuple[int, int]] = []

    report = asyncio.run(
        run_benchmark(
            MockProvider(),
            Path("datasets/sample.jsonl"),
            BenchmarkConfig(concurrency=3, timeout_seconds=1, max_retries=0),
            progress_callback=lambda completed, total: updates.append((completed, total)),
        )
    )

    assert updates[0] == (0, 24)
    assert updates[-1] == (24, 24)
    assert [completed for completed, _ in updates] == list(range(25))
    assert report.metrics.total == 24

    summary = render_summary([report])
    assert "Cases" in summary
    assert "ECE" in summary
    assert "Failures" in summary
    assert "Cost (USD)" in summary
    assert "keyword-baseline-v1" in summary


def test_non_interactive_progress_prints_only_start_and_finish() -> None:
    stream = StringIO()
    progress = _ProgressReporter("mock", stream=stream)

    progress.update(0, 24)
    progress.update(1, 24)
    progress.update(24, 24)

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert "0/24 (0%)" in lines[0]
    assert "24/24 (100%)" in lines[1]
