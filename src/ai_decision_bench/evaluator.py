"""Dataset loading and provider-independent asynchronous evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ai_decision_bench import __version__
from ai_decision_bench.metrics import calculate_metrics
from ai_decision_bench.models import (
    BenchmarkConfig,
    BenchmarkReport,
    CaseEvaluation,
    DecisionResult,
    EvaluationCase,
)
from ai_decision_bench.providers.base import DecisionProvider


class DatasetError(ValueError):
    """Raised when a JSONL dataset cannot be validated."""


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path) -> list[EvaluationCase]:
    """Load unique, validated cases from a UTF-8 JSONL file."""

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Could not read dataset: {path.name}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = EvaluationCase.model_validate_json(line)
        except (ValidationError, ValueError) as exc:
            raise DatasetError(f"Invalid dataset case on line {line_number}.") from exc
        if case.id in seen_ids:
            raise DatasetError(f"Duplicate case id on line {line_number}.")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError("Dataset contains no cases.")
    return cases


async def _evaluate_case(
    provider: DecisionProvider,
    case: EvaluationCase,
    semaphore: asyncio.Semaphore,
) -> CaseEvaluation:
    async with semaphore:
        started = time.perf_counter()
        try:
            result = await provider.decide(case)
        except Exception:  # Providers are isolated; exception text may contain credentials.
            result = DecisionResult(
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="provider_exception",
            )
    return CaseEvaluation(
        case_id=case.id,
        expected=case.expected,
        correct=result.error is None and result.prediction == case.expected,
        result=result,
    )


async def evaluate(
    provider: DecisionProvider,
    cases: list[EvaluationCase],
    *,
    concurrency: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[CaseEvaluation]:
    """Evaluate every case while preserving dataset order."""

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(cases)
    if progress_callback is not None:
        progress_callback(completed, total)

    async def evaluate_tracked(case: EvaluationCase) -> CaseEvaluation:
        nonlocal completed
        evaluation = await _evaluate_case(provider, case, semaphore)
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total)
        return evaluation

    return list(await asyncio.gather(*(evaluate_tracked(case) for case in cases)))


def safe_dataset_name(path: Path) -> str:
    """Keep result files free of machine-specific absolute paths."""

    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


async def run_benchmark(
    provider: DecisionProvider,
    dataset_path: Path,
    config: BenchmarkConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BenchmarkReport:
    try:
        cases = load_dataset(dataset_path)
        evaluations = await evaluate(
            provider,
            cases,
            concurrency=config.concurrency,
            progress_callback=progress_callback,
        )
    finally:
        await provider.aclose()
    identifiers = {
        item.result.model_identifier
        for item in evaluations
        if item.result.model_identifier is not None
    }
    model_identifier = next(iter(identifiers)) if len(identifiers) == 1 else None
    return BenchmarkReport(
        timestamp=datetime.now(UTC),
        provider=provider.name,
        model=provider.model,
        model_identifier=model_identifier,
        dataset=safe_dataset_name(dataset_path),
        dataset_sha256=dataset_sha256(dataset_path),
        benchmark_version=__version__,
        python_version=platform.python_version(),
        config=config,
        metrics=calculate_metrics(evaluations),
        cases=evaluations,
    )
