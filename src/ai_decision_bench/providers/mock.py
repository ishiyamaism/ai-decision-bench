"""Deterministic local provider for tests and smoke runs."""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar

from ai_decision_bench.models import DecisionResult, EvaluationCase, JsonScalar

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("bill", "charge", "charged", "invoice", "payment", "refund", "receipt"),
    "technical_support": (
        "bug",
        "crash",
        "error",
        "fails",
        "failing",
        "load",
        "sync",
        "upload",
    ),
    "account": ("account", "email", "login", "password", "profile", "sign in", "locked"),
    "sales": ("demo", "enterprise", "pricing", "quote", "trial", "upgrade", "volume"),
    "cancellation": ("cancel", "close my subscription", "end my subscription", "terminate"),
    "other": ("feedback", "hello", "press", "partnership", "suggestion"),
}


class MockProvider:
    """Keyword baseline. It is operational scaffolding, not an AI quality claim."""

    name = "mock"
    model = "keyword-baseline-v1"
    required_env_vars: tuple[str, ...] = ()
    benchmark_settings: ClassVar[dict[str, JsonScalar]] = {}

    async def decide(self, case: EvaluationCase) -> DecisionResult:
        started = time.perf_counter()
        await asyncio.sleep(0)
        text = " ".join(str(value) for value in case.input.values()).lower()
        scored = {
            label: sum(keyword in text for keyword in _KEYWORDS.get(label, (label,)))
            for label in case.task.labels
        }
        prediction = max(case.task.labels, key=lambda label: scored[label])
        max_score = scored[prediction]
        transactional_terms = ("charge", "invoice", "payment", "refund", "receipt")
        if scored.get("cancellation", 0) > 0 and not any(
            term in text for term in transactional_terms
        ):
            prediction = "cancellation"
            max_score = scored[prediction]
        if max_score == 0 and "other" in case.task.labels:
            prediction = "other"
        provider_confidence = 0.9 if max_score >= 1 else 0.55
        return DecisionResult(
            prediction=prediction,
            provider_confidence=provider_confidence,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_identifier=self.model,
        )

    async def aclose(self) -> None:
        return None
