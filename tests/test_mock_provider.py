from __future__ import annotations

import asyncio

from ai_decision_bench.models import ClassificationTask, EvaluationCase
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
