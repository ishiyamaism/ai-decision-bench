"""Provider protocol used by the evaluator."""

from __future__ import annotations

from typing import Protocol

from ai_decision_bench.models import DecisionResult, EvaluationCase


class DecisionProvider(Protocol):
    """A provider that normalizes one structured decision."""

    name: str
    model: str | None
    required_env_vars: tuple[str, ...]

    async def decide(self, case: EvaluationCase) -> DecisionResult: ...

    async def aclose(self) -> None: ...
